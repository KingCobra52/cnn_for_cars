"""Classifier heads trained on cached embeddings.

These are the only parameters this project trains. Everything upstream is frozen, so a
head sees a fixed ``(N, D)`` matrix and fits a 196-way classifier over it -- seconds of
work on a CPU.

A linear probe is the standard instrument for measuring how good a representation is:
because it can only draw hyperplanes, its accuracy reflects the backbone's features
rather than the head's capacity. The MLP is included to show what a little head capacity
buys, which on strong self-supervised features is usually not much.
"""

from __future__ import annotations

import torch
from torch import nn


class LinearHead(nn.Module):
    """A single linear layer over cached embeddings.

    Optionally preceded by feature standardisation, which matters because backbones
    differ by an order of magnitude in activation scale and a shared learning rate would
    otherwise suit only one of them.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        *,
        dropout: float = 0.0,
        normalize_features: bool = True,
    ) -> None:
        """Initialise the head.

        Args:
            embedding_dim: Width of the input embeddings.
            num_classes: Number of output classes.
            dropout: Dropout applied to the input features.
            normalize_features: Insert a non-affine LayerNorm before the linear layer.
        """
        super().__init__()
        self.norm = (
            nn.LayerNorm(embedding_dim, elementwise_affine=False)
            if normalize_features
            else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map ``(B, D)`` embeddings to ``(B, num_classes)`` logits."""
        logits: torch.Tensor = self.fc(self.dropout(self.norm(features)))
        return logits


class MLPHead(nn.Module):
    """One hidden layer with GELU, dropout and normalisation."""

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        *,
        hidden_dim: int = 1024,
        dropout: float = 0.3,
        normalize_features: bool = True,
    ) -> None:
        """Initialise the head.

        Args:
            embedding_dim: Width of the input embeddings.
            num_classes: Number of output classes.
            hidden_dim: Width of the hidden layer.
            dropout: Dropout applied after the hidden activation.
            normalize_features: Insert a non-affine LayerNorm on the input.
        """
        super().__init__()
        self.norm = (
            nn.LayerNorm(embedding_dim, elementwise_affine=False)
            if normalize_features
            else nn.Identity()
        )
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map ``(B, D)`` embeddings to ``(B, num_classes)`` logits."""
        logits: torch.Tensor = self.net(self.norm(features))
        return logits


def build_head(
    kind: str,
    embedding_dim: int,
    num_classes: int,
    **kwargs: float | int | bool,
) -> nn.Module:
    """Construct a head by name.

    Args:
        kind: ``linear`` or ``mlp``.
        embedding_dim: Width of the input embeddings.
        num_classes: Number of output classes.
        **kwargs: Passed through to the head constructor.

    Returns:
        The head module.

    Raises:
        ValueError: If ``kind`` is unknown.
    """
    if kind == "linear":
        return LinearHead(embedding_dim, num_classes, **kwargs)  # type: ignore[arg-type]
    if kind == "mlp":
        return MLPHead(embedding_dim, num_classes, **kwargs)  # type: ignore[arg-type]
    raise ValueError(f"Unknown head kind {kind!r}; expected 'linear' or 'mlp'.")


class TemperatureScaler(nn.Module):
    """A single scalar dividing the logits, fit on validation data.

    Modern classifiers are systematically overconfident: a model reporting 95%
    confidence is right rather less than 95% of the time. Dividing the logits by one
    learned temperature fixes most of that without touching accuracy at all -- the argmax
    is unchanged, only the probabilities move. It is the cheapest genuinely useful thing
    that can be done to a trained classifier, which is why the demo reports calibrated
    probabilities rather than raw softmax outputs.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        """Initialise with a starting temperature (1.0 is a no-op)."""
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(float(temperature)).log())

    @property
    def temperature(self) -> float:
        """The current temperature as a plain float."""
        return float(self.log_temperature.exp().item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Divide logits by the temperature.

        Parameterised in log space so the temperature cannot go negative or hit zero
        during optimisation.
        """
        scaled: torch.Tensor = logits / self.log_temperature.exp()
        return scaled

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, *, max_iter: int = 100) -> float:
        """Fit the temperature by minimising validation NLL.

        Args:
            logits: ``(N, C)`` uncalibrated validation logits.
            labels: ``(N,)`` true classes.
            max_iter: LBFGS iteration budget.

        Returns:
            The fitted temperature.
        """
        logits, labels = logits.detach(), labels.detach()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=0.01, max_iter=max_iter)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss: torch.Tensor = criterion(self.forward(logits), labels)
            loss.backward()  # type: ignore[no-untyped-call]
            return loss

        optimizer.step(closure)  # type: ignore[no-untyped-call]
        return self.temperature
