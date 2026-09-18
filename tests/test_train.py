"""Training-loop behaviour, exercised on synthetic embeddings.

No cache, no dataset, no network: the heads are trained directly on the separable blobs
from ``synthetic_embeddings``, which a working classifier must solve almost perfectly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from carvision.models.heads import MLPHead, TemperatureScaler, build_head
from carvision.train import cosine_lr, evaluate
from carvision.utils.seed import set_seed


# ------------------------------------------------------------------ heads


@pytest.mark.parametrize("kind", ["linear", "mlp"])
def test_head_output_shape(kind: str) -> None:
    head = build_head(kind, embedding_dim=16, num_classes=10)
    assert head(torch.randn(4, 16)).shape == (4, 10)


def test_unknown_head_kind_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown head kind"):
        build_head("transformer", 16, 10)


@pytest.mark.parametrize("kind", ["linear", "mlp"])
def test_head_fits_separable_data(
    kind: str, synthetic_embeddings: tuple[np.ndarray, np.ndarray]
) -> None:
    """A canary: if a head cannot solve well-separated blobs, the head is broken."""
    set_seed(0)
    features_np, labels_np = synthetic_embeddings
    features = torch.from_numpy(features_np)
    labels = torch.from_numpy(labels_np)

    head = build_head(kind, features.shape[1], int(labels.max()) + 1)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    head.train()
    for _ in range(300):
        optimizer.zero_grad()
        criterion(head(features), labels).backward()
        optimizer.step()

    _, accuracy = evaluate(head, features, labels, criterion)
    assert accuracy > 0.95


def test_mlp_dropout_is_disabled_in_eval() -> None:
    """Dropout left active at eval makes predictions nondeterministic."""
    head = MLPHead(16, 10, dropout=0.5)
    head.eval()
    features = torch.randn(8, 16)
    torch.testing.assert_close(head(features), head(features))


def test_training_is_reproducible_under_a_fixed_seed(
    synthetic_embeddings: tuple[np.ndarray, np.ndarray],
) -> None:
    features = torch.from_numpy(synthetic_embeddings[0])
    labels = torch.from_numpy(synthetic_embeddings[1])

    def run() -> torch.Tensor:
        set_seed(11)
        head = build_head("mlp", features.shape[1], int(labels.max()) + 1)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-2)
        criterion = nn.CrossEntropyLoss()
        head.train()
        for _ in range(20):
            optimizer.zero_grad()
            criterion(head(features), labels).backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            return head(features)

    torch.testing.assert_close(run(), run())


# ------------------------------------------------------------------ schedule


def test_warmup_rises_then_cosine_decays() -> None:
    values = [cosine_lr(step, total=20, warmup=5, base_lr=1.0) for step in range(20)]
    assert values[:5] == sorted(values[:5]), "warmup must be non-decreasing"
    assert values[5:] == sorted(values[5:], reverse=True), "post-warmup must decay"
    assert values[4] == pytest.approx(1.0), "peak LR reached at end of warmup"


def test_first_warmup_step_is_nonzero() -> None:
    """A zero first step wastes an epoch doing nothing."""
    assert cosine_lr(0, total=20, warmup=5, base_lr=1.0) > 0


def test_schedule_ends_near_zero() -> None:
    assert cosine_lr(19, total=20, warmup=5, base_lr=1.0) < 0.02


def test_no_warmup_is_supported() -> None:
    assert cosine_lr(0, total=10, warmup=0, base_lr=1.0) == pytest.approx(1.0)


# ------------------------------------------------------------------ calibration


def test_temperature_scaling_does_not_change_predictions() -> None:
    """Temperature divides logits, so the argmax -- and thus accuracy -- is invariant."""
    logits = torch.randn(64, 10)
    scaler = TemperatureScaler(2.5)
    torch.testing.assert_close(scaler(logits).argmax(1), logits.argmax(1))


def test_temperature_scaling_softens_overconfident_logits() -> None:
    """Fitting on deliberately overconfident logits must yield a temperature above one."""
    set_seed(0)
    labels = torch.randint(0, 5, (500,))
    # Correct 70% of the time, but with a huge margin: classic overconfidence.
    logits = torch.randn(500, 5) * 0.1
    correct = torch.rand(500) < 0.7
    logits[correct, labels[correct]] += 12.0
    wrong = ~correct
    logits[wrong, (labels[wrong] + 1) % 5] += 12.0

    temperature = TemperatureScaler().fit(logits, labels)
    assert temperature > 1.0
