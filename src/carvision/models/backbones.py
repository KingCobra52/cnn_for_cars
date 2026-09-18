"""Frozen pretrained backbones, behind one interface.

Every backbone here is used purely as a feature extractor: weights are never updated.
That is what makes this project tractable on a CPU. A backbone contributes three things
-- a module, the preprocessing its weights expect, and an embedding dimensionality --
and the registry keeps those three inseparable.

The legacy notebook called ``models.vgg16(pretrained=True)``, deprecated since
torchvision 0.13 and warning on every run. Nothing here uses the ``pretrained`` argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch
from torch import nn

from carvision.data.transforms import CLIP_224, DINOV2_224, IMAGENET_224, PreprocessSpec
from carvision.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


class BackboneFactory(Protocol):
    """Builds a frozen feature extractor."""

    def __call__(self) -> nn.Module:
        """Return the module, already in eval mode with gradients disabled."""
        ...


@dataclass(frozen=True)
class BackboneSpec:
    """Everything needed to produce and identify one backbone's embeddings.

    Attributes:
        name: Registry key, also used in the embedding cache path.
        weights_tag: Identifies the exact pretrained weights. Part of the cache key, so
            swapping ``IMAGENET1K_V1`` for ``V2`` invalidates cached embeddings.
        embedding_dim: Width of the pooled feature vector.
        preprocess: The preprocessing these weights expect.
        factory: Callable building the module.
        supports_zeroshot: Whether the backbone has a text tower for a zero-shot
            baseline.
        gradcam_layer: Dotted attribute path to a spatial feature map suitable for
            Grad-CAM, or None for architectures where attention rollout is used instead.
    """

    name: str
    weights_tag: str
    embedding_dim: int
    preprocess: PreprocessSpec
    factory: BackboneFactory
    supports_zeroshot: bool = False
    gradcam_layer: str | None = None

    def cache_key(self) -> str:
        """Return a stable string identifying this backbone and its preprocessing."""
        return f"{self.name}@{self.weights_tag}|{self.preprocess.cache_key()}"

    def build(self) -> nn.Module:
        """Instantiate the module, frozen and in eval mode."""
        module = self.factory()
        return freeze(module)


def freeze(module: nn.Module) -> nn.Module:
    """Put a module in eval mode and disable gradients for all its parameters.

    Both halves matter. ``eval()`` alone still tracks gradients and wastes memory;
    ``requires_grad_(False)`` alone leaves BatchNorm updating its running statistics from
    the data it sees, which silently makes embeddings depend on batch order.

    Args:
        module: The module to freeze, modified in place.

    Returns:
        The same module, for chaining.
    """
    module.eval()
    module.requires_grad_(False)
    return module


# --------------------------------------------------------------------- factories


def _resnet50() -> nn.Module:
    """ResNet-50 with its classifier removed, pooled to a 2048-d vector."""
    from torchvision.models import ResNet50_Weights, resnet50

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    # Identity in place of the 1000-way fc turns the model into a feature extractor
    # while leaving the global average pool intact.
    model.fc = nn.Identity()
    return model


def _clip_vitb32() -> nn.Module:
    """OpenAI CLIP ViT-B/32 image tower, returning 512-d embeddings."""
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    return _CLIPImageTower(model)


class _CLIPImageTower(nn.Module):
    """Adapter exposing CLIP's image encoder as a plain ``forward``."""

    def __init__(self, clip_model: nn.Module) -> None:
        super().__init__()
        self.clip = clip_model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images.

        Args:
            images: ``(B, 3, H, W)`` normalised with CLIP statistics.

        Returns:
            ``(B, 512)`` image embeddings, unnormalised. L2 normalisation is applied by
            the consumer that needs it (zero-shot does, a linear probe does not have to).
        """
        return self.clip.encode_image(images)


def _dinov2_vits14() -> nn.Module:
    """DINOv2 ViT-S/14, returning the 384-d CLS embedding."""
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
    assert isinstance(model, nn.Module)
    return model


# --------------------------------------------------------------------- registry

REGISTRY: dict[str, BackboneSpec] = {
    "resnet50": BackboneSpec(
        name="resnet50",
        weights_tag="IMAGENET1K_V2",
        embedding_dim=2048,
        preprocess=IMAGENET_224,
        factory=_resnet50,
        gradcam_layer="layer4",
    ),
    "clip_vitb32": BackboneSpec(
        name="clip_vitb32",
        weights_tag="openai",
        embedding_dim=512,
        preprocess=CLIP_224,
        factory=_clip_vitb32,
        supports_zeroshot=True,
    ),
    "dinov2_vits14": BackboneSpec(
        name="dinov2_vits14",
        weights_tag="facebookresearch/dinov2",
        embedding_dim=384,
        preprocess=DINOV2_224,
        factory=_dinov2_vits14,
    ),
}


def get_backbone(name: str) -> BackboneSpec:
    """Look up a backbone specification by name.

    Args:
        name: A key of :data:`REGISTRY`.

    Returns:
        The specification.

    Raises:
        KeyError: If the name is not registered, listing what is.
    """
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown backbone {name!r}. Available: {sorted(REGISTRY)}"
        ) from None


def available_backbones() -> list[str]:
    """Return the registered backbone names, sorted."""
    return sorted(REGISTRY)


@torch.inference_mode()
def embed_batch(module: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Run one batch through a frozen backbone.

    Args:
        module: A frozen backbone from :meth:`BackboneSpec.build`.
        images: ``(B, 3, H, W)`` preprocessed batch.

    Returns:
        ``(B, D)`` float32 embeddings on the CPU.

    Raises:
        ValueError: If the backbone returns something that is not a 2-D tensor, which
            usually means the wrong adapter was used for the architecture.
    """
    output = module(images)
    if not isinstance(output, torch.Tensor) or output.ndim != 2:
        shape = getattr(output, "shape", type(output).__name__)
        raise ValueError(
            f"Backbone returned {shape}; expected a 2-D (B, D) tensor. The module likely "
            f"needs an adapter that pools or selects the CLS token."
        )
    return output.float().cpu()


def get_transform(name: str, *, train: bool = False) -> Callable[..., torch.Tensor]:
    """Return the preprocessing transform belonging to a backbone.

    Args:
        name: A registered backbone name.
        train: Whether to include light augmentation.

    Returns:
        The composed transform.
    """
    return get_backbone(name).preprocess.build(train=train)
