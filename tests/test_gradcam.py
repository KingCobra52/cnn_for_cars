"""Grad-CAM mechanics, on a tiny synthetic CNN.

No pretrained weights and no images: the point is to verify the machinery -- layer
resolution, hook lifecycle, gradient flow, normalisation -- not to judge saliency quality,
which is a human call made on real photographs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from carvision.interpret.gradcam import GradCAM, GradCAMError, resolve_layer
from carvision.models.backbones import freeze


class TinyCNN(nn.Module):
    """Two conv blocks, global pool, linear classifier."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.block2 = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block2(self.block1(x))
        return self.fc(self.pool(x).flatten(1))


@pytest.fixture
def model() -> TinyCNN:
    torch.manual_seed(0)
    return TinyCNN()


@pytest.fixture
def images() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(3, 3, 32, 32)


# ------------------------------------------------------------------ layer resolution


def test_resolve_a_top_level_layer(model: TinyCNN) -> None:
    assert resolve_layer(model, "block1") is model.block1


def test_resolve_indexes_into_sequential(model: TinyCNN) -> None:
    """Paths like 'features.28' are how torchvision models name conv layers."""
    assert resolve_layer(model, "block1.0") is model.block1[0]


def test_resolve_reports_a_bad_path(model: TinyCNN) -> None:
    with pytest.raises(GradCAMError, match="layer99"):
        resolve_layer(model, "layer99")


# ------------------------------------------------------------------ maps


def test_map_shape_matches_the_layer(model: TinyCNN, images: torch.Tensor) -> None:
    """block2 halves the 32px input twice, so its feature map is 8x8."""
    cam = GradCAM(model, "block2")(images)
    assert cam.shape == (3, 8, 8)


def test_map_is_normalised(model: TinyCNN, images: torch.Tensor) -> None:
    cam = GradCAM(model, "block2")(images)
    assert cam.min() >= 0.0
    assert cam.max() <= 1.0 + 1e-6
    assert np.isfinite(cam).all()


def test_each_map_reaches_one(model: TinyCNN, images: torch.Tensor) -> None:
    """Per-image normalisation: every non-degenerate map peaks at 1."""
    cam = GradCAM(model, "block2")(images)
    for single in cam:
        assert single.max() == pytest.approx(1.0)


def test_different_targets_give_different_maps(model: TinyCNN, images: torch.Tensor) -> None:
    gradcam = GradCAM(model, "block2")
    assert not np.allclose(gradcam(images, target=0), gradcam(images, target=1))


def test_per_image_targets_are_supported(model: TinyCNN, images: torch.Tensor) -> None:
    cam = GradCAM(model, "block2")(images, target=torch.tensor([0, 1, 2]))
    assert cam.shape == (3, 8, 8)


def test_default_target_is_the_argmax(model: TinyCNN, images: torch.Tensor) -> None:
    gradcam = GradCAM(model, "block2")
    with torch.no_grad():
        predicted = model(images).argmax(dim=1)
    np.testing.assert_allclose(gradcam(images), gradcam(images, target=predicted), atol=1e-6)


# ------------------------------------------------------------------ side effects


def test_hook_is_removed(model: TinyCNN, images: torch.Tensor) -> None:
    """A leaked forward hook would slow and eventually corrupt every later call."""
    before = len(model.block2._forward_hooks)
    GradCAM(model, "block2")(images)
    assert len(model.block2._forward_hooks) == before


def test_frozen_model_stays_frozen(model: TinyCNN, images: torch.Tensor) -> None:
    """Grad-CAM needs gradients, but must hand the model back exactly as it found it."""
    freeze(model)
    assert not any(p.requires_grad for p in model.parameters())

    GradCAM(model, "block2")(images)

    assert not any(p.requires_grad for p in model.parameters()), "requires_grad leaked"


def test_running_twice_gives_the_same_map(model: TinyCNN, images: torch.Tensor) -> None:
    gradcam = GradCAM(model, "block2")
    np.testing.assert_allclose(gradcam(images, target=0), gradcam(images, target=0))


# ------------------------------------------------------------------ head path


def test_explaining_through_a_separate_head(images: torch.Tensor) -> None:
    """The cached-embedding setup: backbone emits features, a separate head classifies."""
    torch.manual_seed(0)
    backbone = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten()
    )
    head = nn.Linear(8, 5)

    cam = GradCAM(backbone, "0")(images, target=2, head=head)
    assert cam.shape == (3, 32, 32)
    assert np.isfinite(cam).all()
