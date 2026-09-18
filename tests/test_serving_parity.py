"""The deployed preprocessing must match the training preprocessing.

``app/serving.py`` reimplements preprocessing in pure numpy and PIL so the Space and the
Docker image need no torch. Duplicated logic drifts, and preprocessing drift is the exact
failure this project exists to have fixed -- so the duplication is pinned by test rather
than by good intentions.

If these fail, the deployed model is being fed something different from what it was
trained on, and the demo's accuracy will not match the reported accuracy.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

import serving  # noqa: E402
from carvision.data.transforms import (  # noqa: E402
    CLIP_224,
    DINOV2_224,
    IMAGENET_224,
    RESNET50_V2,
    PreprocessSpec,
)


def make_image(width: int, height: int, seed: int = 0) -> Image.Image:
    """A deterministic noisy RGB image of the given size."""
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


# ------------------------------------------------------------------ independence


def test_serving_module_imports_no_torch() -> None:
    """The whole point of the duplication: the serving path must stay torch-free.

    Checked by parsing the imports rather than by grepping, so a comment mentioning
    torch does not fail the test and a real import cannot hide in a string.
    """
    tree = ast.parse((APP_DIR / "serving.py").read_text())

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    forbidden = {"torch", "torchvision", "carvision"}
    assert not (imported & forbidden), f"serving.py must not import {imported & forbidden}"


# ------------------------------------------------------------------ parity


@pytest.mark.parametrize("spec", [IMAGENET_224, CLIP_224, DINOV2_224, RESNET50_V2])
@pytest.mark.parametrize("size", [(400, 300), (300, 400), (224, 224), (1000, 640)])
def test_preprocessing_matches_torchvision(spec: PreprocessSpec, size: tuple[int, int]) -> None:
    """Both implementations must produce the same tensor, for every backbone and shape.

    Both landscape and portrait inputs are covered because the shorter-side resize
    branches on aspect ratio -- an easy place for a reimplementation to get it backwards.
    """
    image = make_image(*size)

    reference = spec.build(train=False)(image).numpy()[None]
    actual = serving.preprocess(
        image,
        resize=spec.resize,
        crop=spec.crop,
        mean=spec.mean,
        std=spec.std,
        interpolation=spec.interpolation,
    )

    assert actual.shape == reference.shape
    # Both resize through PIL bicubic, so any difference is float rounding, not method.
    np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-4)


def test_output_is_float32_nchw() -> None:
    """ONNX Runtime is strict about dtype; float64 would be rejected at the session."""
    batch = serving.preprocess(
        make_image(320, 240), resize=256, crop=224, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
    )
    assert batch.dtype == np.float32
    assert batch.shape == (1, 3, 224, 224)
    assert batch.flags["C_CONTIGUOUS"]


def test_greyscale_input_is_accepted() -> None:
    """Some real uploads are single-channel; the model needs three."""
    grey = Image.new("L", (300, 300), color=128)
    batch = serving.preprocess(
        grey, resize=256, crop=224, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
    )
    assert batch.shape == (1, 3, 224, 224)


# ------------------------------------------------------------------ pieces


@pytest.mark.parametrize(
    ("size", "target", "expected"),
    [
        ((400, 300), 150, (200, 150)),  # landscape: height is shorter
        ((300, 400), 150, (150, 200)),  # portrait: width is shorter
        ((200, 200), 100, (100, 100)),  # square
    ],
)
def test_resize_shorter_side(size: tuple[int, int], target: int, expected: tuple[int, int]) -> None:
    assert serving.resize_shorter_side(make_image(*size), target).size == expected


def test_center_crop_is_centred() -> None:
    """Mark the exact centre and check it survives the crop."""
    array = np.zeros((100, 100, 3), dtype=np.uint8)
    array[50, 50] = [255, 0, 0]
    cropped = np.asarray(serving.center_crop(Image.fromarray(array), 20))

    assert cropped.shape == (20, 20, 3)
    assert tuple(cropped[10, 10]) == (255, 0, 0)


def test_softmax_matches_a_direct_computation() -> None:
    scores = np.array([1.0, 2.0, 3.0])
    expected = np.exp(scores) / np.exp(scores).sum()
    np.testing.assert_allclose(serving.softmax(scores), expected)


def test_softmax_is_stable_on_huge_scores() -> None:
    result = serving.softmax(np.array([1000.0, 1001.0]))
    assert np.isfinite(result).all()
    assert result.sum() == pytest.approx(1.0)
