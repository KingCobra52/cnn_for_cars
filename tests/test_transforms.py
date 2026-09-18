"""Preprocessing specs must be hashable, distinct, and correctly shaped."""

from __future__ import annotations

import pytest

from carvision.data.transforms import (
    CLIP_224,
    CLIP_MEAN,
    DINOV2_224,
    IMAGENET_224,
    IMAGENET_MEAN,
    PreprocessSpec,
)


def test_cache_key_is_stable() -> None:
    assert IMAGENET_224.cache_key() == IMAGENET_224.cache_key()


def test_clip_and_imagenet_keys_differ() -> None:
    """The legacy bug in one assertion: CLIP and ImageNet normalisation are not the same.

    If these two ever hashed alike, the embedding cache would serve CLIP embeddings for
    an ImageNet backbone.
    """
    assert CLIP_MEAN != IMAGENET_MEAN
    assert CLIP_224.cache_key() != IMAGENET_224.cache_key()


def test_changing_any_field_changes_the_key() -> None:
    base = PreprocessSpec(resize=256, crop=224, mean=IMAGENET_MEAN, std=(0.2, 0.2, 0.2))
    variants = [
        PreprocessSpec(resize=232, crop=224, mean=IMAGENET_MEAN, std=(0.2, 0.2, 0.2)),
        PreprocessSpec(resize=256, crop=196, mean=IMAGENET_MEAN, std=(0.2, 0.2, 0.2)),
        PreprocessSpec(resize=256, crop=224, mean=CLIP_MEAN, std=(0.2, 0.2, 0.2)),
        PreprocessSpec(resize=256, crop=224, mean=IMAGENET_MEAN, std=(0.3, 0.2, 0.2)),
        PreprocessSpec(
            resize=256, crop=224, mean=IMAGENET_MEAN, std=(0.2, 0.2, 0.2), interpolation="bilinear"
        ),
    ]
    for variant in variants:
        assert variant.cache_key() != base.cache_key()


def test_dinov2_crop_is_a_multiple_of_the_patch_size() -> None:
    """DINOv2 ViT-S/14 tokenises in 14-pixel patches; a bad crop silently truncates."""
    assert DINOV2_224.crop % 14 == 0


@pytest.mark.parametrize("spec", [IMAGENET_224, CLIP_224, DINOV2_224])
def test_eval_transform_output_shape(spec: PreprocessSpec) -> None:
    torch = pytest.importorskip("torch")
    from PIL import Image

    image = Image.new("RGB", (400, 300), color=(128, 64, 32))
    tensor = spec.build(train=False)(image)

    assert tensor.shape == (3, spec.crop, spec.crop)
    assert tensor.dtype == torch.float32


def test_train_transform_also_yields_the_crop_size() -> None:
    pytest.importorskip("torch")
    from PIL import Image

    image = Image.new("RGB", (400, 300))
    tensor = IMAGENET_224.build(train=True)(image)
    assert tensor.shape == (3, 224, 224)
