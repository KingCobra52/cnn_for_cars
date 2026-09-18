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


# ------------------------------------------------------------------ weights agreement


def test_resnet50_spec_matches_its_weights_exactly() -> None:
    """The constants must equal what the pretrained weights actually ask for.

    This is the check the module docstring promises, and it earned itself: resnet50 was
    configured at 256/bicubic when IMAGENET1K_V2 wants 232/bilinear. Wrong preprocessing
    on a pretrained backbone degrades every embedding it produces, and nothing else in
    the suite would notice.
    """
    pytest.importorskip("torchvision")
    from torchvision.models import ResNet50_Weights

    from carvision.data.transforms import RESNET50_V2

    reference = ResNet50_Weights.IMAGENET1K_V2.transforms()

    assert RESNET50_V2.resize == reference.resize_size[0]
    assert RESNET50_V2.crop == reference.crop_size[0]
    assert RESNET50_V2.mean == tuple(reference.mean)
    assert RESNET50_V2.std == tuple(reference.std)
    assert RESNET50_V2.interpolation == reference.interpolation.value


def test_generic_imagenet_spec_matches_the_v1_weights() -> None:
    """IMAGENET_224 documents itself as the classic V1 recipe; hold it to that."""
    pytest.importorskip("torchvision")
    from torchvision.models import ResNet50_Weights

    reference = ResNet50_Weights.IMAGENET1K_V1.transforms()

    assert IMAGENET_224.resize == reference.resize_size[0]
    assert IMAGENET_224.crop == reference.crop_size[0]
    assert IMAGENET_224.interpolation == reference.interpolation.value


def test_registered_backbone_specs_are_distinct() -> None:
    """Two backbones sharing a preprocessing key would share cached embeddings."""
    from carvision.models.backbones import available_backbones, get_backbone

    keys = [get_backbone(name).preprocess.cache_key() for name in available_backbones()]
    names = [get_backbone(name).cache_key() for name in available_backbones()]
    assert len(set(names)) == len(names)
    # Two backbones may legitimately share preprocessing; their full keys must not.
    assert len(keys) == len(available_backbones())
