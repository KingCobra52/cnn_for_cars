"""Cache-key correctness.

The cache key is the project's main safety property: it must change whenever anything
that affects the embeddings changes. If it did not, a preprocessing edit would silently
reuse stale embeddings and every downstream number would be wrong in a way no test would
catch.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from carvision.data.transforms import CLIP_224, IMAGENET_224
from carvision.features.cache import compute_cache_key
from carvision.models.backbones import BackboneSpec, available_backbones, get_backbone


def make_spec(**overrides: object) -> BackboneSpec:
    base = {
        "name": "fake",
        "weights_tag": "v1",
        "embedding_dim": 8,
        "preprocess": IMAGENET_224,
        "factory": lambda: None,
    }
    return BackboneSpec(**{**base, **overrides})  # type: ignore[arg-type]


IDS = ["img_0", "img_1", "img_2"]


def test_key_is_deterministic() -> None:
    assert compute_cache_key(make_spec(), IDS) == compute_cache_key(make_spec(), IDS)


def test_changing_the_weights_tag_changes_the_key() -> None:
    """IMAGENET1K_V1 and V2 are different weights and must not share a cache."""
    a = compute_cache_key(make_spec(weights_tag="IMAGENET1K_V1"), IDS)
    b = compute_cache_key(make_spec(weights_tag="IMAGENET1K_V2"), IDS)
    assert a != b


def test_changing_preprocessing_changes_the_key() -> None:
    a = compute_cache_key(make_spec(preprocess=IMAGENET_224), IDS)
    b = compute_cache_key(make_spec(preprocess=CLIP_224), IDS)
    assert a != b


def test_changing_the_crop_size_changes_the_key() -> None:
    a = compute_cache_key(make_spec(), IDS)
    b = compute_cache_key(make_spec(preprocess=replace(IMAGENET_224, crop=196)), IDS)
    assert a != b


def test_changing_the_backbone_name_changes_the_key() -> None:
    a = compute_cache_key(make_spec(name="resnet50"), IDS)
    b = compute_cache_key(make_spec(name="dinov2_vits14"), IDS)
    assert a != b


def test_different_images_change_the_key() -> None:
    assert compute_cache_key(make_spec(), IDS) != compute_cache_key(make_spec(), [*IDS, "img_3"])


def test_image_order_changes_the_key() -> None:
    """Embeddings are stored row-wise, so order is part of the contract, not incidental."""
    assert compute_cache_key(make_spec(), IDS) != compute_cache_key(make_spec(), IDS[::-1])


def test_ids_cannot_be_confused_by_concatenation() -> None:
    """Hashing ids without a separator would make ['ab','c'] and ['a','bc'] collide."""
    assert compute_cache_key(make_spec(), ["ab", "c"]) != compute_cache_key(make_spec(), ["a", "bc"])


@pytest.mark.parametrize("name", available_backbones())
def test_registered_backbones_have_distinct_keys(name: str) -> None:
    spec = get_backbone(name)
    others = [get_backbone(other).cache_key() for other in available_backbones() if other != name]
    assert spec.cache_key() not in others


@pytest.mark.parametrize("name", available_backbones())
def test_registered_backbones_declare_a_positive_dim(name: str) -> None:
    assert get_backbone(name).embedding_dim > 0


def test_unknown_backbone_lists_the_available_ones() -> None:
    with pytest.raises(KeyError, match="resnet50"):
        get_backbone("resnet18")
