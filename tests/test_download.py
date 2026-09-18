"""Mirror resolution and schema verification.

No network: the Hub response is stood in for by tiny fakes. The point is that a mirror
whose schema has drifted fails immediately with an actionable message, rather than
producing embeddings from the wrong column.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from carvision.data.download import (
    DEFAULT_MIRROR,
    EXPECTED_NUM_CLASSES,
    DatasetAcquisitionError,
    resolve_mirror,
    verify_mirror,
)


@dataclass
class FakeClassLabel:
    names: list[str]


class FakeSplit:
    def __init__(self, features: dict[str, Any]) -> None:
        self.features = features


def make_dataset_dict(
    *,
    splits: tuple[str, ...] = ("train", "test"),
    columns: tuple[str, ...] = ("image", "label"),
    num_classes: int = EXPECTED_NUM_CLASSES,
    label_is_classlabel: bool = True,
) -> dict[str, FakeSplit]:
    label = FakeClassLabel([f"class_{i}" for i in range(num_classes)])
    features: dict[str, Any] = {name: object() for name in columns}
    if "label" in columns:
        features["label"] = label if label_is_classlabel else object()
    return dict.fromkeys(splits, FakeSplit(features))  # type: ignore[arg-type]


def test_default_mirror_is_known() -> None:
    spec = resolve_mirror()
    assert spec.repo_id == DEFAULT_MIRROR


def test_unknown_mirror_is_allowed_with_default_columns() -> None:
    """An unrecognised mirror is not rejected outright -- the schema check decides."""
    spec = resolve_mirror("someone/their-cars-mirror")
    assert spec.repo_id == "someone/their-cars-mirror"
    assert spec.image_column == "image"
    assert "Unverified" in spec.note


def test_overrides_are_applied() -> None:
    spec = resolve_mirror("someone/mirror", image_column="img", test_split="validation")
    assert spec.image_column == "img"
    assert spec.test_split == "validation"
    assert spec.label_column == "label"


def test_verify_accepts_a_well_formed_mirror() -> None:
    verify_mirror(make_dataset_dict(), resolve_mirror())


def test_missing_split_is_reported_with_what_was_found() -> None:
    dataset = make_dataset_dict(splits=("train", "validation"))
    with pytest.raises(DatasetAcquisitionError, match="validation"):
        verify_mirror(dataset, resolve_mirror())


def test_missing_column_is_reported() -> None:
    dataset = make_dataset_dict(columns=("img", "label"))
    with pytest.raises(DatasetAcquisitionError, match="image-column"):
        verify_mirror(dataset, resolve_mirror())


def test_non_classlabel_label_is_rejected() -> None:
    """Without a ClassLabel there are no class names, so zero-shot could not work."""
    dataset = make_dataset_dict(label_is_classlabel=False)
    with pytest.raises(DatasetAcquisitionError, match="ClassLabel"):
        verify_mirror(dataset, resolve_mirror())


def test_wrong_class_count_is_rejected() -> None:
    """A 10-class mirror is some other dataset, however plausibly it is named."""
    dataset = make_dataset_dict(num_classes=10)
    with pytest.raises(DatasetAcquisitionError, match="196"):
        verify_mirror(dataset, resolve_mirror())
