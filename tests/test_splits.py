"""Split determinism, stratification, and the absence of leakage."""

from __future__ import annotations

import numpy as np
import pytest

from carvision.data.splits import (
    SplitError,
    assert_disjoint,
    stratified_val_split,
)


def test_split_is_deterministic(synthetic_labels: np.ndarray) -> None:
    first = stratified_val_split(synthetic_labels, seed=42)
    second = stratified_val_split(synthetic_labels, seed=42)
    np.testing.assert_array_equal(first, second)


def test_different_seed_gives_different_split(synthetic_labels: np.ndarray) -> None:
    first = stratified_val_split(synthetic_labels, seed=1)
    second = stratified_val_split(synthetic_labels, seed=2)
    assert not np.array_equal(first, second)


def test_per_class_val_counts_are_independent_of_row_order(
    synthetic_labels: np.ndarray,
) -> None:
    """How many images each class contributes to val depends only on its size.

    Which specific images get picked does depend on their positions -- that is inherent
    to selecting by index -- but the per-class budget must not, or a reshuffled manifest
    would silently rebalance the validation set.
    """
    shuffle = np.random.default_rng(0).permutation(len(synthetic_labels))

    first = stratified_val_split(synthetic_labels, seed=5)
    second = stratified_val_split(synthetic_labels[shuffle], seed=5)

    for label in np.unique(synthetic_labels):
        assert first[synthetic_labels == label].sum() == second[
            synthetic_labels[shuffle] == label
        ].sum()


def test_every_class_appears_in_both_train_and_val(synthetic_labels: np.ndarray) -> None:
    """Stratification's whole point: no class may be absent from validation."""
    mask = stratified_val_split(synthetic_labels, val_fraction=0.15, seed=0)
    for label in np.unique(synthetic_labels):
        in_class = synthetic_labels == label
        assert mask[in_class].sum() >= 1, f"class {label} has no validation image"
        assert (~mask[in_class]).sum() >= 1, f"class {label} has no training image"


def test_val_fraction_is_approximately_honoured(synthetic_labels: np.ndarray) -> None:
    mask = stratified_val_split(synthetic_labels, val_fraction=0.2, seed=0)
    # The per-class minimum of one image pulls the realised fraction above the target
    # for small classes, so allow generous slack in that direction only.
    assert 0.2 <= mask.mean() <= 0.35


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_invalid_fraction_rejected(synthetic_labels: np.ndarray, fraction: float) -> None:
    with pytest.raises(ValueError, match="val_fraction"):
        stratified_val_split(synthetic_labels, val_fraction=fraction)


def test_assert_disjoint_detects_leakage() -> None:
    pd = pytest.importorskip("pandas")
    frames = {
        "train": pd.DataFrame({"image_id": ["a", "b", "c"]}),
        "test": pd.DataFrame({"image_id": ["c", "d"]}),
    }
    with pytest.raises(SplitError, match="both"):
        assert_disjoint(frames)


def test_assert_disjoint_passes_on_clean_splits() -> None:
    pd = pytest.importorskip("pandas")
    frames = {
        "train": pd.DataFrame({"image_id": ["a", "b"]}),
        "val": pd.DataFrame({"image_id": ["c"]}),
        "test": pd.DataFrame({"image_id": ["d", "e"]}),
    }
    assert_disjoint(frames)
