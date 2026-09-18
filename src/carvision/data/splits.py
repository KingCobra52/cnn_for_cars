"""Deterministic train / validation / test splits.

The legacy project had no validation set: the test set was the only held-out data and it
was used both for tuning and for the headline number. Here the official Stanford Cars
test split is untouched, and a stratified validation set is carved out of train with a
fixed seed.

The resulting CSVs are small (three columns, ~16k rows total) and are **committed**, so
anyone cloning the repository evaluates on exactly the same images without re-running
anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from carvision.utils.logging import get_logger
from carvision.utils.paths import data_dir, ensure_dir, splits_dir

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

SPLIT_NAMES = ("train", "val", "test")
DEFAULT_VAL_FRACTION = 0.15
DEFAULT_SPLIT_SEED = 20260918


class SplitError(RuntimeError):
    """Raised when splits are missing, inconsistent, or overlapping."""


@dataclass(frozen=True)
class SplitSizes:
    """Image counts per split."""

    train: int
    val: int
    test: int

    @property
    def total(self) -> int:
        """Total number of images across all three splits."""
        return self.train + self.val + self.test


def stratified_val_split(
    labels: np.ndarray,
    *,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
) -> np.ndarray:
    """Choose a stratified validation subset of a label array.

    Stratification matters here: Stanford Cars has 196 classes with roughly 40 training
    images each, so an unstratified 15% draw would leave some classes with no validation
    examples at all. Every class contributes at least one image, and never all of them.

    Args:
        labels: Integer class ids, one per training image.
        val_fraction: Target fraction of the training set to hold out.
        seed: Seed for the permutation within each class.

    Returns:
        A boolean mask, true where the image belongs to validation.

    Raises:
        ValueError: If ``val_fraction`` is not strictly between 0 and 1.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    rng = np.random.default_rng(seed)
    mask = np.zeros(len(labels), dtype=bool)

    for label in np.unique(labels):
        (indices,) = np.nonzero(labels == label)
        # Sort first so the selection is a deterministic function of the class's
        # index set and the seed alone, independent of how np.nonzero happened to
        # order them.
        indices = np.sort(indices)
        permuted = rng.permutation(indices)

        # At least one validation image per class, and at least one left in train.
        n_val = round(len(indices) * val_fraction)
        n_val = max(1, min(n_val, len(indices) - 1))
        mask[permuted[:n_val]] = True

    return mask


def build_splits(
    *,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
    overwrite: bool = False,
) -> SplitSizes:
    """Write ``data/splits/{train,val,test}.csv`` from the download manifest.

    Args:
        val_fraction: Fraction of the official train split held out for validation.
        seed: Seed controlling the stratified draw.
        overwrite: Rewrite the CSVs even if they already exist.

    Returns:
        The image count per split.

    Raises:
        SplitError: If the manifest is missing, or splits already exist and
            ``overwrite`` is false.
    """
    import pandas as pd

    manifest_path = data_dir() / "stanford_cars" / "manifest.csv"
    if not manifest_path.exists():
        raise SplitError(f"{manifest_path} not found. Run `carvision data download` first.")

    out_dir = ensure_dir(splits_dir())
    existing = [p for p in (out_dir / f"{s}.csv" for s in SPLIT_NAMES) if p.exists()]
    if existing and not overwrite:
        raise SplitError(
            f"Splits already exist ({', '.join(p.name for p in existing)}). "
            f"They are committed on purpose so results stay comparable. "
            f"Pass --overwrite only if you intend to invalidate published numbers."
        )

    manifest = pd.read_csv(manifest_path)
    columns = ["image_id", "label_id", "label_name", "relpath"]

    official_train = manifest[manifest["split"] == "train"].sort_values("image_id")
    val_mask = stratified_val_split(
        official_train["label_id"].to_numpy(),
        val_fraction=val_fraction,
        seed=seed,
    )

    frames = {
        "train": official_train[~val_mask][columns],
        "val": official_train[val_mask][columns],
        "test": manifest[manifest["split"] == "test"].sort_values("image_id")[columns],
    }

    for name, frame in frames.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False)
        logger.info("%-5s %5d images, %3d classes", name, len(frame), frame["label_id"].nunique())

    sizes = SplitSizes(**{name: len(frame) for name, frame in frames.items()})
    assert_disjoint(frames)
    return sizes


def assert_disjoint(frames: dict[str, pd.DataFrame]) -> None:
    """Verify no image id appears in more than one split.

    Leakage between train and test is the single most common way a reported accuracy
    becomes a lie, so it is checked rather than assumed.

    Args:
        frames: Mapping of split name to its dataframe.

    Raises:
        SplitError: If any pair of splits shares an image id.
    """
    names = sorted(frames)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            overlap = set(frames[first]["image_id"]) & set(frames[second]["image_id"])
            if overlap:
                sample = sorted(overlap)[:5]
                raise SplitError(
                    f"{len(overlap)} image(s) appear in both {first!r} and {second!r}, "
                    f"e.g. {sample}. This would invalidate every reported metric."
                )


def load_split(name: str) -> pd.DataFrame:
    """Read one committed split CSV.

    Args:
        name: One of ``train``, ``val`` or ``test``.

    Returns:
        The split as a dataframe with ``image_id``, ``label_id``, ``label_name`` and
        ``relpath`` columns, sorted by ``image_id`` so row order is deterministic.

    Raises:
        SplitError: If the name is unknown or the CSV has not been generated.
    """
    import pandas as pd

    if name not in SPLIT_NAMES:
        raise SplitError(f"Unknown split {name!r}; expected one of {SPLIT_NAMES}.")

    path = splits_dir() / f"{name}.csv"
    if not path.exists():
        raise SplitError(f"{path} not found. Run `carvision data split` first.")

    return pd.read_csv(path).sort_values("image_id").reset_index(drop=True)


def split_image_root() -> Path:
    """Return the directory that ``relpath`` values in the splits are relative to."""
    return data_dir() / "stanford_cars"
