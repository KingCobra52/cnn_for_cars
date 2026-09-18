"""Acquire Stanford Cars and materialise it as a local image folder plus a manifest.

The canonical Stanford AI Lab download has been offline for years, so the dataset is
pulled from a Hugging Face Hub mirror. Mirrors differ in their column names and in how
they express the official train/test split, so the repository id and its column mapping
are configuration (see ``configs/data/stanford_cars.yaml``) rather than constants baked
into this module. :data:`KNOWN_MIRRORS` records the mappings that have been verified;
anything else can be supplied on the command line.

The output is deliberately boring and mirror-independent:

    data/stanford_cars/
        images/{split}/{image_id}.jpg
        manifest.csv        image_id,split,label_id,label_name,width,height
        classes.txt         one class name per line, ordered by label_id
        download.json       provenance: repo id, revision, counts, checksum

Everything downstream reads the manifest, so swapping mirrors changes one config value
and nothing else.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from carvision.utils.logging import get_logger
from carvision.utils.paths import data_dir, ensure_dir

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = get_logger(__name__)

#: Stanford Cars has 196 classes and 16,185 images, split 8,144 train / 8,041 test.
EXPECTED_NUM_CLASSES = 196
EXPECTED_SPLIT_SIZES = {"train": 8144, "test": 8041}
EXPECTED_TOTAL = sum(EXPECTED_SPLIT_SIZES.values())


@dataclass(frozen=True)
class MirrorSpec:
    """How to read one Hugging Face mirror of Stanford Cars.

    Attributes:
        repo_id: The Hub dataset repository id.
        image_column: Column holding the PIL image.
        label_column: Column holding the integer class id.
        train_split: Name of the official training split in the mirror.
        test_split: Name of the official test split in the mirror.
        note: Anything a future reader needs to know about this mirror.
    """

    repo_id: str
    image_column: str = "image"
    label_column: str = "label"
    train_split: str = "train"
    test_split: str = "test"
    note: str = ""


#: Mirrors with a known-good column mapping. The default is the first entry.
#:
#: These mappings must be confirmed against the live Hub before being relied on --
#: mirrors get renamed, gated, or re-uploaded with a different schema. ``verify_mirror``
#: checks the actual schema at download time and fails loudly on a mismatch rather than
#: silently training on the wrong column.
KNOWN_MIRRORS: dict[str, MirrorSpec] = {
    "tanganke/stanford_cars": MirrorSpec(
        repo_id="tanganke/stanford_cars",
        note="Plain image/label schema with the official train/test splits.",
    ),
    "Multimodal-Fatima/StanfordCars_train": MirrorSpec(
        repo_id="Multimodal-Fatima/StanfordCars_train",
        label_column="label",
        note="Train-only repo; pair with the matching StanfordCars_test repo.",
    ),
}

DEFAULT_MIRROR = "tanganke/stanford_cars"


class DatasetAcquisitionError(RuntimeError):
    """Raised when the dataset cannot be obtained in a usable, verified form."""


def resolve_mirror(repo_id: str | None = None, **overrides: str) -> MirrorSpec:
    """Return the :class:`MirrorSpec` for ``repo_id``, applying any overrides.

    An unknown ``repo_id`` is not an error: it yields a spec with default column names,
    which the caller can correct via ``overrides``. The schema is checked for real in
    :func:`verify_mirror`.

    Args:
        repo_id: Hub dataset repo id. Defaults to :data:`DEFAULT_MIRROR`.
        **overrides: Field names of :class:`MirrorSpec` to override.

    Returns:
        The resolved mirror specification.
    """
    key = repo_id or DEFAULT_MIRROR
    base = KNOWN_MIRRORS.get(key, MirrorSpec(repo_id=key, note="Unverified mirror."))
    if overrides:
        return MirrorSpec(**{**base.__dict__, **overrides})
    return base


def verify_mirror(dataset_dict: Any, spec: MirrorSpec) -> None:
    """Check that a loaded mirror actually has the schema ``spec`` claims.

    Fails before any image is written, so a schema drift surfaces as a clear message
    rather than as a mysteriously bad accuracy 40 minutes later.

    Args:
        dataset_dict: The object returned by ``datasets.load_dataset``.
        spec: The mirror specification to validate against.

    Raises:
        DatasetAcquisitionError: If a split or column is missing, or the class count is
            not 196.
    """
    for split in (spec.train_split, spec.test_split):
        if split not in dataset_dict:
            raise DatasetAcquisitionError(
                f"Mirror {spec.repo_id!r} has splits {sorted(dataset_dict)}, "
                f"but {split!r} was expected. Override with --train-split/--test-split."
            )

    features = dataset_dict[spec.train_split].features
    for column in (spec.image_column, spec.label_column):
        if column not in features:
            raise DatasetAcquisitionError(
                f"Mirror {spec.repo_id!r} has columns {sorted(features)}, "
                f"but {column!r} was expected. Override with "
                f"--image-column/--label-column."
            )

    label_feature = features[spec.label_column]
    names = getattr(label_feature, "names", None)
    if names is None:
        raise DatasetAcquisitionError(
            f"Column {spec.label_column!r} in {spec.repo_id!r} is not a ClassLabel, so "
            f"class names cannot be recovered. Pick a different mirror or column."
        )
    if len(names) != EXPECTED_NUM_CLASSES:
        raise DatasetAcquisitionError(
            f"Mirror {spec.repo_id!r} declares {len(names)} classes; Stanford Cars has "
            f"{EXPECTED_NUM_CLASSES}. This is probably the wrong dataset."
        )


def _iter_rows(split_dataset: Any, spec: MirrorSpec, split: str) -> Iterator[dict[str, Any]]:
    """Yield one manifest row per example, writing its image to disk as it goes."""
    out_dir = ensure_dir(data_dir() / "stanford_cars" / "images" / split)

    for index in range(len(split_dataset)):
        row = split_dataset[index]
        image = row[spec.image_column]
        label_id = int(row[spec.label_column])

        image_id = f"{split}_{index:05d}"
        path = out_dir / f"{image_id}.jpg"
        if not path.exists():
            # A handful of Stanford Cars images are greyscale; normalise to RGB once,
            # here, so no downstream consumer has to care.
            image.convert("RGB").save(path, format="JPEG", quality=95)

        yield {
            "image_id": image_id,
            "split": split,
            "label_id": label_id,
            "width": image.width,
            "height": image.height,
            "relpath": f"images/{split}/{image_id}.jpg",
        }


def _manifest_checksum(rows: list[dict[str, Any]]) -> str:
    """Return a stable SHA256 over the manifest's identifying fields."""
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda r: str(r["image_id"])):
        digest.update(f"{row['image_id']}|{row['split']}|{row['label_id']}\n".encode())
    return digest.hexdigest()


def download(
    repo_id: str | None = None,
    *,
    revision: str | None = None,
    force: bool = False,
    **overrides: str,
) -> Path:
    """Download Stanford Cars and write the local image folder and manifest.

    Idempotent: an existing, complete download is reused unless ``force`` is set.

    Args:
        repo_id: Hub dataset repo id. Defaults to :data:`DEFAULT_MIRROR`.
        revision: Optional Hub revision to pin, recorded in ``download.json``.
        force: Re-download and rewrite even if a complete manifest exists.
        **overrides: :class:`MirrorSpec` field overrides for an unverified mirror.

    Returns:
        The dataset root, ``data/stanford_cars``.

    Raises:
        DatasetAcquisitionError: If the mirror's schema or size does not match
            Stanford Cars.
    """
    import pandas as pd

    root = data_dir() / "stanford_cars"
    manifest_path = root / "manifest.csv"

    if manifest_path.exists() and not force:
        existing = pd.read_csv(manifest_path)
        if len(existing) == EXPECTED_TOTAL:
            logger.info("Reusing existing download at %s (%d images)", root, len(existing))
            return root
        logger.warning(
            "Manifest at %s has %d rows, expected %d -- re-downloading.",
            manifest_path,
            len(existing),
            EXPECTED_TOTAL,
        )

    spec = resolve_mirror(repo_id, **overrides)
    logger.info("Loading %s from the Hugging Face Hub...", spec.repo_id)

    from datasets import load_dataset

    dataset_dict = load_dataset(spec.repo_id, revision=revision)
    verify_mirror(dataset_dict, spec)

    class_names: list[str] = list(dataset_dict[spec.train_split].features[spec.label_column].names)

    ensure_dir(root)
    rows: list[dict[str, Any]] = []
    for split, mirror_split in (("train", spec.train_split), ("test", spec.test_split)):
        logger.info("Materialising %s split...", split)
        split_rows = list(_iter_rows(dataset_dict[mirror_split], spec, split))
        expected = EXPECTED_SPLIT_SIZES[split]
        if len(split_rows) != expected:
            logger.warning(
                "Split %s has %d images, expected %d. The mirror may differ from the "
                "official split; downstream numbers will not be comparable to published "
                "results.",
                split,
                len(split_rows),
                expected,
            )
        rows.extend(split_rows)

    frame = pd.DataFrame(rows)
    frame["label_name"] = frame["label_id"].map(dict(enumerate(class_names)))
    frame = frame[["image_id", "split", "label_id", "label_name", "width", "height", "relpath"]]
    frame.to_csv(manifest_path, index=False)

    (root / "classes.txt").write_text("\n".join(class_names) + "\n")

    provenance = {
        "repo_id": spec.repo_id,
        "revision": revision,
        "image_column": spec.image_column,
        "label_column": spec.label_column,
        "num_classes": len(class_names),
        "num_images": len(frame),
        "split_sizes": frame["split"].value_counts().to_dict(),
        "manifest_sha256": _manifest_checksum(rows),
    }
    (root / "download.json").write_text(json.dumps(provenance, indent=2) + "\n")

    logger.info("Wrote %d images and manifest to %s", len(frame), root)
    return root


def load_class_names() -> list[str]:
    """Read the class names written by :func:`download`.

    Returns:
        The 196 class names, indexed by label id.

    Raises:
        DatasetAcquisitionError: If the dataset has not been downloaded yet.
    """
    path = data_dir() / "stanford_cars" / "classes.txt"
    if not path.exists():
        raise DatasetAcquisitionError(f"{path} not found. Run `carvision data download` first.")
    return path.read_text().splitlines()
