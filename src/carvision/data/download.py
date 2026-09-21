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
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from carvision.config import DataConfig, load_data_config
from carvision.utils.logging import get_logger
from carvision.utils.paths import data_dir, ensure_dir

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = get_logger(__name__)

#: Stanford Cars has 196 classes and 16,185 images, split 8,144 train / 8,041 test.
EXPECTED_NUM_CLASSES = 196
EXPECTED_SPLIT_SIZES = {"train": 8144, "test": 8041}
EXPECTED_TOTAL = sum(EXPECTED_SPLIT_SIZES.values())
EXPECTED_REVISION = "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40"


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
        note=(
            "Dataset card documents Stanford Cars official train=8144/test=8041 "
            "partitions; pinned to the parquet conversion commit."
        ),
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

    names_by_split = []
    for split in (spec.train_split, spec.test_split):
        features = dataset_dict[split].features
        for column in (spec.image_column, spec.label_column):
            if column not in features:
                kind = "image-column" if column == spec.image_column else "label-column"
                raise DatasetAcquisitionError(f"{split} is missing required {kind} {column!r}")
        names = getattr(features[spec.label_column], "names", None)
        if names is None or len(names) != EXPECTED_NUM_CLASSES:
            raise DatasetAcquisitionError(
                f"{split} label column must be a {EXPECTED_NUM_CLASSES}-class ClassLabel"
            )
        names_by_split.append(list(names))
    if names_by_split[0] != names_by_split[1]:
        raise DatasetAcquisitionError("Train and test class names differ or are out of order")
    for split, expected in zip(
        (spec.train_split, spec.test_split), EXPECTED_SPLIT_SIZES.values(), strict=True
    ):
        try:
            actual = len(dataset_dict[split])
        except TypeError:
            actual = None
        if actual is not None:
            if actual != expected:
                raise DatasetAcquisitionError(
                    f"{split} has {actual} rows; expected exactly {expected}"
                )
            for index in range(actual):
                label = int(dataset_dict[split][index][spec.label_column])
                if not 0 <= label < EXPECTED_NUM_CLASSES:
                    raise DatasetAcquisitionError(f"{split} row {index} has invalid label {label}")


def _iter_rows(
    split_dataset: Any,
    spec: MirrorSpec,
    split: str,
    *,
    out_root: Path,
) -> Iterator[dict[str, Any]]:
    """Yield one manifest row per example, writing its image to disk as it goes.

    Args:
        out_root: Staging directory receiving materialized images.
        split_dataset: The mirror's split.
        spec: How to read the mirror's columns.
        split: ``train`` or ``test``.
        overwrite: Rewrite images that already exist. This must be true for a forced
            re-download: skipping existing files would keep the *previous* mirror's
            images while adopting the new mirror's labels, silently mislabelling the
            whole dataset.
    """
    out_dir = ensure_dir(out_root / "images" / split)

    for index in range(len(split_dataset)):
        row = split_dataset[index]
        image = row[spec.image_column]
        label_id = int(row[spec.label_column])

        image_id = f"{split}_{index:05d}"
        path = out_dir / f"{image_id}.jpg"
        # A handful of images are greyscale; retain the established RGB/JPEG behavior.
        image.convert("RGB").save(path, format="JPEG", quality=95)
        with path.open("rb") as handle:
            content_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()

        yield {
            "image_id": image_id,
            "split": split,
            "label_id": label_id,
            "width": image.width,
            "height": image.height,
            "relpath": f"images/{split}/{image_id}.jpg",
            "content_sha256": content_sha256,
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
    config: DataConfig | None = None,
    **overrides: str,
) -> Path:
    """Download Stanford Cars and write the local image folder and manifest.

    Idempotent: an existing, complete download is reused unless ``force`` is set.

    Args:
        repo_id: Hub dataset repo id. Defaults to the one in the data config.
        revision: Optional Hub revision to pin, recorded in ``download.json``.
        force: Re-download and rewrite even if a complete manifest exists.
        config: Data configuration. Defaults to ``configs/data/stanford_cars.yaml``.
        **overrides: :class:`MirrorSpec` field overrides, applied on top of the config.

    Returns:
        The dataset root, ``data/stanford_cars``.

    Raises:
        DatasetAcquisitionError: If the mirror's schema or size does not match
            Stanford Cars.
    """
    import pandas as pd

    root = data_dir() / "stanford_cars"
    manifest_path = root / "manifest.csv"

    settings = config or load_data_config()
    spec = resolve_mirror(
        repo_id or settings.hf_repo_id,
        **{**settings.mirror_overrides(), **overrides},
    )
    revision = revision or settings.hf_revision
    if not revision or len(revision) != 40:
        raise DatasetAcquisitionError(
            "The configured source revision must be a 40-character commit SHA"
        )
    if spec.repo_id == DEFAULT_MIRROR and revision != EXPECTED_REVISION:
        raise DatasetAcquisitionError(
            f"Unsupported Stanford Cars revision {revision}; expected {EXPECTED_REVISION}"
        )
    if manifest_path.exists() and not force:
        try:
            provenance = json.loads((root / "download.json").read_text())
            requested = {
                "repo_id": spec.repo_id,
                "revision": revision,
                "image_column": spec.image_column,
                "label_column": spec.label_column,
                "train_split": spec.train_split,
                "test_split": spec.test_split,
            }
            if all(provenance.get(k) == v for k, v in requested.items()):
                verify_local_dataset(root, expected_provenance=provenance)
                logger.info("Reusing verified download at %s", root)
                return root
            raise DatasetAcquisitionError(
                "Existing dataset provenance does not match requested source/mapping; use --force"
            )
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            raise DatasetAcquisitionError(
                f"Existing dataset provenance is unreadable: {exc}; use --force"
            ) from exc
    logger.info("Loading %s from the Hugging Face Hub...", spec.repo_id)

    from datasets import load_dataset

    # Read only the benchmark parquet files; the repo also carries nine robustness
    # variants that are outside this acquisition contract.
    base = f"https://huggingface.co/datasets/{spec.repo_id}/resolve/{revision}/data"
    train_data, test_data = load_dataset(
        "parquet",
        data_files={
            "train": [
                f"{base}/train-00000-of-00002.parquet",
                f"{base}/train-00001-of-00002.parquet",
            ],
            "test": [
                f"{base}/test-00000-of-00002.parquet",
                f"{base}/test-00001-of-00002.parquet",
            ],
        },
        split=["train", "test"],
    )
    dataset_dict = {spec.train_split: train_data, spec.test_split: test_data}
    verify_mirror(dataset_dict, spec)

    class_names: list[str] = list(dataset_dict[spec.train_split].features[spec.label_column].names)

    staging = Path(tempfile.mkdtemp(prefix="stanford_cars-", dir=str(data_dir())))
    rows: list[dict[str, Any]] = []
    try:
        for split, mirror_split in (("train", spec.train_split), ("test", spec.test_split)):
            logger.info("Materialising %s split...", split)
            rows.extend(list(_iter_rows(dataset_dict[mirror_split], spec, split, out_root=staging)))

        frame = pd.DataFrame(rows)
        frame["label_name"] = frame["label_id"].map(dict(enumerate(class_names)))
        frame = frame[
            [
                "image_id",
                "split",
                "label_id",
                "label_name",
                "width",
                "height",
                "relpath",
                "content_sha256",
            ]
        ]
        frame.to_csv(staging / "manifest.csv", index=False)

        (staging / "classes.txt").write_text("\n".join(class_names) + "\n")

        provenance = {
            "repo_id": spec.repo_id,
            "revision": revision,
            "image_column": spec.image_column,
            "label_column": spec.label_column,
            "train_split": spec.train_split,
            "test_split": spec.test_split,
            "num_classes": len(class_names),
            "num_images": len(frame),
            "split_sizes": frame["split"].value_counts().to_dict(),
            "manifest_sha256": _manifest_checksum(rows),
            "image_content_sha256": hashlib.sha256(
                "".join(sorted(r["content_sha256"] for r in rows)).encode()
            ).hexdigest(),
            "duplicate_content_audit": _duplicate_audit(rows),
            "conversion": {"mode": "RGB", "format": "JPEG", "quality": 95},
            "library_versions": _library_versions(),
        }
        (staging / "download.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        )
        verify_local_dataset(staging, expected_provenance=provenance)
        if root.exists():
            backup = root.with_name(root.name + ".old")
            if backup.exists():
                shutil.rmtree(backup)
            root.rename(backup)
        staging.rename(root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    logger.info("Wrote %d images and manifest to %s", len(frame), root)
    return root


def _library_versions() -> dict[str, str]:
    import datasets
    import pandas
    import PIL

    return {
        "pillow": PIL.__version__,
        "datasets": datasets.__version__,
        "pandas": pandas.__version__,
    }


def _duplicate_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[str]] = {}
    for row in rows:
        groups.setdefault(str(row["content_sha256"]), []).append(str(row["image_id"]))
    duplicates = [ids for ids in groups.values() if len(ids) > 1]
    return {
        "duplicate_groups": len(duplicates),
        "duplicate_images": sum(len(ids) for ids in duplicates),
        "groups": duplicates,
    }


def verify_local_dataset(
    root: Path | None = None, *, expected_provenance: dict[str, Any] | None = None
) -> None:
    """Validate manifests, checksums, counts, labels, and every decodable image."""
    import pandas as pd
    from PIL import Image

    root = root or data_dir() / "stanford_cars"
    provenance = expected_provenance or json.loads((root / "download.json").read_text())
    frame = pd.read_csv(root / "manifest.csv")
    required = [
        "image_id",
        "split",
        "label_id",
        "label_name",
        "width",
        "height",
        "relpath",
        "content_sha256",
    ]
    if list(frame.columns) != required:
        raise DatasetAcquisitionError(
            f"Manifest columns are {list(frame.columns)}, expected {required}"
        )
    if (
        len(frame) != EXPECTED_TOTAL
        or frame["split"].value_counts().to_dict() != EXPECTED_SPLIT_SIZES
    ):
        raise DatasetAcquisitionError("Manifest does not contain the exact official split counts")
    if (
        frame["image_id"].duplicated().any()
        or not frame["label_id"].between(0, EXPECTED_NUM_CLASSES - 1).all()
    ):
        raise DatasetAcquisitionError("Manifest has duplicate image IDs or invalid labels")
    class_names = (root / "classes.txt").read_text().splitlines()
    if len(class_names) != EXPECTED_NUM_CLASSES or frame["label_name"].tolist() != [
        class_names[int(i)] for i in frame["label_id"]
    ]:
        raise DatasetAcquisitionError("Manifest labels do not match the ordered class mapping")
    digest_rows: list[dict[str, Any]] = [
        {str(key): value for key, value in row.items()} for row in frame.to_dict("records")
    ]
    if provenance.get("manifest_sha256") != _manifest_checksum(digest_rows):
        raise DatasetAcquisitionError("Manifest checksum mismatch")
    for row in digest_rows:
        path = root / row["relpath"]
        try:
            with Image.open(path) as image:
                image.convert("RGB").load()
            with path.open("rb") as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != row["content_sha256"]:
                    raise DatasetAcquisitionError(f"Image checksum mismatch: {path}")
        except (OSError, ValueError) as exc:
            raise DatasetAcquisitionError(f"Image missing or corrupt: {path}: {exc}") from exc


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
