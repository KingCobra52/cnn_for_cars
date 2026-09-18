"""Cache frozen-backbone embeddings so the expensive pass happens exactly once.

This module is the reason the project is feasible without a GPU.

Running a ViT over 16,185 images on four CPU cores takes the better part of an hour.
Fine-tuning, which would repeat that work every epoch, is simply not available. But the
backbone is frozen, so its output for a given image never changes -- which means the pass
is not a per-epoch cost at all. It is a one-time cost, and the result is a 16,185 x 384
array that fits comfortably in memory.

Once embeddings are cached, training a linear probe is a few seconds of dense linear
algebra. A sweep of three backbones by two heads by five seeds costs three cache builds
and thirty near-free runs, which is how this project affords confidence intervals that
GPU-rich projects often skip.

Correctness rests on the cache key. It hashes the backbone identity, the exact pretrained
weights tag, the full preprocessing specification, and the ordered list of image ids. Any
change to any of those produces a different key and a rebuild, so it is not possible to
train on embeddings that were computed with different preprocessing than the config
claims.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader

from carvision.data.dataset import CarsDataset, collate_with_ids
from carvision.models.backbones import BackboneSpec, embed_batch, get_backbone
from carvision.utils.logging import get_logger
from carvision.utils.paths import cache_dir, ensure_dir
from carvision.utils.seed import seed_worker

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger(__name__)

#: Bump when the on-disk layout changes in a way older caches cannot satisfy.
CACHE_FORMAT_VERSION = 1

#: Rows written per flush. Small enough that a laptop going to sleep costs little,
#: large enough that the manifest write is not the bottleneck.
CHUNK_ROWS = 512


class CacheError(RuntimeError):
    """Raised when a cache entry is missing, corrupt, or inconsistent with its manifest."""


@dataclass(frozen=True)
class CacheEntry:
    """A materialised set of embeddings for one (backbone, split) pair.

    Attributes:
        key: The content hash identifying this entry.
        backbone: Backbone name.
        split: Split name.
        num_rows: Number of embedded images.
        dim: Embedding dimensionality.
        directory: Where the arrays live.
    """

    key: str
    backbone: str
    split: str
    num_rows: int
    dim: int
    directory: Path


def compute_cache_key(spec: BackboneSpec, image_ids: Sequence[str]) -> str:
    """Hash everything that determines the embeddings.

    Args:
        spec: The backbone, which carries both its weights tag and its preprocessing.
        image_ids: The images to embed, in the order they will be embedded.

    Returns:
        A 16-character hex digest. Truncated because it names a directory and full
        SHA256 is unwieldy; 64 bits of collision resistance is ample for a local cache.
    """
    digest = hashlib.sha256()
    digest.update(f"v{CACHE_FORMAT_VERSION}\n".encode())
    digest.update(f"{spec.cache_key()}\n".encode())
    digest.update(f"n={len(image_ids)}\n".encode())
    for image_id in image_ids:
        digest.update(f"{image_id}\n".encode())
    return digest.hexdigest()[:16]


def entry_dir(backbone: str, split: str, key: str) -> Path:
    """Return the directory holding one cache entry."""
    return cache_dir() / backbone / f"{split}-{key}"


def _write_manifest(directory: Path, payload: dict[str, object]) -> None:
    (directory / "manifest.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")


def _read_manifest(directory: Path) -> dict[str, object]:
    path = directory / "manifest.json"
    if not path.exists():
        raise CacheError(f"No manifest at {path}")
    loaded = json.loads(path.read_text())
    assert isinstance(loaded, dict)
    return loaded


def _git_sha() -> str | None:
    """Return the current commit, for provenance. None outside a git checkout."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def build(
    backbone: str,
    split: str,
    *,
    batch_size: int = 32,
    num_workers: int = 2,
    force: bool = False,
) -> CacheEntry:
    """Embed one split with one backbone, writing the result to the cache.

    Resumable. Embeddings are flushed to a memory-mapped array every
    :data:`CHUNK_ROWS` rows and progress is recorded, so an interrupted build restarts
    from the last flush rather than from zero -- which matters when a build takes an hour.

    Args:
        backbone: A registered backbone name.
        split: ``train``, ``val`` or ``test``.
        batch_size: Images per forward pass. Larger is marginally faster on CPU but uses
            more memory; 32 is a reasonable default for 224x224 input.
        num_workers: DataLoader workers for image decoding, which is the real bottleneck
            on CPU alongside the forward pass itself.
        force: Rebuild even if a complete entry exists.

    Returns:
        The cache entry.
    """
    spec = get_backbone(backbone)
    dataset = CarsDataset(split, transform=spec.preprocess.build(train=False))

    key = compute_cache_key(spec, dataset.image_ids)
    directory = entry_dir(backbone, split, key)

    if not force and is_complete(directory, len(dataset)):
        logger.info("Cache hit: %s/%s (%s)", backbone, split, key)
        return _entry_from_manifest(directory)

    ensure_dir(directory)
    num_rows, dim = len(dataset), spec.embedding_dim

    embeddings_path = directory / "embeddings.npy"
    progress_path = directory / "progress.json"

    start_row = 0
    if not force and embeddings_path.exists() and progress_path.exists():
        start_row = int(json.loads(progress_path.read_text())["rows_done"])
        logger.info("Resuming %s/%s from row %d of %d", backbone, split, start_row, num_rows)

    memmap = np.lib.format.open_memmap(
        embeddings_path,
        mode="r+" if start_row else "w+",
        dtype=np.float32,
        shape=(num_rows, dim),
    )

    module = spec.build()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Row order must match image_ids exactly.
        num_workers=num_workers,
        collate_fn=collate_with_ids,
        worker_init_fn=seed_worker,
    )

    logger.info(
        "Building %s/%s: %d images -> (%d, %d). This is the slow step; it happens once.",
        backbone,
        split,
        num_rows,
        num_rows,
        dim,
    )
    started = time.monotonic()
    row = 0
    last_flush = start_row

    from tqdm import tqdm

    for images, _, _ in tqdm(loader, desc=f"{backbone}/{split}", unit="batch"):
        batch_rows = len(images)
        if row + batch_rows <= start_row:
            # Already embedded in a previous, interrupted run.
            row += batch_rows
            continue

        vectors = embed_batch(module, images)
        if vectors.shape[1] != dim:
            raise CacheError(
                f"{backbone} produced {vectors.shape[1]}-d embeddings but the registry "
                f"declares {dim}. Fix BackboneSpec.embedding_dim."
            )
        memmap[row : row + batch_rows] = vectors.numpy()
        row += batch_rows

        if row - last_flush >= CHUNK_ROWS:
            memmap.flush()
            progress_path.write_text(json.dumps({"rows_done": row}))
            last_flush = row

    memmap.flush()
    del memmap

    np.save(directory / "labels.npy", dataset.labels)
    (directory / "image_ids.txt").write_text("\n".join(dataset.image_ids) + "\n")

    elapsed = time.monotonic() - started
    _write_manifest(
        directory,
        {
            "format_version": CACHE_FORMAT_VERSION,
            "key": key,
            "backbone": backbone,
            "backbone_cache_key": spec.cache_key(),
            "weights_tag": spec.weights_tag,
            "preprocess": asdict(spec.preprocess),
            "split": split,
            "num_rows": num_rows,
            "dim": dim,
            "seconds": round(elapsed, 1),
            "torch_version": torch.__version__,
            "git_sha": _git_sha(),
        },
    )
    progress_path.unlink(missing_ok=True)

    logger.info(
        "Built %s/%s in %.1f min (%.1f images/s)",
        backbone,
        split,
        elapsed / 60,
        num_rows / max(elapsed, 1e-9),
    )
    return _entry_from_manifest(directory)


def is_complete(directory: Path, expected_rows: int) -> bool:
    """Return whether ``directory`` holds a finished, consistent cache entry.

    A half-written entry from an interrupted build is deliberately reported as
    incomplete: it still has a ``progress.json``, and no manifest.

    Args:
        directory: Candidate entry directory.
        expected_rows: Row count the entry must have.
    """
    if not (directory / "manifest.json").exists():
        return False
    if (directory / "progress.json").exists():
        return False
    try:
        manifest = _read_manifest(directory)
    except (CacheError, json.JSONDecodeError):
        return False
    return (
        manifest.get("num_rows") == expected_rows
        and (directory / "embeddings.npy").exists()
        and (directory / "labels.npy").exists()
    )


def _entry_from_manifest(directory: Path) -> CacheEntry:
    manifest = _read_manifest(directory)
    return CacheEntry(
        key=str(manifest["key"]),
        backbone=str(manifest["backbone"]),
        split=str(manifest["split"]),
        num_rows=int(manifest["num_rows"]),  # type: ignore[arg-type]
        dim=int(manifest["dim"]),  # type: ignore[arg-type]
        directory=directory,
    )


def load(backbone: str, split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read cached embeddings, building them first if absent.

    This is the function head training calls. Callers never touch cache paths or keys.

    Args:
        backbone: A registered backbone name.
        split: ``train``, ``val`` or ``test``.

    Returns:
        ``(embeddings, labels, image_ids)`` with shapes ``(N, D)``, ``(N,)`` and length
        ``N``. Image ids are returned so predictions can be joined back to images.

    Raises:
        CacheError: If the cached arrays disagree with each other.
    """
    entry = build(backbone, split)

    embeddings = np.load(entry.directory / "embeddings.npy")
    labels = np.load(entry.directory / "labels.npy")
    image_ids = (entry.directory / "image_ids.txt").read_text().splitlines()

    if not len(embeddings) == len(labels) == len(image_ids):
        raise CacheError(
            f"Corrupt cache entry {entry.directory}: {len(embeddings)} embeddings, "
            f"{len(labels)} labels, {len(image_ids)} ids. Rebuild with --force."
        )
    return embeddings, labels, image_ids


def clear(backbone: str | None = None) -> int:
    """Delete cache entries.

    Args:
        backbone: Clear only this backbone's entries, or all of them when None.

    Returns:
        The number of entry directories removed.
    """
    import shutil

    root = cache_dir() / backbone if backbone else cache_dir()
    if not root.exists():
        return 0

    removed = 0
    for directory in sorted(root.glob("**/manifest.json")):
        shutil.rmtree(directory.parent)
        removed += 1
    logger.info("Removed %d cache entries from %s", removed, root)
    return removed
