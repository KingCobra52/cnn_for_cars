"""Torch datasets over the committed splits."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from carvision.data.splits import load_split, split_image_root

if TYPE_CHECKING:
    from collections.abc import Callable


class CarsDataset(Dataset[tuple[torch.Tensor, int, str]]):
    """Stanford Cars images for one split.

    Yields ``(image, label_id, image_id)``. The image id travels with the sample so that
    predictions can be joined back to specific images for error analysis -- the legacy
    notebook could only ever say "some truck was wrong", never which one.

    Attributes:
        image_ids: Image ids in row order.
        labels: Class ids in row order, as an int64 array.
        class_names: Class names indexed by class id.
    """

    def __init__(
        self,
        split: str,
        transform: Callable[[Image.Image], torch.Tensor],
        *,
        root: Path | None = None,
    ) -> None:
        """Initialise the dataset.

        Args:
            split: One of ``train``, ``val`` or ``test``.
            transform: Preprocessing applied to each PIL image. Supply the transform
                that belongs to the backbone (see :mod:`carvision.data.transforms`);
                there is deliberately no default, because the legacy project's worst bug
                was feeding CIFAR-normalised 32x32 tensors to an ImageNet VGG16.
            root: Directory that ``relpath`` values are relative to. Defaults to the
                downloaded dataset root.
        """
        frame = load_split(split)
        self.split = split
        self.transform = transform
        self.root = root if root is not None else split_image_root()

        self.image_ids: list[str] = frame["image_id"].astype(str).tolist()
        self._relpaths: list[str] = frame["relpath"].astype(str).tolist()
        self.labels: np.ndarray = frame["label_id"].to_numpy(dtype=np.int64)

        pairs = frame[["label_id", "label_name"]].drop_duplicates().sort_values("label_id")
        self.class_names: list[str] = pairs["label_name"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        path = self.root / self._relpaths[index]
        with Image.open(path) as handle:
            image = handle.convert("RGB")
            tensor = self.transform(image)
        return tensor, int(self.labels[index]), self.image_ids[index]

    @property
    def num_classes(self) -> int:
        """Number of distinct classes present in this split."""
        return len(self.class_names)


class EmbeddingDataset(Dataset[tuple[torch.Tensor, int]]):
    """Cached backbone embeddings and their labels.

    This is what head training actually consumes. Because the embeddings are already in
    memory as one contiguous array, training a linear probe over 8k samples takes
    seconds on a CPU -- which is the whole reason the sweep in this project is
    affordable.
    """

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        """Initialise from in-memory arrays.

        Args:
            embeddings: Float array of shape ``(N, D)``.
            labels: Integer array of shape ``(N,)``.

        Raises:
            ValueError: If the two arrays disagree on ``N``.
        """
        if len(embeddings) != len(labels):
            raise ValueError(f"embeddings has {len(embeddings)} rows but labels has {len(labels)}")
        self.embeddings = torch.from_numpy(np.ascontiguousarray(embeddings, dtype=np.float32))
        self.labels = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.embeddings[index], int(self.labels[index])

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        return int(self.embeddings.shape[1])

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the whole split as ``(embeddings, labels)``.

        Full-batch access beats iterating a DataLoader when the split fits in memory,
        which it always does here (8k x 768 floats is ~25 MB).
        """
        return self.embeddings, self.labels


def collate_with_ids(
    batch: list[tuple[torch.Tensor, int, str]],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Collate ``(image, label, image_id)`` samples, keeping ids as a list.

    Args:
        batch: Samples from :class:`CarsDataset`.

    Returns:
        Stacked images, stacked labels, and the list of image ids.
    """
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    ids = [item[2] for item in batch]
    return images, labels, ids


def describe(dataset: CarsDataset) -> dict[str, Any]:
    """Summarise a split for logging and the data-exploration notebook.

    Args:
        dataset: The split to summarise.

    Returns:
        Counts and per-class balance statistics.
    """
    counts = np.bincount(dataset.labels, minlength=dataset.num_classes)
    return {
        "split": dataset.split,
        "num_images": len(dataset),
        "num_classes": dataset.num_classes,
        "images_per_class_min": int(counts.min()),
        "images_per_class_max": int(counts.max()),
        "images_per_class_mean": float(counts.mean()),
    }
