"""The legacy 3-class CIFAR-10 task, kept as a fast smoke test.

This is the original project's dataset -- ``background`` / ``car`` / ``truck`` at 32x32 --
rebuilt cleanly. It is useful for two things: exercising the whole pipeline in seconds
without the 16k-image Stanford Cars download, and giving the README an honest
apples-to-apples "here is what the old task looked like" row.

Two fixes relative to the original ``construct_vehicle_dataset``:

* **Stratified background.** The original took the first N images of the other eight
  CIFAR classes, unshuffled, so the ``background`` class was biased toward whichever
  classes happen to come first in CIFAR's storage order. Here each of the eight
  contributes an equal share, drawn with a seeded RNG.
* **No TensorFlow.** The original imported all of TensorFlow solely to call
  ``keras.datasets.cifar10.load_data()``. torchvision downloads the same bytes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from carvision.utils.logging import get_logger
from carvision.utils.paths import data_dir, ensure_dir

logger = get_logger(__name__)

CLASS_NAMES = ("background", "car", "truck")

#: CIFAR-10 label ids for the two vehicle classes we keep.
CIFAR_AUTOMOBILE = 1
CIFAR_TRUCK = 9

#: Normalisation statistics for CIFAR-10. The legacy notebook used these values under a
#: comment calling them CIFAR-100 statistics; they are in fact CIFAR-10's.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


@dataclass(frozen=True)
class VehicleSubset:
    """A 3-class subset of CIFAR-10.

    Attributes:
        images: ``(N, 32, 32, 3)`` uint8 array in HWC order.
        labels: ``(N,)`` int64 array with values 0, 1 or 2.
    """

    images: np.ndarray
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)


def construct_vehicle_subset(
    images: np.ndarray,
    labels: np.ndarray,
    images_per_class: int,
    *,
    seed: int = 0,
) -> VehicleSubset:
    """Build a balanced 3-class subset from raw CIFAR-10 arrays.

    Args:
        images: ``(N, 32, 32, 3)`` uint8 CIFAR-10 images.
        labels: ``(N,)`` CIFAR-10 class ids in ``0..9``.
        images_per_class: Images drawn for each of the three output classes.
        seed: Seed for the per-class sampling.

    Returns:
        The balanced subset, ordered ``background``, ``car``, ``truck``.

    Raises:
        ValueError: If any source class has too few images to satisfy the request.
    """
    rng = np.random.default_rng(seed)
    labels = labels.squeeze()

    def take(mask: np.ndarray, count: int) -> np.ndarray:
        (indices,) = np.nonzero(mask)
        if len(indices) < count:
            raise ValueError(f"Requested {count} images but only {len(indices)} available.")
        return rng.choice(np.sort(indices), size=count, replace=False)

    # Background: an equal share from each of the eight non-vehicle classes, rather than
    # the first N rows in storage order.
    background_classes = [c for c in range(10) if c not in (CIFAR_AUTOMOBILE, CIFAR_TRUCK)]
    per_source, remainder = divmod(images_per_class, len(background_classes))
    background_indices: list[np.ndarray] = []
    for position, cifar_class in enumerate(background_classes):
        count = per_source + (1 if position < remainder else 0)
        background_indices.append(take(labels == cifar_class, count))

    selected = [
        np.concatenate(background_indices),
        take(labels == CIFAR_AUTOMOBILE, images_per_class),
        take(labels == CIFAR_TRUCK, images_per_class),
    ]

    return VehicleSubset(
        images=np.concatenate([images[idx] for idx in selected]),
        labels=np.repeat(np.arange(3, dtype=np.int64), images_per_class),
    )


def normalize(images: np.ndarray) -> np.ndarray:
    """Scale uint8 images to ``[0, 1]`` and apply CIFAR-10 normalisation.

    Args:
        images: ``(N, 32, 32, 3)`` uint8 array.

    Returns:
        A float32 array of the same shape.
    """
    scaled = images.astype(np.float32) / 255.0
    return (scaled - np.array(CIFAR10_MEAN, dtype=np.float32)) / np.array(
        CIFAR10_STD, dtype=np.float32
    )


def to_nchw(images: np.ndarray) -> np.ndarray:
    """Convert an NHWC image array to the NCHW layout torch convolutions expect.

    Args:
        images: ``(N, H, W, C)`` array.

    Returns:
        A contiguous ``(N, C, H, W)`` array.
    """
    return np.ascontiguousarray(images.transpose(0, 3, 1, 2))


def load(
    images_per_class: int = 5000,
    test_images_per_class: int = 1000,
    *,
    seed: int = 0,
) -> tuple[VehicleSubset, VehicleSubset]:
    """Download CIFAR-10 and build the train and test vehicle subsets.

    Args:
        images_per_class: Training images per output class.
        test_images_per_class: Test images per output class.
        seed: Seed for the sampling.

    Returns:
        A ``(train, test)`` pair of subsets.
    """
    from torchvision.datasets import CIFAR10

    root = ensure_dir(data_dir() / "cifar10")
    train_raw = CIFAR10(root=str(root), train=True, download=True)
    test_raw = CIFAR10(root=str(root), train=False, download=True)

    train = construct_vehicle_subset(
        train_raw.data, np.asarray(train_raw.targets), images_per_class, seed=seed
    )
    test = construct_vehicle_subset(
        test_raw.data, np.asarray(test_raw.targets), test_images_per_class, seed=seed + 1
    )
    logger.info("cifar_smoke: %d train, %d test images", len(train), len(test))
    return train, test
