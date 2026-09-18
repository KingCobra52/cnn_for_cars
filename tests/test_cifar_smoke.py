"""The rebuilt legacy dataset, including the stratification bug it fixes."""

from __future__ import annotations

import numpy as np
import pytest

from carvision.data.cifar_smoke import (
    CIFAR_AUTOMOBILE,
    CIFAR_TRUCK,
    construct_vehicle_subset,
    normalize,
    to_nchw,
)


@pytest.fixture
def fake_cifar() -> tuple[np.ndarray, np.ndarray]:
    """A CIFAR-10-shaped array where pixel values encode the source class.

    Encoding the class in the pixels lets a test verify *which* source images were
    chosen, not merely how many.
    """
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(10, dtype=np.int64), 100)
    images = np.zeros((len(labels), 32, 32, 3), dtype=np.uint8)
    images[:, 0, 0, 0] = labels * 25
    images[:, 1:, :, :] = rng.integers(0, 256, size=(len(labels), 31, 32, 3), dtype=np.uint8)
    return images, labels


def test_subset_is_balanced(fake_cifar: tuple[np.ndarray, np.ndarray]) -> None:
    images, labels = fake_cifar
    subset = construct_vehicle_subset(images, labels, images_per_class=80)
    counts = np.bincount(subset.labels)
    assert list(counts) == [80, 80, 80]
    assert len(subset) == 240


def test_background_draws_from_all_eight_source_classes(
    fake_cifar: tuple[np.ndarray, np.ndarray],
) -> None:
    """The legacy bug: background was the first N rows, biased to storage order.

    Here every non-vehicle class must contribute roughly an equal share.
    """
    images, labels = fake_cifar
    subset = construct_vehicle_subset(images, labels, images_per_class=80)

    background = subset.images[subset.labels == 0]
    source_classes = np.unique(background[:, 0, 0, 0] // 25)

    expected = sorted(c for c in range(10) if c not in (CIFAR_AUTOMOBILE, CIFAR_TRUCK))
    assert sorted(source_classes.tolist()) == expected

    per_source = np.bincount(background[:, 0, 0, 0] // 25, minlength=10)
    present = per_source[per_source > 0]
    assert present.max() - present.min() <= 1, "background classes are not evenly sampled"


def test_vehicle_classes_come_from_the_right_cifar_labels(
    fake_cifar: tuple[np.ndarray, np.ndarray],
) -> None:
    images, labels = fake_cifar
    subset = construct_vehicle_subset(images, labels, images_per_class=50)

    cars = subset.images[subset.labels == 1]
    trucks = subset.images[subset.labels == 2]
    assert set((cars[:, 0, 0, 0] // 25).tolist()) == {CIFAR_AUTOMOBILE}
    assert set((trucks[:, 0, 0, 0] // 25).tolist()) == {CIFAR_TRUCK}


def test_construction_is_seeded(fake_cifar: tuple[np.ndarray, np.ndarray]) -> None:
    images, labels = fake_cifar
    first = construct_vehicle_subset(images, labels, 40, seed=11)
    second = construct_vehicle_subset(images, labels, 40, seed=11)
    third = construct_vehicle_subset(images, labels, 40, seed=12)

    np.testing.assert_array_equal(first.images, second.images)
    assert not np.array_equal(first.images, third.images)


def test_requesting_too_many_images_fails_loudly(
    fake_cifar: tuple[np.ndarray, np.ndarray],
) -> None:
    images, labels = fake_cifar
    with pytest.raises(ValueError, match="only"):
        construct_vehicle_subset(images, labels, images_per_class=5000)


def test_normalize_produces_roughly_zero_mean() -> None:
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(64, 32, 32, 3), dtype=np.uint8)
    out = normalize(images)
    assert out.dtype == np.float32
    # Uniform pixels are not CIFAR, so only a loose bound is meaningful here.
    assert abs(float(out.mean())) < 1.0


def test_to_nchw_moves_channels_and_stays_contiguous() -> None:
    images = np.zeros((4, 32, 32, 3), dtype=np.float32)
    out = to_nchw(images)
    assert out.shape == (4, 3, 32, 32)
    assert out.flags["C_CONTIGUOUS"]
