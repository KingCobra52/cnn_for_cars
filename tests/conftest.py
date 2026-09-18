"""Shared fixtures.

Every test in this suite runs on synthetic data with no network access and no dataset
download, so CI stays fast and green on a fresh clone.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded generator, so a failing test fails the same way twice."""
    return np.random.default_rng(1234)


@pytest.fixture
def synthetic_labels() -> np.ndarray:
    """Labels for 40 classes with a deliberately uneven number of images each.

    The imbalance (2 to 41 images per class) is what makes stratification testable:
    an unstratified draw would leave the smallest classes empty.
    """
    return np.concatenate([np.full(index + 2, index, dtype=np.int64) for index in range(40)])


@pytest.fixture
def synthetic_embeddings(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Linearly separable embeddings for 10 classes.

    Each class is a tight Gaussian blob around its own random centre, so any working
    classifier head should reach near-perfect accuracy. That makes the fixture a useful
    canary: if a head cannot fit this, the bug is in the head, not the data.
    """
    num_classes, per_class, dim = 10, 20, 16
    centres = rng.normal(scale=4.0, size=(num_classes, dim))
    labels = np.repeat(np.arange(num_classes, dtype=np.int64), per_class)
    embeddings = centres[labels] + rng.normal(scale=0.2, size=(len(labels), dim))
    return embeddings.astype(np.float32), labels


@pytest.fixture
def tmp_repo_root(tmp_path, monkeypatch) -> object:
    """Redirect every carvision path helper at a temporary directory."""
    monkeypatch.setenv("CARVISION_ROOT", str(tmp_path))
    return tmp_path
