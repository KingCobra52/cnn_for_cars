"""Confidence intervals by bootstrap resampling.

A single accuracy number invites comparisons it cannot support. On an 8,041-image test
set, the standard error on a top-1 around 85% is roughly 0.4 points, so two models
differing by half a point are not distinguishable -- and reporting them as "84.9% vs
85.4%, the second is better" is a claim the data does not license.

The bootstrap gives the interval directly and without distributional assumptions:
resample the test set with replacement, recompute the metric, and read the percentiles
of the resulting distribution.

The paired variant matters as much as the interval. Two models evaluated on the same
images make correlated errors -- both find the same photographs hard -- so comparing
their independent intervals understates how well the difference is resolved. Resampling
the *same* images for both and taking the interval of the difference is the correct
comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_RESAMPLES = 1000
DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True)
class Interval:
    """A point estimate with a confidence interval.

    Attributes:
        point: The metric computed on the observed data.
        low: Lower percentile of the bootstrap distribution.
        high: Upper percentile.
        confidence: The nominal coverage, e.g. 0.95.
    """

    point: float
    low: float
    high: float
    confidence: float = DEFAULT_CONFIDENCE

    def __str__(self) -> str:
        return f"{100 * self.point:.2f}% [{100 * self.low:.2f}, {100 * self.high:.2f}]"

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval lies wholly above or below zero.

        For a difference interval, this is the question "is the gap resolved?".
        """
        return (self.low > 0) or (self.high < 0)

    def as_dict(self) -> dict[str, float]:
        """Return the interval as a plain dict, ready for JSON."""
        return {
            "point": self.point,
            "low": self.low,
            "high": self.high,
            "confidence": self.confidence,
        }


def accuracy_interval(
    correct: np.ndarray,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> Interval:
    """Bootstrap a confidence interval for an accuracy.

    Args:
        correct: Boolean array, true where the prediction was right.
        resamples: Number of bootstrap resamples.
        confidence: Nominal coverage, e.g. 0.95.
        seed: Seed for the resampling, so the interval is reproducible.

    Returns:
        The point estimate and interval.

    Raises:
        ValueError: If ``correct`` is empty or ``confidence`` is not in (0, 1).
    """
    correct = np.asarray(correct, dtype=bool)
    if correct.size == 0:
        raise ValueError("Cannot bootstrap an empty array.")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    rng = np.random.default_rng(seed)
    n = len(correct)
    # Draw all resamples at once: (resamples, n) indices, then mean along axis 1.
    indices = rng.integers(0, n, size=(resamples, n))
    distribution = correct[indices].mean(axis=1)

    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(distribution, [tail, 1.0 - tail])
    return Interval(float(correct.mean()), float(low), float(high), confidence)


def paired_difference_interval(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> Interval:
    """Bootstrap the accuracy difference between two models on the same images.

    Resamples image indices once per iteration and applies them to both models, which
    preserves the correlation between their errors. An interval that excludes zero means
    the difference is resolved at this sample size; one that straddles zero means it is
    not, whatever the point estimates look like.

    Args:
        correct_a: Boolean correctness for model A.
        correct_b: Boolean correctness for model B, aligned image-for-image with A.
        resamples: Number of bootstrap resamples.
        confidence: Nominal coverage.
        seed: Seed for the resampling.

    Returns:
        The interval for ``accuracy(a) - accuracy(b)``.

    Raises:
        ValueError: If the two arrays are not the same length, which would mean they
            were not evaluated on the same images.
    """
    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)
    if len(correct_a) != len(correct_b):
        raise ValueError(
            f"Paired comparison needs the same images: got {len(correct_a)} and {len(correct_b)}."
        )

    rng = np.random.default_rng(seed)
    n = len(correct_a)
    indices = rng.integers(0, n, size=(resamples, n))
    distribution = correct_a[indices].mean(axis=1) - correct_b[indices].mean(axis=1)

    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(distribution, [tail, 1.0 - tail])
    point = float(correct_a.mean() - correct_b.mean())
    return Interval(point, float(low), float(high), confidence)


def seed_spread(values: list[float]) -> dict[str, float]:
    """Summarise a metric measured across repeated seeds.

    Distinct from the bootstrap interval, and worth reporting alongside it: the
    bootstrap captures uncertainty from which *images* are in the test set, while this
    captures uncertainty from the training run's own randomness.

    Args:
        values: One metric value per seed.

    Returns:
        Mean, sample standard deviation, min, max and count.

    Raises:
        ValueError: If ``values`` is empty.
    """
    if not values:
        raise ValueError("Need at least one value.")
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        # ddof=1: these are a sample of possible runs, not the population.
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
        "n": len(array),
    }
