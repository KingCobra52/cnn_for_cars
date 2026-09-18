"""Metrics verified against hand-computed cases, not against themselves."""

from __future__ import annotations

import numpy as np
import pytest

from carvision.metrics import bootstrap, calibration, classification

# ------------------------------------------------------------------ top-k


def test_top1_and_top5_on_a_hand_built_case() -> None:
    # Row 0: true class 0 is top-1. Row 1: true class 2 is ranked third -- in the top 5
    # but not top 1. Row 2: true class 1 is last.
    logits = np.array(
        [
            [9.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [5.0, 4.0, 3.0, 0.0, 0.0, 0.0],
            [5.0, -9.0, 3.0, 2.0, 1.0, 0.0],
        ]
    )
    labels = np.array([0, 2, 1])

    assert classification.top_k_accuracy(logits, labels, k=1) == pytest.approx(1 / 3)
    assert classification.top_k_accuracy(logits, labels, k=3) == pytest.approx(2 / 3)
    assert classification.top_k_accuracy(logits, labels, k=5) == pytest.approx(2 / 3)
    assert classification.top_k_accuracy(logits, labels, k=6) == pytest.approx(1.0)


def test_top_k_is_monotonic_in_k() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(200, 20))
    labels = rng.integers(0, 20, size=200)
    values = [classification.top_k_accuracy(logits, labels, k=k) for k in range(1, 21)]
    assert values == sorted(values)
    assert values[-1] == pytest.approx(1.0)


def test_k_larger_than_class_count_is_clipped() -> None:
    logits = np.eye(3)
    labels = np.arange(3)
    assert classification.top_k_accuracy(logits, labels, k=99) == pytest.approx(1.0)


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_rejected(k: int) -> None:
    with pytest.raises(ValueError, match="k must be"):
        classification.top_k_accuracy(np.eye(3), np.arange(3), k=k)


def test_mismatched_lengths_rejected() -> None:
    with pytest.raises(ValueError, match="labels"):
        classification.top_k_accuracy(np.eye(3), np.arange(2))


# ------------------------------------------------------------------ confusion


def test_most_confused_pairs_ignores_the_diagonal() -> None:
    # Class 0 is confused with class 1 seven times; the diagonal is much larger and must
    # not appear.
    matrix = np.array([[90, 7, 1], [2, 80, 0], [0, 3, 70]])
    pairs = classification.most_confused_pairs(matrix, ["a", "b", "c"], top_n=3)

    assert pairs[0] == ("a", "b", 7)
    assert all(true != predicted for true, predicted, _ in pairs)


def test_most_confused_pairs_stops_at_zero() -> None:
    """Asking for more pairs than exist must not pad the list with zero-count entries."""
    matrix = np.array([[10, 1, 0], [0, 10, 0], [0, 0, 10]])
    pairs = classification.most_confused_pairs(matrix, ["a", "b", "c"], top_n=10)
    assert pairs == [("a", "b", 1)]


# ------------------------------------------------------------------ bootstrap


def test_interval_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(0)
    correct = rng.random(2000) < 0.85
    interval = bootstrap.accuracy_interval(correct, seed=0)

    assert interval.low <= interval.point <= interval.high
    assert interval.point == pytest.approx(correct.mean())


def test_interval_narrows_as_the_sample_grows() -> None:
    """The width should scale roughly as 1/sqrt(n); only the direction is asserted."""
    rng = np.random.default_rng(0)
    small = bootstrap.accuracy_interval(rng.random(100) < 0.8, seed=0)
    large = bootstrap.accuracy_interval(rng.random(10000) < 0.8, seed=0)
    assert (large.high - large.low) < (small.high - small.low)


def test_interval_is_reproducible() -> None:
    correct = np.random.default_rng(0).random(500) < 0.8
    first = bootstrap.accuracy_interval(correct, seed=7)
    second = bootstrap.accuracy_interval(correct, seed=7)
    assert (first.low, first.high) == (second.low, second.high)


def test_a_perfect_model_has_a_degenerate_interval() -> None:
    interval = bootstrap.accuracy_interval(np.ones(100, dtype=bool))
    assert interval.point == 1.0
    assert interval.low == interval.high == 1.0


def test_empty_input_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        bootstrap.accuracy_interval(np.array([], dtype=bool))


def test_paired_interval_resolves_a_real_difference() -> None:
    """A large, consistent gap must produce an interval that excludes zero."""
    rng = np.random.default_rng(0)
    a = rng.random(4000) < 0.90
    b = rng.random(4000) < 0.70
    interval = bootstrap.paired_difference_interval(a, b, seed=0)

    assert interval.point > 0
    assert interval.excludes_zero


def test_paired_interval_does_not_resolve_identical_models() -> None:
    """Comparing a model against itself must give an interval containing zero."""
    correct = np.random.default_rng(0).random(1000) < 0.8
    interval = bootstrap.paired_difference_interval(correct, correct, seed=0)

    assert interval.point == pytest.approx(0.0)
    assert not interval.excludes_zero


def test_paired_interval_requires_the_same_images() -> None:
    with pytest.raises(ValueError, match="same images"):
        bootstrap.paired_difference_interval(np.ones(10, dtype=bool), np.ones(9, dtype=bool))


def test_seed_spread_uses_the_sample_standard_deviation() -> None:
    spread = bootstrap.seed_spread([0.80, 0.82, 0.84])
    assert spread["mean"] == pytest.approx(0.82)
    assert spread["std"] == pytest.approx(0.02)  # ddof=1
    assert spread["n"] == 3


def test_seed_spread_of_one_value_has_zero_spread() -> None:
    assert bootstrap.seed_spread([0.8])["std"] == 0.0


# ------------------------------------------------------------------ calibration


def test_softmax_rows_sum_to_one() -> None:
    probabilities = calibration.softmax(np.random.default_rng(0).normal(size=(10, 7)))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6)


def test_softmax_is_stable_on_huge_logits() -> None:
    """Without the max-subtraction this overflows to NaN."""
    probabilities = calibration.softmax(np.array([[1000.0, 1001.0, 999.0]]))
    assert np.isfinite(probabilities).all()
    assert probabilities.sum() == pytest.approx(1.0)


def test_a_perfectly_calibrated_model_has_near_zero_ece() -> None:
    """Build predictions whose confidence matches their accuracy by construction."""
    rng = np.random.default_rng(0)
    n = 20000
    confidence = rng.uniform(0.5, 1.0, size=n)
    correct = rng.random(n) < confidence

    # Two-class logits that realise exactly `confidence` for the chosen class.
    logits = np.zeros((n, 2))
    logits[:, 0] = np.log(confidence)
    logits[:, 1] = np.log(1 - confidence)
    labels = np.where(correct, 0, 1)

    assert calibration.compute(logits, labels).ece < 0.02


def test_an_overconfident_model_is_detected() -> None:
    rng = np.random.default_rng(0)
    n = 5000
    labels = rng.integers(0, 4, size=n)
    logits = rng.normal(scale=0.1, size=(n, 4))
    # Right only 60% of the time, but always with an enormous margin.
    correct = rng.random(n) < 0.6
    logits[np.arange(n), np.where(correct, labels, (labels + 1) % 4)] += 15.0

    result = calibration.compute(logits, labels)
    assert result.overconfidence > 0.3
    assert result.ece > 0.3


def test_temperature_scaling_reduces_ece_without_moving_accuracy() -> None:
    rng = np.random.default_rng(0)
    n = 6000
    labels = rng.integers(0, 5, size=n)
    logits = rng.normal(scale=0.1, size=(n, 5))
    correct = rng.random(n) < 0.65
    logits[np.arange(n), np.where(correct, labels, (labels + 1) % 5)] += 10.0

    # Fit on the first half, evaluate on the second: never fit calibration on test.
    half = n // 2
    temperature = calibration.fit_temperature(logits[:half], labels[:half])
    scaled = calibration.apply_temperature(logits[half:], temperature)

    before = calibration.compute(logits[half:], labels[half:])
    after = calibration.compute(scaled, labels[half:])

    assert after.ece < before.ece
    assert after.accuracy == pytest.approx(before.accuracy), "temperature must not move argmax"


def test_bin_counts_account_for_every_sample() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(500, 6))
    labels = rng.integers(0, 6, size=500)
    result = calibration.compute(logits, labels, num_bins=15)
    assert int(result.bin_count.sum()) == 500


def test_calibration_rejects_empty_and_zero_bins() -> None:
    with pytest.raises(ValueError, match="zero samples"):
        calibration.compute(np.zeros((0, 3)), np.array([], dtype=int))
    with pytest.raises(ValueError, match="num_bins"):
        calibration.compute(np.eye(3), np.arange(3), num_bins=0)


def test_apply_temperature_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        calibration.apply_temperature(np.eye(3), 0.0)
