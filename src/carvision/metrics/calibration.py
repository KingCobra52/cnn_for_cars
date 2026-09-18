"""Calibration: does a stated confidence mean what it says?

A model that says "90% sure" should be right about 90% of the time it says that. Modern
networks are reliably overconfident, so the raw softmax makes a poor probability. This
matters for the demo: a prediction shown as "97%" should be trustworthy at 97%, and if it
is not, the number is worse than no number.

Expected Calibration Error bins predictions by confidence and measures the average gap
between confidence and accuracy within each bin. Temperature scaling
(:class:`carvision.models.heads.TemperatureScaler`) usually removes most of that gap by
fitting one scalar on validation data, and it cannot change accuracy -- so it is close to
free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_BINS = 15


@dataclass(frozen=True)
class CalibrationResult:
    """Binned reliability statistics.

    Attributes:
        ece: Expected calibration error, the sample-weighted mean gap.
        mce: Maximum calibration error, the worst single bin.
        bin_edges: ``(B + 1,)`` confidence bin boundaries.
        bin_confidence: Mean predicted confidence per bin, NaN where the bin is empty.
        bin_accuracy: Observed accuracy per bin, NaN where the bin is empty.
        bin_count: Number of samples per bin.
        mean_confidence: Mean confidence over all samples.
        accuracy: Overall accuracy, for comparison with ``mean_confidence``.
    """

    ece: float
    mce: float
    bin_edges: np.ndarray
    bin_confidence: np.ndarray
    bin_accuracy: np.ndarray
    bin_count: np.ndarray
    mean_confidence: float
    accuracy: float

    @property
    def overconfidence(self) -> float:
        """Mean confidence minus accuracy. Positive means overconfident."""
        return self.mean_confidence - self.accuracy

    def as_dict(self) -> dict[str, float]:
        """Return the scalar summary, ready for JSON."""
        return {
            "ece": self.ece,
            "mce": self.mce,
            "mean_confidence": self.mean_confidence,
            "accuracy": self.accuracy,
            "overconfidence": self.overconfidence,
        }


def softmax(logits: np.ndarray, *, axis: int = 1) -> np.ndarray:
    """Numerically stable softmax.

    Args:
        logits: Scores.
        axis: Axis to normalise over.

    Returns:
        Probabilities summing to one along ``axis``.
    """
    # Subtracting the max prevents overflow on large logits; it leaves the result
    # unchanged mathematically.
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=axis, keepdims=True)


def compute(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    num_bins: int = DEFAULT_BINS,
) -> CalibrationResult:
    """Compute ECE, MCE and the reliability-diagram bins.

    Args:
        logits: ``(N, C)`` scores.
        labels: ``(N,)`` true classes.
        num_bins: Number of equal-width confidence bins.

    Returns:
        The binned statistics.

    Raises:
        ValueError: If there are no samples or fewer than one bin.
    """
    if len(logits) == 0:
        raise ValueError("Cannot compute calibration on zero samples.")
    if num_bins < 1:
        raise ValueError(f"num_bins must be >= 1, got {num_bins}")

    probabilities = softmax(logits)
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels

    edges = np.linspace(0.0, 1.0, num_bins + 1)
    # Bin by confidence; np.digitize with right=True puts an exact 1.0 in the last bin.
    assignment = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, num_bins - 1)

    bin_confidence = np.full(num_bins, np.nan)
    bin_accuracy = np.full(num_bins, np.nan)
    bin_count = np.zeros(num_bins, dtype=np.int64)

    ece = 0.0
    mce = 0.0
    for index in range(num_bins):
        mask = assignment == index
        count = int(mask.sum())
        bin_count[index] = count
        if count == 0:
            continue

        mean_conf = float(confidence[mask].mean())
        mean_acc = float(correct[mask].mean())
        bin_confidence[index] = mean_conf
        bin_accuracy[index] = mean_acc

        gap = abs(mean_conf - mean_acc)
        ece += (count / len(confidence)) * gap
        mce = max(mce, gap)

    return CalibrationResult(
        ece=ece,
        mce=mce,
        bin_edges=edges,
        bin_confidence=bin_confidence,
        bin_accuracy=bin_accuracy,
        bin_count=bin_count,
        mean_confidence=float(confidence.mean()),
        accuracy=float(correct.mean()),
    )


def fit_temperature(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
) -> float:
    """Fit a temperature on validation logits.

    Fitting on validation rather than test is the whole point: a temperature fit on the
    test set would report a calibration the model does not actually have.

    Args:
        val_logits: ``(N, C)`` validation scores.
        val_labels: ``(N,)`` validation classes.

    Returns:
        The fitted temperature. Above 1.0 means the model was overconfident.
    """
    import torch

    from carvision.models.heads import TemperatureScaler

    return TemperatureScaler().fit(
        torch.from_numpy(np.asarray(val_logits, dtype=np.float32)),
        torch.from_numpy(np.asarray(val_labels, dtype=np.int64)),
    )


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Divide logits by a temperature.

    Args:
        logits: ``(N, C)`` scores.
        temperature: The scalar to divide by.

    Returns:
        The scaled logits.

    Raises:
        ValueError: If the temperature is not positive.
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    return (logits / temperature).astype(np.float32)
