"""Classification metrics.

Top-1 alone is a thin summary for a 196-class problem with a long tail. Top-5 says
whether the right answer was close; macro-F1 weights every class equally rather than
letting well-represented classes dominate; per-class recall is what error analysis is
built on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class ClassificationMetrics:
    """Summary metrics for one set of predictions.

    Attributes:
        top1: Fraction of images whose argmax prediction is correct.
        top5: Fraction whose true class is among the five highest-scoring.
        macro_f1: F1 averaged over classes without weighting by class size.
        balanced_accuracy: Mean per-class recall.
        num_samples: Number of images scored.
    """

    top1: float
    top5: float
    macro_f1: float
    balanced_accuracy: float
    num_samples: int

    def as_dict(self) -> dict[str, float | int]:
        """Return the metrics as a plain dict, ready for JSON."""
        return {
            "top1": self.top1,
            "top5": self.top5,
            "macro_f1": self.macro_f1,
            "balanced_accuracy": self.balanced_accuracy,
            "num_samples": self.num_samples,
        }


def top_k_accuracy(logits: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    """Fraction of samples whose true label is among the ``k`` highest scores.

    Args:
        logits: ``(N, C)`` scores.
        labels: ``(N,)`` true classes.
        k: How many top predictions count as a hit. Clipped to the number of classes.

    Returns:
        The accuracy as a fraction.

    Raises:
        ValueError: If ``k`` is not positive or the shapes disagree.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if len(logits) != len(labels):
        raise ValueError(f"{len(logits)} logit rows but {len(labels)} labels")

    k = min(k, logits.shape[1])
    # argpartition is O(N*C); a full argsort would be wasted work for large C.
    top_k = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    return float((top_k == labels[:, None]).any(axis=1).mean())


def compute(logits: np.ndarray, labels: np.ndarray) -> ClassificationMetrics:
    """Compute the full metric set.

    Args:
        logits: ``(N, C)`` scores.
        labels: ``(N,)`` true classes.

    Returns:
        The metrics.
    """
    from sklearn.metrics import balanced_accuracy_score, f1_score

    predictions = logits.argmax(axis=1)
    return ClassificationMetrics(
        top1=float((predictions == labels).mean()),
        top5=top_k_accuracy(logits, labels, k=5),
        macro_f1=float(f1_score(labels, predictions, average="macro", zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(labels, predictions)),
        num_samples=len(labels),
    )


def per_class_report(
    logits: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Per-class precision, recall, F1 and support.

    The legacy notebook imported ``classification_report`` and never called it. This is
    that call, kept as structured data rather than a printed string so the analysis
    notebook can sort and join on it.

    Args:
        logits: ``(N, C)`` scores.
        labels: ``(N,)`` true classes.
        class_names: Class names indexed by class id.

    Returns:
        A mapping from class name to its metrics.
    """
    from sklearn.metrics import classification_report

    report = classification_report(
        labels,
        logits.argmax(axis=1),
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    return {
        name: values
        for name, values in report.items()
        if name in set(class_names) and isinstance(values, dict)
    }


def confusion(logits: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Return the ``(C, C)`` confusion matrix, rows true and columns predicted.

    Args:
        logits: ``(N, C)`` scores.
        labels: ``(N,)`` true classes.
        num_classes: Total number of classes, so absent classes still get a row.

    Returns:
        An integer confusion matrix.
    """
    from sklearn.metrics import confusion_matrix

    return confusion_matrix(
        labels, logits.argmax(axis=1), labels=list(range(num_classes))
    ).astype(np.int64)


def most_confused_pairs(
    matrix: np.ndarray,
    class_names: Sequence[str],
    *,
    top_n: int = 20,
) -> list[tuple[str, str, int]]:
    """Find the class pairs the model mixes up most.

    On a fine-grained dataset this is the single most informative view of a model's
    behaviour: the pairs are almost never random, and reading them tells you whether the
    model has learned body shape, badge, or just colour.

    Args:
        matrix: A confusion matrix from :func:`confusion`.
        class_names: Class names indexed by class id.
        top_n: How many pairs to return.

    Returns:
        ``(true_name, predicted_name, count)`` tuples, most frequent first.
    """
    errors = matrix.copy()
    np.fill_diagonal(errors, 0)

    flat = np.argsort(errors, axis=None)[::-1][:top_n]
    pairs: list[tuple[str, str, int]] = []
    for index in flat:
        true_id, predicted_id = np.unravel_index(index, errors.shape)
        count = int(errors[true_id, predicted_id])
        if count == 0:
            break
        pairs.append((class_names[int(true_id)], class_names[int(predicted_id)], count))
    return pairs
