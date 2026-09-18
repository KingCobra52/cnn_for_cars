"""Error analysis: what kind of wrong is the model?

For a 196-class fine-grained problem, "85% accurate" says almost nothing about model
behaviour. The interesting question is the structure of the remaining 15%.

Stanford Cars class names carry that structure explicitly -- ``2012 Tesla Model S
Sedan`` names a make, a model, a body style and a year. So every error can be bucketed:
did the model confuse two years of the same model, two body styles of the same model,
two models from the same make, or two entirely unrelated cars? A model whose errors are
overwhelmingly within-make has learned something real about car identity and is failing
on genuinely hard distinctions. A model making cross-make errors is failing at something
much more basic.

That distinction is the finding worth putting in a README, and it is not visible in an
accuracy number.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from carvision.metrics.calibration import softmax
from carvision.models.zeroshot import parse_class_name

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Error categories, from most to least excusable.
ERROR_KINDS = ("same_model_different_year", "same_model_different_body", "same_make", "cross_make")


@dataclass(frozen=True)
class ErrorCase:
    """One misclassified image.

    Attributes:
        image_id: Which image, so the case can be rendered.
        true_name: Ground-truth class name.
        predicted_name: Predicted class name.
        confidence: Probability assigned to the prediction.
        true_rank: Zero-based rank of the true class in the score ordering. A rank of 1
            means the model's second guess was right.
        kind: One of :data:`ERROR_KINDS`.
    """

    image_id: str
    true_name: str
    predicted_name: str
    confidence: float
    true_rank: int
    kind: str


def classify_error(true_name: str, predicted_name: str) -> str:
    """Bucket a confusion by how closely related the two classes are.

    Args:
        true_name: Ground-truth class name.
        predicted_name: Predicted class name.

    Returns:
        One of :data:`ERROR_KINDS`.
    """
    true = parse_class_name(true_name)
    predicted = parse_class_name(predicted_name)

    if true["make"].lower() != predicted["make"].lower():
        return "cross_make"
    if true["model"].lower() != predicted["model"].lower():
        return "same_make"
    # Same make and model from here: the difference is year or body style.
    if true["body"].lower() != predicted["body"].lower():
        return "same_model_different_body"
    return "same_model_different_year"


def collect(
    logits: np.ndarray,
    labels: np.ndarray,
    image_ids: Sequence[str],
    class_names: Sequence[str],
) -> list[ErrorCase]:
    """Gather every misclassified image with its context.

    Args:
        logits: ``(N, C)`` test scores.
        labels: ``(N,)`` true classes.
        image_ids: Image ids aligned with the rows of ``logits``.
        class_names: Class names indexed by class id.

    Returns:
        One :class:`ErrorCase` per wrong prediction, ordered by descending confidence --
        so the most confidently wrong come first, which is where the interesting failures
        are.
    """
    probabilities = softmax(logits)
    predictions = probabilities.argmax(axis=1)

    # Rank of every class per row; the true class's rank tells us how close a miss was.
    order = np.argsort(-logits, axis=1)
    ranks = np.argsort(order, axis=1)

    cases: list[ErrorCase] = []
    for index in np.flatnonzero(predictions != labels):
        true_name = class_names[int(labels[index])]
        predicted_name = class_names[int(predictions[index])]
        cases.append(
            ErrorCase(
                image_id=str(image_ids[index]),
                true_name=true_name,
                predicted_name=predicted_name,
                confidence=float(probabilities[index, predictions[index]]),
                true_rank=int(ranks[index, labels[index]]),
                kind=classify_error(true_name, predicted_name),
            )
        )

    cases.sort(key=lambda case: case.confidence, reverse=True)
    return cases


def summarise(cases: Sequence[ErrorCase], total_samples: int) -> dict[str, object]:
    """Summarise where the error mass sits.

    Args:
        cases: The collected errors.
        total_samples: Size of the evaluated split, so shares can be expressed against
            the whole set rather than only against the errors.

    Returns:
        Counts and shares per error kind, plus near-miss statistics.
    """
    kind_counts = Counter(case.kind for case in cases)
    num_errors = len(cases)

    near_miss = sum(1 for case in cases if case.true_rank < 5)
    second_guess = sum(1 for case in cases if case.true_rank == 1)

    return {
        "total_samples": total_samples,
        "num_errors": num_errors,
        "error_rate": num_errors / total_samples if total_samples else 0.0,
        "by_kind": {
            kind: {
                "count": kind_counts.get(kind, 0),
                "share_of_errors": kind_counts.get(kind, 0) / num_errors if num_errors else 0.0,
            }
            for kind in ERROR_KINDS
        },
        # "The answer was the model's second guess" and "the answer was in the top five"
        # are the two numbers that say whether the remaining errors are near misses.
        "true_class_was_second_guess": second_guess,
        "true_class_in_top5": near_miss,
        "share_of_errors_in_top5": near_miss / num_errors if num_errors else 0.0,
    }


def hardest_classes(
    logits: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    *,
    top_n: int = 15,
) -> list[tuple[str, float, int]]:
    """Rank classes by recall, worst first.

    Args:
        logits: ``(N, C)`` scores.
        labels: ``(N,)`` true classes.
        class_names: Class names indexed by class id.
        top_n: How many classes to return.

    Returns:
        ``(class_name, recall, support)`` tuples for the worst classes.
    """
    predictions = logits.argmax(axis=1)
    rows: list[tuple[str, float, int]] = []

    for class_id, name in enumerate(class_names):
        mask = labels == class_id
        support = int(mask.sum())
        if support == 0:
            continue
        rows.append((name, float((predictions[mask] == class_id).mean()), support))

    rows.sort(key=lambda row: (row[1], -row[2]))
    return rows[:top_n]


def confident_mistakes(cases: Sequence[ErrorCase], *, top_n: int = 12) -> list[ErrorCase]:
    """Return the most confidently wrong predictions.

    These are the cases worth looking at by eye. A model that is 99% certain and wrong is
    either seeing a genuinely ambiguous image, or has latched onto a spurious cue -- and
    a Grad-CAM overlay usually tells you which within seconds.

    Args:
        cases: Collected errors (already sorted by confidence).
        top_n: How many to return.

    Returns:
        The ``top_n`` highest-confidence errors.
    """
    return list(cases[:top_n])


def make_level_accuracy(
    logits: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
) -> float:
    """Accuracy when only the make has to be right.

    The gap between this and top-1 quantifies how much of the task is telling makes
    apart versus telling models within a make apart -- which is the honest way to say
    "the remaining errors are the hard part".

    Args:
        logits: ``(N, C)`` scores.
        labels: ``(N,)`` true classes.
        class_names: Class names indexed by class id.

    Returns:
        The make-level accuracy as a fraction.
    """
    makes = np.array([parse_class_name(name)["make"].lower() for name in class_names])
    predictions = logits.argmax(axis=1)
    return float((makes[predictions] == makes[labels]).mean())
