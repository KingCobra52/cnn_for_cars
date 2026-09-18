"""Error bucketing and the class-name parser it depends on."""

from __future__ import annotations

import numpy as np
import pytest

from carvision.interpret.errors import (
    classify_error,
    collect,
    hardest_classes,
    make_level_accuracy,
    summarise,
)
from carvision.models.zeroshot import parse_class_name

# ------------------------------------------------------------------ name parsing


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "2012 Tesla Model S Sedan",
            {"make": "Tesla", "model": "Model S", "body": "Sedan", "year": "2012"},
        ),
        (
            "2012 BMW 3 Series Sedan",
            {"make": "BMW", "model": "3 Series", "body": "Sedan", "year": "2012"},
        ),
        (
            "AM General Hummer SUV 2000",
            {"make": "AM", "model": "General Hummer", "body": "SUV", "year": "2000"},
        ),
        (
            "2007 Ford Focus Sedan",
            {"make": "Ford", "model": "Focus", "body": "Sedan", "year": "2007"},
        ),
    ],
)
def test_parse_class_name(name: str, expected: dict[str, str]) -> None:
    parsed = parse_class_name(name)
    for key, value in expected.items():
        assert parsed[key] == value, f"{key} of {name!r}"


def test_parse_keeps_the_original_name() -> None:
    assert parse_class_name("2012 Tesla Model S Sedan")["full"] == "2012 Tesla Model S Sedan"


def test_parse_tolerates_a_name_with_no_body_style() -> None:
    parsed = parse_class_name("2012 Fisker Karma")
    assert parsed["make"] == "Fisker"
    assert parsed["body"] == ""
    assert parsed["model"] == "Karma"


def test_parse_tolerates_a_name_with_no_year() -> None:
    assert parse_class_name("Ford Focus Sedan")["year"] == ""


# ------------------------------------------------------------------ bucketing


@pytest.mark.parametrize(
    ("true", "predicted", "kind"),
    [
        # Same make, model and body; only the year differs.
        ("2012 Ford Focus Sedan", "2007 Ford Focus Sedan", "same_model_different_year"),
        # Same make and model, different body style.
        ("2012 Ford Focus Sedan", "2012 Ford Focus Coupe", "same_model_different_body"),
        # Same make, different model.
        ("2012 Ford Focus Sedan", "2012 Ford Fiesta Sedan", "same_make"),
        # Different make entirely: the least excusable error.
        ("2012 Ford Focus Sedan", "2012 Tesla Model S Sedan", "cross_make"),
    ],
)
def test_classify_error(true: str, predicted: str, kind: str) -> None:
    assert classify_error(true, predicted) == kind


def test_classification_is_case_insensitive() -> None:
    assert classify_error("2012 FORD Focus Sedan", "2007 ford focus sedan") == (
        "same_model_different_year"
    )


# ------------------------------------------------------------------ collection


@pytest.fixture
def scenario() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Four images: one correct, one same-make error, one cross-make, one near miss."""
    class_names = [
        "2012 Ford Focus Sedan",
        "2012 Ford Fiesta Sedan",
        "2012 Tesla Model S Sedan",
        "2007 Ford Focus Sedan",
    ]
    logits = np.array(
        [
            [5.0, 1.0, 0.0, 0.0],  # true 0, predicted 0 -- correct
            [1.0, 5.0, 0.0, 0.0],  # true 0, predicted 1 -- same make
            [1.0, 0.0, 5.0, 0.0],  # true 0, predicted 2 -- cross make
            [4.0, 0.0, 0.0, 5.0],  # true 0, predicted 3 -- year; true rank 1
        ]
    )
    labels = np.array([0, 0, 0, 0])
    image_ids = ["img_a", "img_b", "img_c", "img_d"]
    return logits, labels, image_ids, class_names


def test_collect_finds_only_the_errors(
    scenario: tuple[np.ndarray, np.ndarray, list[str], list[str]],
) -> None:
    cases = collect(*scenario)
    assert len(cases) == 3
    assert "img_a" not in {case.image_id for case in cases}


def test_collect_records_the_true_rank(
    scenario: tuple[np.ndarray, np.ndarray, list[str], list[str]],
) -> None:
    """img_d's true class was the model's second choice, i.e. rank 1."""
    cases = {case.image_id: case for case in collect(*scenario)}
    assert cases["img_d"].true_rank == 1


def test_collect_buckets_each_error(
    scenario: tuple[np.ndarray, np.ndarray, list[str], list[str]],
) -> None:
    cases = {case.image_id: case.kind for case in collect(*scenario)}
    assert cases["img_b"] == "same_make"
    assert cases["img_c"] == "cross_make"
    assert cases["img_d"] == "same_model_different_year"


def test_collect_sorts_by_descending_confidence(
    scenario: tuple[np.ndarray, np.ndarray, list[str], list[str]],
) -> None:
    confidences = [case.confidence for case in collect(*scenario)]
    assert confidences == sorted(confidences, reverse=True)


def test_summarise_shares_add_up(
    scenario: tuple[np.ndarray, np.ndarray, list[str], list[str]],
) -> None:
    cases = collect(*scenario)
    summary = summarise(cases, total_samples=4)

    assert summary["num_errors"] == 3
    assert summary["error_rate"] == pytest.approx(0.75)
    shares = [entry["share_of_errors"] for entry in summary["by_kind"].values()]  # type: ignore[index,union-attr]
    assert sum(shares) == pytest.approx(1.0)


def test_summarise_handles_a_perfect_model() -> None:
    """No errors must not divide by zero."""
    summary = summarise([], total_samples=100)
    assert summary["num_errors"] == 0
    assert summary["error_rate"] == 0.0
    assert summary["share_of_errors_in_top5"] == 0.0


def test_make_level_accuracy_is_at_least_top1(
    scenario: tuple[np.ndarray, np.ndarray, list[str], list[str]],
) -> None:
    """Getting the make right is a strictly easier task than getting the class right."""
    logits, labels, _, class_names = scenario
    top1 = float((logits.argmax(axis=1) == labels).mean())
    assert make_level_accuracy(logits, labels, class_names) >= top1


def test_make_level_accuracy_forgives_within_make_errors(
    scenario: tuple[np.ndarray, np.ndarray, list[str], list[str]],
) -> None:
    """Three of four predictions are Fords; only the Tesla is a make-level miss."""
    logits, labels, _, class_names = scenario
    assert make_level_accuracy(logits, labels, class_names) == pytest.approx(0.75)


def test_hardest_classes_puts_the_worst_first() -> None:
    # Class 0 is always right, class 1 always wrong.
    logits = np.array([[5.0, 0.0], [5.0, 0.0], [5.0, 0.0], [5.0, 0.0]])
    labels = np.array([0, 0, 1, 1])
    ranked = hardest_classes(logits, labels, ["good", "bad"], top_n=2)

    assert ranked[0][0] == "bad"
    assert ranked[0][1] == pytest.approx(0.0)
    assert ranked[1][1] == pytest.approx(1.0)
