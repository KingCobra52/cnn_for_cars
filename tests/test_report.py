"""The results document is a rendering of run output, not a hand-written file."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carvision.report import RunSummary, collect_runs, headline_table, render


def write_run(
    root: Path,
    name: str,
    *,
    backbone: str,
    head: str,
    seed: int,
    top1: float,
    val_top1: float | None = None,
) -> None:
    """Create a minimal but complete evaluated-run directory.

    ``val_top1`` defaults to tracking ``top1`` so most callers need not think about it.
    Tests that care about selection discipline set the two apart deliberately.
    """
    run_dir = root / name
    run_dir.mkdir(parents=True)

    (run_dir / "config.json").write_text(
        json.dumps({"backbone": backbone, "head": head, "seed": seed})
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "best_val_top1": top1 if val_top1 is None else val_top1,
                "best_epoch": 3,
                "epochs_run": 4,
                "history": [],
            }
        )
    )
    (run_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "run": name,
                "metrics": {
                    "top1": top1,
                    "top5": min(1.0, top1 + 0.1),
                    "macro_f1": top1 - 0.01,
                    "num_samples": 8041,
                },
                "top1_interval": {"point": top1, "low": top1 - 0.01, "high": top1 + 0.01},
                "make_level_top1": min(1.0, top1 + 0.08),
                "calibration": {
                    "temperature": 1.4,
                    "before": {"ece": 0.08},
                    "after": {"ece": 0.01},
                },
                "errors": {
                    "by_kind": {
                        "same_model_different_year": {"count": 40, "share_of_errors": 0.4},
                        "same_model_different_body": {"count": 20, "share_of_errors": 0.2},
                        "same_make": {"count": 30, "share_of_errors": 0.3},
                        "cross_make": {"count": 10, "share_of_errors": 0.1},
                    },
                    "share_of_errors_in_top5": 0.65,
                    "true_class_was_second_guess": 33,
                },
                "most_confused_pairs": [
                    {
                        "true": "2012 Ford Focus Sedan",
                        "predicted": "2007 Ford Focus Sedan",
                        "count": 6,
                    }
                ],
                "hardest_classes": [],
            }
        )
    )


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    write_run(
        root, "dinov2-linear-seed0", backbone="dinov2_vits14", head="linear", seed=0, top1=0.86
    )
    write_run(
        root, "dinov2-linear-seed1", backbone="dinov2_vits14", head="linear", seed=1, top1=0.85
    )
    write_run(
        root, "dinov2-linear-seed2", backbone="dinov2_vits14", head="linear", seed=2, top1=0.87
    )
    write_run(root, "resnet-linear-seed0", backbone="resnet50", head="linear", seed=0, top1=0.55)
    return root


def test_collects_every_evaluated_run(runs: Path) -> None:
    assert len(collect_runs(runs)) == 4


def test_unevaluated_runs_are_skipped(runs: Path, tmp_path: Path) -> None:
    """A trained-but-not-evaluated run must not appear half-reported."""
    (runs / "pending-run").mkdir()
    (runs / "pending-run" / "config.json").write_text(
        json.dumps({"backbone": "resnet50", "head": "mlp", "seed": 0})
    )
    assert len(collect_runs(runs)) == 4


def test_headline_groups_seeds_into_one_row(runs: Path) -> None:
    table = headline_table(collect_runs(runs))
    body = [line for line in table.splitlines() if line.startswith("| `")]

    assert len(body) == 2, "three dinov2 seeds should collapse into a single row"
    assert "dinov2_vits14" in body[0], "the stronger backbone should be listed first"
    assert body[0].endswith("| 3 |"), "the row should report how many seeds it aggregates"
    assert "86.00% ± 1.00%" in body[0], "the row should report the seed mean and sample SD"


def test_headline_labels_representative_and_aggregate_metrics(runs: Path) -> None:
    table = headline_table(collect_runs(runs))

    assert "Representative top-1 (95% CI)" in table
    assert "Top-1 mean ± SD" in table
    assert "Representative top-5" in table
    assert "Representative macro-F1" in table


def test_headline_uses_the_median_seed_not_the_best(runs: Path) -> None:
    """Reporting the luckiest seed is the standard way to overstate a result."""
    table = headline_table(collect_runs(runs))
    assert "87.00%" not in table
    assert "86.00%" in table


def test_selection_follows_validation_not_test(tmp_path: Path) -> None:
    """The headline run must be chosen on validation, never on test.

    Picking the run with the best *test* score and then reporting that score is
    selection on the test set. Here the run that wins on validation is deliberately the
    weaker one on test, so a test-based selector would pick the other and be caught.
    """
    root = tmp_path / "runs"
    write_run(root, "wins-on-val", backbone="a", head="linear", seed=0, top1=0.70, val_top1=0.95)
    write_run(root, "wins-on-test", backbone="b", head="linear", seed=0, top1=0.90, val_top1=0.60)

    summaries = collect_runs(root)
    document = render(summaries)

    assert "wins-on-val" in document
    assert "Best run: `wins-on-val`" in document


def test_render_produces_a_complete_document(runs: Path) -> None:
    document = render(collect_runs(runs))

    for section in ("## Headline", "## Calibration", "What kind of wrong", "paired"):
        assert section in document

    assert "Do not edit by hand" in document
    assert "8,041" in document
    assert "median-validation seed" in document
    assert "summarises all training seeds" in document


def test_render_refuses_with_no_runs() -> None:
    with pytest.raises(ValueError, match="No evaluated runs"):
        render([])


def test_summaries_carry_the_interval(runs: Path) -> None:
    summary: RunSummary = max(collect_runs(runs), key=lambda s: s.top1)
    assert summary.top1_low < summary.top1 < summary.top1_high
