"""Generate ``docs/RESULTS.md`` from the run directories.

The results table is a rendering of `evaluation.json`, never a hand-written document.
That is the whole reason this module exists: a results table maintained by hand drifts
from the runs it describes, and the project this replaced demonstrated exactly that by
plotting an accuracy its model had never scored.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carvision.metrics.bootstrap import seed_spread
from carvision.utils.logging import get_logger
from carvision.utils.paths import ensure_dir, repo_root, runs_dir

logger = get_logger(__name__)

PLACEHOLDER = "—"


@dataclass(frozen=True)
class RunSummary:
    """One evaluated run, flattened for tabulation."""

    name: str
    backbone: str
    head: str
    seed: int
    top1: float
    top5: float
    macro_f1: float
    top1_low: float
    top1_high: float
    evaluation: dict[str, Any]


def collect_runs(root: Path | None = None) -> list[RunSummary]:
    """Read every evaluated run under ``root``.

    Runs that have been trained but not evaluated are skipped rather than partially
    reported, so the table never mixes stale and fresh numbers.

    Args:
        root: Directory holding run subdirectories. Defaults to ``artifacts/runs``.

    Returns:
        One summary per evaluated run.
    """
    summaries: list[RunSummary] = []

    for evaluation_path in sorted((root or runs_dir()).glob("*/evaluation.json")):
        run_dir = evaluation_path.parent
        evaluation = json.loads(evaluation_path.read_text())
        config = json.loads((run_dir / "config.json").read_text())

        summaries.append(
            RunSummary(
                name=run_dir.name,
                backbone=str(config["backbone"]),
                head=str(config["head"]),
                seed=int(config["seed"]),
                top1=float(evaluation["metrics"]["top1"]),
                top5=float(evaluation["metrics"]["top5"]),
                macro_f1=float(evaluation["metrics"]["macro_f1"]),
                top1_low=float(evaluation["top1_interval"]["low"]),
                top1_high=float(evaluation["top1_interval"]["high"]),
                evaluation=evaluation,
            )
        )

    logger.info("Collected %d evaluated runs", len(summaries))
    return summaries


def _percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def _interval(summary: RunSummary) -> str:
    return f"{_percent(summary.top1)} [{100 * summary.top1_low:.1f}, {100 * summary.top1_high:.1f}]"


def headline_table(summaries: list[RunSummary]) -> str:
    """Render one row per (backbone, head), aggregated over seeds.

    Args:
        summaries: Evaluated runs.

    Returns:
        A markdown table.
    """
    grouped: dict[tuple[str, str], list[RunSummary]] = defaultdict(list)
    for summary in summaries:
        grouped[(summary.backbone, summary.head)].append(summary)

    lines = [
        "| Backbone | Head | Top-1 (95% CI) | Top-5 | Macro-F1 | Seeds |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for (backbone, head), runs in sorted(
        grouped.items(), key=lambda item: -max(r.top1 for r in item[1])
    ):
        # Report the median-accuracy seed's interval, with the seed spread beside it, so
        # the row shows both sources of uncertainty rather than the luckiest run.
        ordered = sorted(runs, key=lambda r: r.top1)
        representative = ordered[len(ordered) // 2]
        spread = seed_spread([r.top1 for r in runs])

        lines.append(
            f"| `{backbone}` | {head} | {_interval(representative)} "
            f"| {_percent(representative.top5)} "
            f"| {representative.macro_f1:.3f} "
            f"| {spread['n']} (± {100 * spread['std']:.2f}) |"
        )
    return "\n".join(lines)


def error_structure_table(summary: RunSummary) -> str:
    """Render the error breakdown for one run."""
    labels = {
        "same_model_different_year": "Same model, different year",
        "same_model_different_body": "Same model, different body style",
        "same_make": "Same make, different model",
        "cross_make": "Different make entirely",
    }
    by_kind = summary.evaluation["errors"]["by_kind"]

    lines = ["| Error kind | Count | Share of errors |", "| --- | --- | --- |"]
    for kind, label in labels.items():
        entry = by_kind[kind]
        lines.append(f"| {label} | {entry['count']} | {_percent(entry['share_of_errors'])} |")
    return "\n".join(lines)


def confused_pairs_table(summary: RunSummary, *, limit: int = 15) -> str:
    """Render the most-confused class pairs for one run."""
    lines = ["| True | Predicted | Count |", "| --- | --- | --- |"]
    for pair in summary.evaluation["most_confused_pairs"][:limit]:
        lines.append(f"| {pair['true']} | {pair['predicted']} | {pair['count']} |")
    return "\n".join(lines)


def calibration_table(summaries: list[RunSummary]) -> str:
    """Render ECE before and after temperature scaling, per run."""
    lines = [
        "| Run | ECE (raw) | ECE (scaled) | Temperature |",
        "| --- | --- | --- | --- |",
    ]
    for summary in sorted(summaries, key=lambda s: -s.top1):
        calibration = summary.evaluation["calibration"]
        lines.append(
            f"| `{summary.name}` | {calibration['before']['ece']:.4f} "
            f"| {calibration['after']['ece']:.4f} "
            f"| {calibration['temperature']:.3f} |"
        )
    return "\n".join(lines)


def render(summaries: list[RunSummary]) -> str:
    """Render the whole results document.

    Args:
        summaries: Evaluated runs.

    Returns:
        The markdown document.

    Raises:
        ValueError: If there are no evaluated runs to report.
    """
    if not summaries:
        raise ValueError(
            "No evaluated runs found. Run `carvision sweep` then `carvision eval` first."
        )

    best = max(summaries, key=lambda s: s.top1)
    errors = best.evaluation["errors"]

    return f"""# Results

<!-- Generated by `carvision report`. Do not edit by hand: regenerate it. -->

Stanford Cars, official test split of {best.evaluation["metrics"]["num_samples"]:,} images,
read once. Intervals are 95% bootstrap CIs over the test set; `±` is the standard
deviation across training seeds.

## Headline

{headline_table(summaries)}

## Best run: `{best.name}`

- Top-1: **{_interval(best)}**
- Top-5: {_percent(best.top5)}
- Make-level top-1 (only the manufacturer has to be right): \
{_percent(best.evaluation["make_level_top1"])}
- Share of errors where the true class was still in the top 5: \
{_percent(errors["share_of_errors_in_top5"])}
- Errors where the true class was the model's *second* guess: \
{errors["true_class_was_second_guess"]}

### What kind of wrong is it?

{error_structure_table(best)}

The gap between top-1 and make-level top-1 is the part of the task that is telling
*models within a make* apart, rather than telling makes apart.

### Most confused pairs

{confused_pairs_table(best)}

## Calibration

{calibration_table(summaries)}

Temperature is fit on validation and applied to test. It cannot change accuracy; it only
makes the reported confidence mean closer to what it says.

## Comparing models

Differences between configurations should be read from a **paired** bootstrap
(`carvision compare A B`), not by eye from the intervals above: both models saw the same
images and make correlated errors, so their independent intervals overstate the
uncertainty in the difference. Where a paired interval contains zero, the difference is
not resolved at this test-set size.
"""


def write(output_path: Path | None = None) -> Path:
    """Generate and write the results document.

    Args:
        output_path: Destination. Defaults to ``docs/RESULTS.md``.

    Returns:
        The path written.
    """
    path = output_path or (repo_root() / "docs" / "RESULTS.md")
    # Create the directory rather than assuming it, the way figures._save does. It is
    # absent whenever carvision runs outside a checkout, or when --out names a new
    # location, and the failure is otherwise a bare FileNotFoundError.
    ensure_dir(path.parent)
    path.write_text(render(collect_runs()))
    logger.info("Wrote %s", path)
    return path
