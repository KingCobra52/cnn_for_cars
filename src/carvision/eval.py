"""Evaluate a trained run on the test set, once.

The test set is touched here and nowhere else. Model selection, early stopping and
temperature fitting all happen on validation; by the time this module runs, every choice
has been made. That discipline is the difference between a reported number and an
honest one.

Output is a single ``evaluation.json`` per run, which is what generates the results table
and every figure. No number in this repository is typed by hand -- the legacy notebook's
plots hardcoded an accuracy that disagreed with what the model actually scored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from carvision.data.download import load_class_names
from carvision.features import cache
from carvision.interpret import errors as error_analysis
from carvision.metrics import bootstrap, calibration, classification
from carvision.models.heads import build_head
from carvision.utils.logging import get_logger
from carvision.utils.paths import runs_dir
from carvision.utils.seed import set_seed

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Everything computed for one run on the test set."""

    run_name: str
    metrics: classification.ClassificationMetrics
    top1_interval: bootstrap.Interval
    temperature: float
    calibration_before: calibration.CalibrationResult
    calibration_after: calibration.CalibrationResult
    error_summary: dict[str, object]
    confused_pairs: list[tuple[str, str, int]]
    hardest: list[tuple[str, float, int]]
    make_level_top1: float

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "run": self.run_name,
            "metrics": self.metrics.as_dict(),
            "top1_interval": self.top1_interval.as_dict(),
            "make_level_top1": self.make_level_top1,
            "calibration": {
                "temperature": self.temperature,
                "before": self.calibration_before.as_dict(),
                "after": self.calibration_after.as_dict(),
            },
            "errors": self.error_summary,
            "most_confused_pairs": [
                {"true": true, "predicted": predicted, "count": count}
                for true, predicted, count in self.confused_pairs
            ],
            "hardest_classes": [
                {"class": name, "recall": recall, "support": support}
                for name, recall, support in self.hardest
            ],
        }


def _fingerprint(path: Path) -> str:
    """Return a stable fingerprint for an artifact file."""
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(run_dir: Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a trained head from a run directory.

    Args:
        run_dir: Directory written by :func:`carvision.train.train`.

    Returns:
        The head in eval mode, and the checkpoint payload.

    Raises:
        FileNotFoundError: If the run has no checkpoint.
    """
    path = run_dir / "checkpoint.pt"
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path}")

    payload = torch.load(path, map_location="cpu", weights_only=True)
    head = build_head(payload["head_kind"], payload["embedding_dim"], payload["num_classes"])
    head.load_state_dict(payload["state_dict"])
    head.eval()
    return head, payload


@torch.inference_mode()
def _logits_for(head: torch.nn.Module, embeddings: np.ndarray) -> np.ndarray:
    logits: np.ndarray = head(torch.from_numpy(embeddings.astype(np.float32))).numpy()
    return logits


def evaluate_run(
    run_dir: Path,
    *,
    resamples: int = bootstrap.DEFAULT_RESAMPLES,
    seed: int = 0,
) -> EvaluationResult:
    """Score one trained run on the test set and write ``evaluation.json``.

    Args:
        run_dir: The run to evaluate.
        resamples: Bootstrap resamples for the confidence interval.
        seed: Seed for the bootstrap.

    Returns:
        The evaluation result.
    """
    set_seed(seed)
    head, payload = load_run(run_dir)
    backbone = str(payload["backbone"])
    class_names = load_class_names()

    val_x, val_y, _ = cache.load(backbone, "val")
    test_x, test_y, test_ids = cache.load(backbone, "test")

    test_logits = _logits_for(head, test_x)

    # Temperature is fit on validation, then applied to test. Fitting it on test would
    # report a calibration the model does not have.
    temperature = calibration.fit_temperature(_logits_for(head, val_x), val_y)
    calibrated = calibration.apply_temperature(test_logits, temperature)

    metrics = classification.compute(test_logits, test_y)
    correct = test_logits.argmax(axis=1) == test_y

    matrix = classification.confusion(test_logits, test_y, len(class_names))
    cases = error_analysis.collect(test_logits, test_y, test_ids, class_names)

    result = EvaluationResult(
        run_name=run_dir.name,
        metrics=metrics,
        top1_interval=bootstrap.accuracy_interval(correct, resamples=resamples, seed=seed),
        temperature=temperature,
        calibration_before=calibration.compute(test_logits, test_y),
        calibration_after=calibration.compute(calibrated, test_y),
        error_summary=error_analysis.summarise(cases, len(test_y)),
        confused_pairs=classification.most_confused_pairs(matrix, class_names),
        hardest=error_analysis.hardest_classes(test_logits, test_y, class_names),
        make_level_top1=error_analysis.make_level_accuracy(test_logits, test_y, class_names),
    )

    (run_dir / "evaluation.json").write_text(json.dumps(result.as_dict(), indent=2) + "\n")
    np.savez_compressed(
        run_dir / "test_predictions.npz",
        logits=test_logits,
        labels=test_y,
        image_ids=np.array(test_ids),
        confusion=matrix,
    )
    evaluation_path = run_dir / "evaluation.json"
    saved = json.loads(evaluation_path.read_text())
    saved["provenance"] = {
        "checkpoint_sha256": _fingerprint(run_dir / "checkpoint.pt"),
        "predictions_sha256": _fingerprint(run_dir / "test_predictions.npz"),
        "backbone": backbone,
        "test_image_ids": len(test_ids),
    }
    evaluation_path.write_text(json.dumps(saved, indent=2) + "\n")

    logger.info(
        "%s: top-1 %s, top-5 %.2f%%, make-level %.2f%%, ECE %.4f -> %.4f (T=%.3f)",
        run_dir.name,
        result.top1_interval,
        100 * metrics.top5,
        100 * result.make_level_top1,
        result.calibration_before.ece,
        result.calibration_after.ece,
        temperature,
    )
    return result


def evaluate_all(
    *, resamples: int = bootstrap.DEFAULT_RESAMPLES, seed: int = 0
) -> list[EvaluationResult]:
    """Evaluate every expected completed sweep run in deterministic order."""
    manifest_path = runs_dir().parent / "sweep_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("No sweep manifest found; run `carvision sweep` first.")
    expected = json.loads(manifest_path.read_text())["expected_runs"]
    results: list[EvaluationResult] = []
    missing: list[str] = []
    for name in expected:
        path = runs_dir() / name
        if not (path / "checkpoint.pt").exists():
            missing.append(name)
            continue
        results.append(evaluate_run(path, resamples=resamples, seed=seed))
    if missing:
        raise FileNotFoundError("Missing completed runs: " + ", ".join(missing))
    return results


def compare_runs(first: Path, second: Path, *, seed: int = 0) -> bootstrap.Interval:
    """Compare two evaluated runs with a paired bootstrap.

    Both runs must have been evaluated already, so their per-image predictions exist.
    The paired form is the right one here: the two models saw the same images and make
    correlated errors, so comparing independent intervals would overstate the
    uncertainty in their difference.

    Args:
        first: Run directory for model A.
        second: Run directory for model B.
        seed: Seed for the bootstrap.

    Returns:
        The interval for ``top1(A) - top1(B)``. If it contains zero, the two models are
        not distinguishable at this test-set size.

    Raises:
        FileNotFoundError: If either run lacks saved predictions.
        ValueError: If the two runs were evaluated on different images.
    """

    def correctness(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
        path = run_dir / "test_predictions.npz"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `carvision eval` on it first.")
        data = np.load(path, allow_pickle=False)
        ids = data["image_ids"]
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"{run_dir.name} contains duplicate image IDs.")
        return data["logits"].argmax(axis=1) == data["labels"], ids

    correct_a, ids_a = correctness(first)
    correct_b, ids_b = correctness(second)
    if not np.array_equal(ids_a, ids_b):
        raise ValueError(
            f"{first.name} and {second.name} were evaluated on different images, so a "
            f"paired comparison is not valid."
        )

    interval = bootstrap.paired_difference_interval(correct_a, correct_b, seed=seed)
    verdict = "resolved" if interval.excludes_zero else "NOT resolved at this sample size"
    logger.info("%s - %s = %s (%s)", first.name, second.name, interval, verdict)
    return interval


def compare_all(*, baseline: Path | None = None) -> list[dict[str, Any]]:
    """Save all pairwise comparisons for representative evaluated runs."""
    from itertools import combinations

    grouped: dict[tuple[str, str], list[Path]] = {}
    for evaluation in sorted(runs_dir().glob("*/evaluation.json")):
        if evaluation.parent.name == "zeroshot-baseline":
            continue
        config = json.loads((evaluation.parent / "config.json").read_text())
        grouped.setdefault((str(config["backbone"]), str(config["head"])), []).append(evaluation)
    evaluated = []
    for paths in grouped.values():
        ordered = sorted(
            paths,
            key=lambda path: (
                float(json.loads((path.parent / "metrics.json").read_text())["best_val_top1"]),
                -int(json.loads((path.parent / "config.json").read_text())["seed"]),
            ),
        )
        evaluated.append(ordered[len(ordered) // 2])
    if baseline is None:
        candidate = runs_dir() / "zeroshot-baseline"
        baseline = candidate if (candidate / "test_predictions.npz").exists() else None
    if baseline is not None:
        evaluated.append(baseline / "evaluation.json" if baseline.is_dir() else baseline)
    rows: list[dict[str, Any]] = []
    for first, second in combinations(evaluated, 2):
        interval = compare_runs(first.parent, second.parent)
        rows.append(
            {
                "first": first.parent.name,
                "second": second.parent.name,
                **interval.as_dict(),
                "label": "unadjusted exploratory",
            }
        )
    output = runs_dir().parent / "comparisons.json"
    output.write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def find_best_run() -> Path:
    """Return the run with the highest validation accuracy.

    Selection is on validation, never on test -- otherwise "the best run" would be
    chosen using the very data it is then reported on.

    Returns:
        The best run's directory.

    Raises:
        FileNotFoundError: If no runs have been trained.
    """
    candidates = sorted(runs_dir().glob("*/metrics.json"))
    if not candidates:
        raise FileNotFoundError(f"No runs under {runs_dir()}. Train one first.")

    def val_top1(path: Path) -> float:
        return float(json.loads(path.read_text())["best_val_top1"])

    return max(candidates, key=val_top1).parent
