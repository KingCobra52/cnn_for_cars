"""Validate and assemble publication inputs without rerunning experiments."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

from carvision.utils.artifacts import (
    active_names,
    fingerprint,
    read,
    representatives,
    validate,
    validate_evaluation,
)
from carvision.utils.paths import artifacts_dir, figures_dir, repo_root, runs_dir


def gather(strict: bool) -> tuple[str, list[str]]:
    """Render supporting results, rejecting incomplete inputs in strict mode."""
    from carvision.eval import find_best_run

    errors: list[str] = []
    sections: list[str] = []

    def require(path: Path, *, evaluation: bool = False) -> dict[str, Any]:
        try:
            data = validate_evaluation(path.parent) if evaluation else read(path)
            if not evaluation:
                validate(data.get("provenance", {}))
            return data
        except (ValueError, OSError, KeyError, RuntimeError) as exc:
            errors.append(f"{path.name}: {exc}")
            return {}

    expected = active_names()
    if not expected:
        errors.append("Missing or empty sweep manifest; rerun carvision sweep.")
    for name in sorted(expected or []):
        evaluation = require(runs_dir() / name / "evaluation.json", evaluation=True)
        if evaluation:
            for filename in (
                "checkpoint.pt",
                "config.json",
                "metrics.json",
                "test_predictions.npz",
            ):
                if not (runs_dir() / name / filename).is_file():
                    errors.append(f"{name}: missing {filename}")
    baseline = require(runs_dir() / "zeroshot-baseline/evaluation.json", evaluation=True)
    if baseline:
        if baseline.get("split") != "test" or not baseline.get("baseline"):
            errors.append("Publication requires a test zero-shot baseline.")
        m, interval = baseline["metrics"], baseline["top1_interval"]
        sections.append(
            "## Zero-shot baseline\n\n| Approach | Top-1 (95% CI) | Top-5 | Macro-F1 |\n"
            "| --- | --- | --- | --- |\n"
            f"| CLIP zero-shot | {m['top1']:.2%} "
            f"[{interval['low']:.2%}, {interval['high']:.2%}] | "
            f"{m['top5']:.2%} | {m['macro_f1']:.3f} |"
        )
    comparison = require(artifacts_dir() / "comparisons.json")
    if comparison:
        selected = [p.name for p in representatives()] + ["zeroshot-baseline"]
        wanted = {frozenset(pair) for pair in combinations(selected, 2)}
        rows = comparison["rows"]
        got = {frozenset((row["first"], row["second"])) for row in rows}
        if got != wanted or len(rows) != len(wanted):
            errors.append(
                "Comparison set is incomplete or duplicated; rerun carvision compare-all."
            )
        sections.append(
            "## Paired comparisons\n\nUnadjusted exploratory 95% intervals; A minus B.\n\n"
            "| A | B | Difference | 95% CI |\n| --- | --- | --- | --- |\n"
            + "\n".join(
                f"| {r['first']} | {r['second']} | {r['point']:.2%} | "
                f"[{r['low']:.2%}, {r['high']:.2%}] |"
                for r in rows
            )
        )
    try:
        best = find_best_run()
        evaluation = validate_evaluation(best)
        hardest = evaluation["hardest_classes"]
        sections.append(
            "## Hardest classes\n\n| Class | Recall | Support |\n| --- | --- | --- |\n"
            + "\n".join(f"| {r['class']} | {r['recall']:.2%} | {r['support']} |" for r in hardest)
        )
        bundle = require(artifacts_dir() / "serving/serving.json")
        if bundle:
            if (
                bundle["run"] != best.name
                or bundle["temperature"] != evaluation["calibration"]["temperature"]
            ):
                errors.append("Serving bundle does not match selected model/calibration.")
            graph = artifacts_dir() / "serving/model.onnx"
            if not graph.exists() or fingerprint(graph) != bundle["onnx"]["sha256"]:
                errors.append("Serving graph missing or changed; rerun carvision export.")
        latency = require(best / "latency.json")
        if latency:
            rows = latency["rows"]
            measured = {r["runtime"] for r in rows if r["batch_size"] == 1}
            if not {"pytorch", "onnxruntime"} <= measured:
                errors.append("Missing batch-1 runtime measurements; rerun carvision bench.")
            sections.append(
                "## CPU latency\n\n| Runtime | Batch | p50 (ms) | p95 (ms) |\n"
                "| --- | --- | --- | --- |\n"
                + "\n".join(
                    f"| {r['runtime']} | {r['batch_size']} | "
                    f"{r['p50_ms']:.2f} | {r['p95_ms']:.2f} |"
                    for r in rows
                )
            )
            sections.append(
                f"Hardware: `{latency['hardware']}`. Versions: `{latency['versions']}`. "
                f"Warmup: {latency['warmup']}; timed iterations: {latency['runs']}."
            )
        figure_record = read(figures_dir() / "provenance.json")
        validate(figure_record)
        required_figures = [
            "training_curves.png",
            "calibration.png",
            "error_structure.png",
            "confusion.png",
        ]
        for path in [best / "evaluation.json"] + [
            figures_dir() / name for name in required_figures
        ]:
            if str(path.resolve()) not in figure_record["files"] or not path.exists():
                errors.append(f"Missing figure source/output {path}; rerun carvision figures.")
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        errors.append(str(exc))
    timing = require(artifacts_dir() / "sweep_timing.json")
    if timing:
        sections.append(
            "## Training cost\n\nSweep training wall time (excluding cache preparation): "
            f"{timing['seconds']:.3f} seconds."
        )
    try:
        from carvision.data.splits import load_split
        from carvision.features.cache import compute_cache_key, entry_dir
        from carvision.models.backbones import get_backbone

        manifest = read(artifacts_dir() / "sweep_manifest.json")
        rows_text = []
        for backbone in manifest["backbones"]:
            seconds, size = 0.0, 0
            for split in ("train", "val", "test"):
                frame = load_split(split)
                key = compute_cache_key(
                    get_backbone(backbone), frame.image_id.tolist(), frame.label_id.to_numpy()
                )
                directory = entry_dir(backbone, split, key)
                metadata = read(directory / "manifest.json")
                if not metadata.get("timing_complete"):
                    errors.append(
                        f"{backbone}/{split}: incomplete historical cache timing; rebuild cache."
                    )
                seconds += metadata["seconds"]
                actual_bytes = sum(
                    (directory / name).stat().st_size
                    for name in ("embeddings.npy", "labels.npy", "image_ids.txt")
                )
                if actual_bytes != metadata.get("artifact_bytes"):
                    errors.append(
                        f"{backbone}/{split}: missing or stale artifact size measurement."
                    )
                size += actual_bytes
            rows_text.append(f"| {backbone} | {seconds:.3f} | {size} |")
        sections.append(
            "## Cache cost\n\nRecorded active build time excludes downtime; "
            "interrupted work after the last saved checkpoint is not counted.\n\n"
            "| Backbone | Seconds | Artifact bytes |\n| --- | --- | --- |\n" + "\n".join(rows_text)
        )
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        errors.append(f"Cache measurements: {exc}")
    source = repo_root() / "data/stanford_cars/download.json"
    if source.exists():
        audit = read(source).get("duplicate_content_audit", {})
        groups = audit.get("groups", [])
        cross = sum(
            any(i.startswith("train_") for i in g) and any(i.startswith("test_") for i in g)
            for g in groups
        )
        sections.append(
            f"## Dataset limitation\n\nThe recorded content audit found {len(groups)} "
            f"duplicate groups, including {cross} spanning the official train/test partitions. "
            "The benchmark partition is preserved; unique IDs do not imply unique image content."
        )
    if strict and errors:
        raise ValueError("Strict publication refused:\n" + "\n".join(errors))
    if errors:
        sections.insert(0, "> INCOMPLETE / UNVERIFIED: " + "; ".join(errors))
    return "\n\n".join(sections), errors
