"""Command-line entry point.

Every stage of the pipeline is reachable from one command, so the README's quickstart is
a list of commands rather than "open the notebook and run the cells in order" -- which
was the legacy project's only interface, and which did not work because the cells had
been executed out of order.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import TYPE_CHECKING

from carvision import __version__
from carvision.utils.logging import configure_logging, get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

logger = get_logger(__name__)

DEFAULT_SWEEP_BACKBONES = ("resnet50", "clip_vitb32", "dinov2_vits14")
DEFAULT_SWEEP_HEADS = ("linear", "mlp")
DEFAULT_SWEEP_SEEDS = (0, 1, 2, 3, 4)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for every subcommand."""
    parser = argparse.ArgumentParser(
        prog="carvision",
        description="Fine-grained car classification on cached frozen-backbone features.",
    )
    parser.add_argument("--version", action="version", version=f"carvision {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug-level logging.")

    sub = parser.add_subparsers(dest="command", required=True)

    # ---- data
    data = sub.add_parser("data", help="Download the dataset and build splits.")
    data_sub = data.add_subparsers(dest="data_command", required=True)

    download = data_sub.add_parser("download", help="Fetch Stanford Cars from the Hub.")
    download.add_argument(
        "--repo-id",
        default=None,
        help="Hub dataset repo id. Defaults to hf_repo_id in configs/data/stanford_cars.yaml.",
    )
    download.add_argument(
        "--data-config",
        default="stanford_cars",
        help="Name of the config under configs/data/ to read.",
    )
    download.add_argument("--revision", default=None, help="Hub revision to pin.")
    download.add_argument("--image-column", default=None)
    download.add_argument("--label-column", default=None)
    download.add_argument("--train-split", default=None)
    download.add_argument("--test-split", default=None)
    download.add_argument("--force", action="store_true", help="Re-download.")

    split = data_sub.add_parser("split", help="Write the deterministic split CSVs.")
    split.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help="Overrides val_fraction from the data config.",
    )
    verify = data_sub.add_parser("verify", help="Validate the local dataset and committed splits.")
    verify.add_argument("--data-config", default="stanford_cars")
    split.add_argument(
        "--seed", type=int, default=None, help="Overrides split_seed from the data config."
    )
    split.add_argument("--data-config", default="stanford_cars")
    split.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate committed splits. This invalidates every published number.",
    )

    # ---- cache
    cache_parser = sub.add_parser("cache", help="Manage the embedding cache.")
    cache_sub = cache_parser.add_subparsers(dest="cache_command", required=True)

    cache_build = cache_sub.add_parser("build", help="Embed splits with a frozen backbone.")
    cache_build.add_argument("--backbone", required=True)
    cache_build.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    cache_build.add_argument("--batch-size", type=int, default=32)
    cache_build.add_argument("--num-workers", type=int, default=2)
    cache_build.add_argument("--force", action="store_true")

    cache_clear = cache_sub.add_parser("clear", help="Delete cache entries.")
    cache_clear.add_argument("--backbone", default=None)

    # ---- train
    train_parser = sub.add_parser("train", help="Train one head on cached embeddings.")
    train_parser.add_argument("--backbone", default="dinov2_vits14")
    train_parser.add_argument("--head", default="linear", choices=["linear", "mlp"])
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--max-epochs", type=int, default=100)
    train_parser.add_argument("--lr", type=float, default=1e-3)

    sweep_parser = sub.add_parser("sweep", help="Train every backbone x head x seed.")
    sweep_parser.add_argument("--backbones", nargs="+", default=list(DEFAULT_SWEEP_BACKBONES))
    sweep_parser.add_argument("--heads", nargs="+", default=list(DEFAULT_SWEEP_HEADS))
    sweep_parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SWEEP_SEEDS))

    zeroshot = sub.add_parser("zeroshot", help="CLIP zero-shot baseline (no training).")
    zeroshot.add_argument("--backbone", default="clip_vitb32")
    zeroshot.add_argument("--split", default="test")

    # ---- eval
    eval_parser = sub.add_parser("eval", help="Score a run on the test set.")
    eval_choice = eval_parser.add_mutually_exclusive_group()
    eval_choice.add_argument(
        "--run", default="best", help="Run directory name, or 'best' for the top val run."
    )
    eval_choice.add_argument("--all", action="store_true")
    eval_parser.add_argument("--resamples", type=int, default=1000)

    compare = sub.add_parser("compare", help="Paired bootstrap between two evaluated runs.")
    compare.add_argument("first")
    compare.add_argument("second")
    sub.add_parser("compare-all", help="Generate all pairwise evaluated-run comparisons.")

    # ---- export / bench
    export_parser = sub.add_parser("export", help="Export a run to ONNX and verify parity.")
    export_parser.add_argument("--run", default="best")
    export_parser.add_argument("--out", default="artifacts/serving")

    bench = sub.add_parser("bench", help="Benchmark CPU latency, PyTorch vs ONNX Runtime.")
    bench.add_argument("--run", default="best")
    bench.add_argument("--bundle", default="artifacts/serving")
    bench.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8])
    bench.add_argument("--runs", type=int, default=100)

    figures = sub.add_parser("figures", help="Regenerate every figure under docs/figures/.")
    figures.add_argument("--run", default="best")

    report = sub.add_parser("report", help="Regenerate docs/RESULTS.md from the evaluated runs.")
    report.add_argument("--strict", action="store_true")

    return parser


def _cmd_data(args: argparse.Namespace) -> int:
    from carvision.config import load_data_config
    from carvision.data import download as download_module
    from carvision.data import splits as splits_module

    if args.data_command == "download":
        overrides = {
            key: value
            for key, value in (
                ("image_column", args.image_column),
                ("label_column", args.label_column),
                ("train_split", args.train_split),
                ("test_split", args.test_split),
            )
            if value is not None
        }
        settings = load_data_config(args.data_config)
        root = download_module.download(
            args.repo_id,
            revision=args.revision,
            force=args.force,
            config=settings,
            **overrides,
        )
        logger.info("Dataset ready at %s", root)
        return 0

    if args.data_command == "verify":
        from carvision.data.download import verify_local_dataset
        from carvision.data.splits import validate_splits

        verify_local_dataset()
        settings = load_data_config(args.data_config)
        validate_splits(val_fraction=settings.val_fraction, seed=settings.split_seed)
        logger.info("Dataset and committed splits verified")
        return 0

    from carvision.config import load_data_config

    settings = load_data_config(args.data_config)
    sizes = splits_module.build_splits(
        val_fraction=args.val_fraction if args.val_fraction is not None else settings.val_fraction,
        seed=args.seed if args.seed is not None else settings.split_seed,
        overwrite=args.overwrite,
    )
    logger.info(
        "Splits written: %d train / %d val / %d test (%d total)",
        sizes.train,
        sizes.val,
        sizes.test,
        sizes.total,
    )
    return 0


def _cmd_cache(args: argparse.Namespace) -> int:
    from carvision.features import cache as cache_module

    if args.cache_command == "clear":
        removed = cache_module.clear(args.backbone)
        logger.info("Removed %d entries", removed)
        return 0

    for split in args.splits:
        entry = cache_module.build(
            args.backbone,
            split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            force=args.force,
        )
        logger.info("%s/%s -> (%d, %d)", entry.backbone, entry.split, entry.num_rows, entry.dim)
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from carvision.train import TrainConfig, train

    result = train(
        TrainConfig(
            backbone=args.backbone,
            head=args.head,
            seed=args.seed,
            max_epochs=args.max_epochs,
            lr=args.lr,
        )
    )
    print(json.dumps({"val_top1": result.best_val_top1, "run_dir": str(result.run_dir)}, indent=2))
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    from carvision.train import sweep

    results = sweep(args.backbones, args.heads, args.seeds)
    for result in results:
        print(
            f"{result.config.backbone:16s} {result.config.head:7s} "
            f"seed={result.config.seed} val_top1={result.best_val_top1:.4f}"
        )
    return 0


def _cmd_zeroshot(args: argparse.Namespace) -> int:
    import numpy as np

    from carvision.data.download import load_class_names
    from carvision.features import cache as cache_module
    from carvision.models.zeroshot import PROMPT_TEMPLATES, build_text_classifier, predict
    from carvision.utils.paths import runs_dir

    if args.backbone != "clip_vitb32":
        raise ValueError("Zero-shot requires clip_vitb32 with matching OpenAI weights.")
    from carvision.utils.artifacts import dependencies, evaluation_inputs

    baseline_dir = runs_dir() / (
        "zeroshot-baseline" if args.split == "test" else f"zeroshot-{args.split}"
    )
    # Check all required inputs before replacing a prior baseline.
    dependencies(
        evaluation_inputs(baseline_dir, args.backbone, [args.split], baseline=True),
        "carvision zeroshot",
    )
    embeddings, labels, image_ids = cache_module.load(args.backbone, args.split)
    classifier = build_text_classifier(load_class_names())
    scores = predict(embeddings, classifier)

    top1 = float((scores.argmax(axis=1) == labels).mean())
    top5 = float(
        np.mean(
            [label in row for label, row in zip(labels, np.argsort(-scores)[:, :5], strict=True)]
        )
    )
    baseline_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        baseline_dir / "test_predictions.npz",
        logits=scores,
        labels=labels,
        image_ids=np.asarray(image_ids),
    )
    (baseline_dir / "config.json").write_text(
        json.dumps({"backbone": args.backbone, "head": "zero-shot", "seed": 0}) + "\n"
    )
    (baseline_dir / "metrics.json").write_text(
        json.dumps({"best_val_top1": 0.0, "baseline": True}) + "\n"
    )
    from carvision.metrics import bootstrap, classification
    from carvision.models.backbones import get_backbone
    from carvision.utils.artifacts import fingerprint

    provenance = dependencies(
        evaluation_inputs(baseline_dir, args.backbone, [args.split], baseline=True),
        "carvision zeroshot",
    )
    provenance["predictions_sha256"] = fingerprint(baseline_dir / "test_predictions.npz")
    (baseline_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "run": baseline_dir.name,
                "baseline": True,
                "metrics": classification.compute(scores, labels).as_dict(),
                "top1_interval": bootstrap.accuracy_interval(
                    scores.argmax(axis=1) == labels
                ).as_dict(),
                "class_names": load_class_names(),
                "split": args.split,
                "weights": get_backbone(args.backbone).weights_tag,
                "prompt_templates": list(PROMPT_TEMPLATES),
                "provenance": provenance,
            },
            indent=2,
        )
        + "\n"
    )
    from carvision.utils.artifacts import read, seal_evaluation

    evaluation_path = baseline_dir / "evaluation.json"
    evaluation = read(evaluation_path)
    seal_evaluation(evaluation, args.backbone)
    evaluation_path.write_text(json.dumps(evaluation, indent=2) + "\n")
    print(
        json.dumps(
            {"split": args.split, "top1": top1, "top5": top5, "path": str(baseline_dir)}, indent=2
        )
    )
    return 0


def _resolve_run(name: str) -> Path:
    """Resolve a --run argument to a directory, accepting the literal 'best'."""
    from carvision.eval import find_best_run
    from carvision.utils.paths import runs_dir

    if name == "best":
        return find_best_run()
    path = runs_dir() / name
    if not path.is_dir():
        raise FileNotFoundError(f"No run directory at {path}")
    return path


def _cmd_eval(args: argparse.Namespace) -> int:
    from carvision.eval import evaluate_all, evaluate_run

    if args.all:
        print(json.dumps([r.as_dict() for r in evaluate_all(resamples=args.resamples)], indent=2))
        return 0
    result = evaluate_run(_resolve_run(args.run), resamples=args.resamples)
    print(json.dumps(result.as_dict(), indent=2))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from carvision.eval import compare_runs

    interval = compare_runs(_resolve_run(args.first), _resolve_run(args.second))
    print(
        json.dumps({**interval.as_dict(), "difference_resolved": interval.excludes_zero}, indent=2)
    )
    return 0


def _cmd_compare_all(args: argparse.Namespace) -> int:
    del args
    from carvision.eval import compare_all

    print(json.dumps(compare_all(), indent=2))
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from pathlib import Path

    from carvision.export import prepare_serving_bundle

    bundle = prepare_serving_bundle(_resolve_run(args.run), Path(args.out))
    print(f"Serving bundle: {bundle}")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from pathlib import Path

    import torch

    from carvision.eval import load_run
    from carvision.export import ServingModel, benchmark_onnx, benchmark_pytorch
    from carvision.models.backbones import get_backbone

    run_dir = _resolve_run(args.run)
    head, payload = load_run(run_dir)
    spec = get_backbone(str(payload["backbone"]))
    import platform

    import onnxruntime

    from carvision.utils.artifacts import (
        dependencies,
        fingerprint,
        read,
        validate,
        validate_evaluation,
    )

    evaluation = validate_evaluation(run_dir)
    bundle = read(Path(args.bundle) / "serving.json")
    validate(bundle.get("provenance", {}))
    if (
        bundle["run"] != run_dir.name
        or bundle["temperature"] != evaluation["calibration"]["temperature"]
    ):
        raise ValueError(
            "Bundle does not match selected run and calibration; rerun carvision export."
        )
    if bundle["onnx"]["sha256"] != fingerprint(Path(args.bundle) / "model.onnx"):
        raise ValueError("ONNX graph changed; rerun carvision export.")
    model = ServingModel(spec.build(), head, temperature=bundle["temperature"]).eval()
    onnx_path = Path(args.bundle) / "model.onnx"

    rows = []
    for batch_size in args.batch_sizes:
        example = torch.randn(batch_size, 3, spec.preprocess.crop, spec.preprocess.crop)
        rows.append(benchmark_pytorch(model, example, runs=args.runs, warmup=10).as_dict())
        if onnx_path.exists():
            rows.append(benchmark_onnx(onnx_path, example, runs=args.runs, warmup=10).as_dict())

    (run_dir / "latency.json").write_text(
        json.dumps(
            {
                "rows": rows,
                "warmup": 10,
                "runs": args.runs,
                "hardware": {
                    "platform": platform.platform(),
                    "processor": platform.processor(),
                    "threads": torch.get_num_threads(),
                },
                "versions": {"torch": torch.__version__, "onnxruntime": onnxruntime.__version__},
                "provenance": dependencies(
                    [run_dir / "evaluation.json", Path(args.bundle) / "serving.json", onnx_path],
                    "carvision bench",
                ),
            },
            indent=2,
        )
        + "\n"
    )
    for row in rows:
        print(
            f"{row['runtime']:12s} batch={row['batch_size']:<3d} "
            f"p50={row['p50_ms']:7.1f}ms  p95={row['p95_ms']:7.1f}ms"
        )
    return 0


def _cmd_figures(args: argparse.Namespace) -> int:
    from carvision.figures import generate_all

    paths = generate_all(_resolve_run(args.run))
    for path in paths:
        print(path)
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from carvision.report import write

    print(write(strict=args.strict))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    import logging

    args = build_parser().parse_args(argv)
    configure_logging(logging.DEBUG if args.verbose else logging.INFO)

    handlers = {
        "data": _cmd_data,
        "cache": _cmd_cache,
        "train": _cmd_train,
        "sweep": _cmd_sweep,
        "zeroshot": _cmd_zeroshot,
        "eval": _cmd_eval,
        "compare": _cmd_compare,
        "compare-all": _cmd_compare_all,
        "export": _cmd_export,
        "bench": _cmd_bench,
        "figures": _cmd_figures,
        "report": _cmd_report,
    }
    try:
        return handlers[args.command](args)
    except (RuntimeError, ValueError, KeyError, FileNotFoundError) as error:
        # Surface the actionable message without a traceback; -v restores the detail.
        logger.error("%s", error)
        if args.verbose:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
