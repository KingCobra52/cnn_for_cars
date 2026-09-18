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
    download.add_argument("--repo-id", default=None, help="Hub dataset repo id.")
    download.add_argument("--revision", default=None, help="Hub revision to pin.")
    download.add_argument("--image-column", default=None)
    download.add_argument("--label-column", default=None)
    download.add_argument("--train-split", default=None)
    download.add_argument("--test-split", default=None)
    download.add_argument("--force", action="store_true", help="Re-download.")

    split = data_sub.add_parser("split", help="Write the deterministic split CSVs.")
    split.add_argument("--val-fraction", type=float, default=0.15)
    split.add_argument("--seed", type=int, default=20260918)
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

    return parser


def _cmd_data(args: argparse.Namespace) -> int:
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
        root = download_module.download(
            args.repo_id, revision=args.revision, force=args.force, **overrides
        )
        logger.info("Dataset ready at %s", root)
        return 0

    sizes = splits_module.build_splits(
        val_fraction=args.val_fraction, seed=args.seed, overwrite=args.overwrite
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
    from carvision.models.zeroshot import build_text_classifier, predict

    embeddings, labels, _ = cache_module.load(args.backbone, args.split)
    classifier = build_text_classifier(load_class_names())
    scores = predict(embeddings, classifier)

    top1 = float((scores.argmax(axis=1) == labels).mean())
    top5 = float(
        np.mean([label in row for label, row in zip(labels, np.argsort(-scores)[:, :5], strict=True)])
    )
    print(json.dumps({"split": args.split, "top1": top1, "top5": top5}, indent=2))
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
    }
    try:
        return handlers[args.command](args)
    except (RuntimeError, ValueError, KeyError) as error:
        # Surface the actionable message without a traceback; -v restores the detail.
        logger.error("%s", error)
        if args.verbose:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
