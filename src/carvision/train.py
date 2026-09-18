"""Train a classifier head on cached embeddings.

Training consumes a fixed ``(N, D)`` matrix, so an epoch is a handful of dense matrix
multiplies and the whole run finishes in seconds. Every run writes a checkpoint, its
fully resolved configuration, the git SHA, and a metrics file -- the legacy project saved
none of these, so neither of its models survived the kernel that produced them.

Two details are deliberate. Epoch losses are computed by explicit reduction over a
running sum divided by the sample count *at the end of the epoch*; the legacy notebook
appended inside the batch loop without resetting its accumulator, so its loss curves were
running totals rather than epoch means. And early stopping watches validation accuracy,
never test -- the legacy project had no validation split and selected against test.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from carvision.data.dataset import EmbeddingDataset
from carvision.features import cache
from carvision.models.backbones import get_backbone
from carvision.models.heads import build_head
from carvision.utils.logging import get_logger
from carvision.utils.paths import ensure_dir, runs_dir
from carvision.utils.seed import set_seed

logger = get_logger(__name__)


@dataclass
class TrainConfig:
    """Hyperparameters for one head-training run."""

    backbone: str = "dinov2_vits14"
    head: str = "linear"
    seed: int = 0
    max_epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    warmup_epochs: int = 5
    early_stopping_patience: int = 10
    head_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpochRecord:
    """One epoch's metrics. Exactly one of these per epoch, by construction."""

    epoch: int
    train_loss: float
    val_loss: float
    val_top1: float
    lr: float


@dataclass
class TrainResult:
    """Everything a completed run produced."""

    config: TrainConfig
    history: list[EpochRecord]
    best_epoch: int
    best_val_top1: float
    run_dir: Path
    seconds: float


def cosine_lr(step: int, total: int, *, warmup: int, base_lr: float) -> float:
    """Return the learning rate for a step under linear warmup then cosine decay.

    Args:
        step: Zero-based epoch index.
        total: Total epochs.
        warmup: Epochs of linear warmup.
        base_lr: Peak learning rate, reached at the end of warmup.

    Returns:
        The learning rate for this step.
    """
    if warmup > 0 and step < warmup:
        # step+1 so the first epoch gets a nonzero rate.
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def evaluate(
    head: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
) -> tuple[float, float]:
    """Compute loss and top-1 accuracy over a whole split in one pass.

    Args:
        head: The head being trained.
        features: ``(N, D)`` embeddings.
        labels: ``(N,)`` true classes.
        criterion: Loss function.

    Returns:
        ``(loss, top1_accuracy)`` with accuracy as a fraction.
    """
    head.eval()
    logits = head(features)
    loss = float(criterion(logits, labels).item())
    top1 = float((logits.argmax(dim=1) == labels).float().mean().item())
    return loss, top1


def train(config: TrainConfig) -> TrainResult:
    """Train one head on cached embeddings.

    Args:
        config: The run's hyperparameters.

    Returns:
        The run result, including the path its artifacts were written to.
    """
    set_seed(config.seed)
    started = time.monotonic()

    spec = get_backbone(config.backbone)
    train_x, train_y, _ = cache.load(config.backbone, "train")
    val_x, val_y, _ = cache.load(config.backbone, "val")

    train_set = EmbeddingDataset(train_x, train_y)
    val_set = EmbeddingDataset(val_x, val_y)
    num_classes = int(max(train_y.max(), val_y.max())) + 1

    head = build_head(config.head, spec.embedding_dim, num_classes, **config.head_kwargs)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    train_features, train_labels = train_set.tensors()
    val_features, val_labels = val_set.tensors()

    history: list[EpochRecord] = []
    best_val_top1 = -1.0
    best_epoch = -1
    best_state: dict[str, torch.Tensor] = {}
    epochs_without_improvement = 0

    generator = torch.Generator().manual_seed(config.seed)

    for epoch in range(config.max_epochs):
        lr = cosine_lr(epoch, config.max_epochs, warmup=config.warmup_epochs, base_lr=config.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        head.train()
        # Reset every epoch. The legacy notebook did not, which is why its loss curve
        # only ever went up.
        epoch_loss_sum = 0.0
        epoch_samples = 0

        permutation = torch.randperm(len(train_set), generator=generator)
        for start in range(0, len(permutation), config.batch_size):
            batch_index = permutation[start : start + config.batch_size]
            features = train_features[batch_index]
            labels = train_labels[batch_index]

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(features), labels)
            loss.backward()
            optimizer.step()

            # Weight by batch size so a short final batch does not skew the mean.
            epoch_loss_sum += float(loss.item()) * len(batch_index)
            epoch_samples += len(batch_index)

        # One record per epoch, appended here and nowhere else.
        train_loss = epoch_loss_sum / epoch_samples
        val_loss, val_top1 = evaluate(head, val_features, val_labels, criterion)
        history.append(EpochRecord(epoch, train_loss, val_loss, val_top1, lr))

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                logger.info(
                    "Early stop at epoch %d; best val top-1 %.4f at epoch %d",
                    epoch,
                    best_val_top1,
                    best_epoch,
                )
                break

    head.load_state_dict(best_state)
    elapsed = time.monotonic() - started

    run_dir = _write_run(config, head, history, best_epoch, best_val_top1, elapsed, num_classes)
    logger.info(
        "%s/%s seed=%d: val top-1 %.2f%% in %.1fs -> %s",
        config.backbone,
        config.head,
        config.seed,
        100 * best_val_top1,
        elapsed,
        run_dir.name,
    )

    return TrainResult(
        config=config,
        history=history,
        best_epoch=best_epoch,
        best_val_top1=best_val_top1,
        run_dir=run_dir,
        seconds=elapsed,
    )


def _write_run(
    config: TrainConfig,
    head: nn.Module,
    history: list[EpochRecord],
    best_epoch: int,
    best_val_top1: float,
    seconds: float,
    num_classes: int,
) -> Path:
    """Persist the checkpoint, resolved config, and metrics for one run."""
    from carvision.utils.provenance import git_sha

    name = f"{config.backbone}-{config.head}-seed{config.seed}"
    run_dir = ensure_dir(runs_dir() / name)

    torch.save(
        {
            "state_dict": head.state_dict(),
            "head_kind": config.head,
            "backbone": config.backbone,
            "num_classes": num_classes,
            "embedding_dim": get_backbone(config.backbone).embedding_dim,
        },
        run_dir / "checkpoint.pt",
    )

    (run_dir / "config.json").write_text(json.dumps(config.__dict__, indent=2) + "\n")
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_val_top1": best_val_top1,
                "epochs_run": len(history),
                "seconds": round(seconds, 2),
                "git_sha": git_sha(),
                "history": [record.__dict__ for record in history],
            },
            indent=2,
        )
        + "\n"
    )
    return run_dir


def sweep(
    backbones: list[str],
    heads: list[str],
    seeds: list[int],
    *,
    base: TrainConfig | None = None,
) -> list[TrainResult]:
    """Train every combination of backbone, head and seed.

    Cheap precisely because the embeddings are cached: the backbone passes happen once,
    up front, and each of the resulting runs is seconds of linear algebra. Five seeds per
    configuration is what makes the reported spread meaningful rather than decorative.

    Args:
        backbones: Backbone names.
        heads: Head kinds.
        seeds: Seeds to repeat each configuration with.
        base: Template config supplying the shared hyperparameters.

    Returns:
        One result per combination, in iteration order.
    """
    template = base or TrainConfig()
    results: list[TrainResult] = []

    for backbone in backbones:
        # Warm the cache once per backbone rather than inside the seed loop.
        for split in ("train", "val", "test"):
            cache.build(backbone, split)

        for head in heads:
            for seed in seeds:
                config = TrainConfig(
                    **{**template.__dict__, "backbone": backbone, "head": head, "seed": seed}
                )
                results.append(train(config))

    accuracies = np.array([r.best_val_top1 for r in results])
    logger.info(
        "Sweep finished: %d runs, val top-1 %.2f%% +/- %.2f%%",
        len(results),
        100 * accuracies.mean(),
        100 * accuracies.std(),
    )
    return results
