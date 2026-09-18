"""Generate every committed figure from evaluation output.

No figure here takes a number as an argument. Each one reads the run's
``evaluation.json`` and ``test_predictions.npz``, so a plot cannot disagree with the
metrics it illustrates. The legacy notebook's bar chart hardcoded ``83.26`` while the
model had actually scored ``82.83``, and nothing in the notebook could have caught that.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from carvision.metrics import calibration
from carvision.utils.logging import get_logger
from carvision.utils.paths import ensure_dir, figures_dir

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = get_logger(__name__)

DPI = 150


def _style() -> None:
    """Apply a consistent, readable style to every figure."""
    import matplotlib

    matplotlib.use("Agg")  # No display on CI or in a Space.
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


def _save(figure: Figure, name: str) -> Path:
    path = ensure_dir(figures_dir()) / name
    figure.savefig(path)
    logger.info("Wrote %s", path)
    return path


def reliability_diagram(evaluation: dict[str, Any], predictions: dict[str, np.ndarray]) -> Path:
    """Plot calibration before and after temperature scaling.

    A perfectly calibrated model sits on the diagonal. The gap between the bars and the
    diagonal is what ECE summarises into one number.
    """
    import matplotlib.pyplot as plt

    _style()
    temperature = float(evaluation["calibration"]["temperature"])
    logits, labels = predictions["logits"], predictions["labels"]

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for axis, (title, scaled) in zip(
        axes,
        [
            ("Before temperature scaling", logits),
            (f"After (T = {temperature:.2f})", calibration.apply_temperature(logits, temperature)),
        ],
        strict=True,
    ):
        result = calibration.compute(scaled, labels)
        centres = (result.bin_edges[:-1] + result.bin_edges[1:]) / 2
        present = ~np.isnan(result.bin_accuracy)

        axis.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
        axis.bar(
            centres[present],
            result.bin_accuracy[present],
            width=1 / len(centres) * 0.9,
            alpha=0.75,
            edgecolor="black",
            linewidth=0.5,
            label="observed accuracy",
        )
        axis.set_title(f"{title}\nECE = {result.ece:.4f}")
        axis.set_xlabel("confidence")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)

    axes[0].set_ylabel("accuracy")
    axes[0].legend(loc="upper left", fontsize=8)
    figure.suptitle("Calibration: does a stated confidence mean what it says?")
    return _save(figure, "calibration.png")


def error_structure(evaluation: dict[str, Any]) -> Path:
    """Plot where the error mass sits, by how related the confused classes are.

    This is the figure that carries the project's main qualitative finding.
    """
    import matplotlib.pyplot as plt

    _style()
    by_kind = evaluation["errors"]["by_kind"]
    labels = {
        "same_model_different_year": "Same model,\ndifferent year",
        "same_model_different_body": "Same model,\ndifferent body",
        "same_make": "Same make,\ndifferent model",
        "cross_make": "Different make",
    }
    kinds = list(labels)
    shares = [by_kind[kind]["share_of_errors"] for kind in kinds]
    counts = [by_kind[kind]["count"] for kind in kinds]

    figure, axis = plt.subplots(figsize=(7.5, 4.2))
    bars = axis.bar(
        [labels[kind] for kind in kinds],
        [100 * share for share in shares],
        color=["#2a9d8f", "#8ab17d", "#e9c46a", "#e76f51"],
        edgecolor="black",
        linewidth=0.5,
    )
    for bar, count in zip(bars, counts, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"n={count}",
            ha="center",
            fontsize=9,
        )

    axis.set_ylabel("share of all errors (%)")
    axis.set_title("What kind of wrong is it?")
    axis.set_ylim(0, max(100 * s for s in shares) * 1.2)
    return _save(figure, "error_structure.png")


def confusion_overview(predictions: dict[str, np.ndarray]) -> Path:
    """Plot the confusion matrix as a heatmap.

    At 196x196 individual cells are unreadable, and that is the point: what the figure
    shows is whether error mass concentrates in blocks (structured, within-make
    confusion) or scatters uniformly (the model is guessing).
    """
    import matplotlib.pyplot as plt

    _style()
    matrix = predictions["confusion"].astype(float)
    # Row-normalise so classes with more test images do not dominate visually.
    normalised = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(normalised, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    axis.set_xlabel("predicted class")
    axis.set_ylabel("true class")
    axis.set_title("Confusion matrix (row-normalised, 196 classes)")
    axis.grid(False)
    figure.colorbar(image, ax=axis, label="share of true class")
    return _save(figure, "confusion.png")


def training_curves(run_dir: Path) -> Path:
    """Plot the training and validation curves for one run.

    One point per epoch, read from ``metrics.json``. The legacy notebook's equivalent
    plotted a running total appended once per batch, and crashed when it did not.
    """
    import matplotlib.pyplot as plt

    _style()
    history = json.loads((run_dir / "metrics.json").read_text())["history"]
    epochs = [record["epoch"] for record in history]

    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(10, 4))

    loss_axis.plot(epochs, [r["train_loss"] for r in history], label="train")
    loss_axis.plot(epochs, [r["val_loss"] for r in history], label="validation")
    loss_axis.set_xlabel("epoch")
    loss_axis.set_ylabel("cross-entropy loss")
    loss_axis.set_title("Loss")
    loss_axis.legend()

    accuracy = [100 * r["val_top1"] for r in history]
    best = int(np.argmax(accuracy))
    accuracy_axis.plot(epochs, accuracy, color="#2a9d8f")
    accuracy_axis.axvline(
        epochs[best], linestyle="--", color="grey", label=f"best epoch ({epochs[best]})"
    )
    accuracy_axis.set_xlabel("epoch")
    accuracy_axis.set_ylabel("validation top-1 (%)")
    accuracy_axis.set_title("Validation accuracy")
    accuracy_axis.legend()

    figure.suptitle(run_dir.name)
    return _save(figure, "training_curves.png")


def generate_all(run_dir: Path) -> list[Path]:
    """Regenerate every figure for one evaluated run.

    Args:
        run_dir: A run that has been trained and evaluated.

    Returns:
        The paths written.

    Raises:
        FileNotFoundError: If the run has not been evaluated.
    """
    evaluation_path = run_dir / "evaluation.json"
    predictions_path = run_dir / "test_predictions.npz"
    if not evaluation_path.exists() or not predictions_path.exists():
        raise FileNotFoundError(
            f"{run_dir.name} has no evaluation output. Run `carvision eval` first."
        )

    evaluation = json.loads(evaluation_path.read_text())
    predictions = dict(np.load(predictions_path, allow_pickle=False))

    return [
        training_curves(run_dir),
        reliability_diagram(evaluation, predictions),
        error_structure(evaluation),
        confusion_overview(predictions),
    ]
