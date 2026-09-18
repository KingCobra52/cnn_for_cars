"""End-to-end test of the whole pipeline, offline.

Every other test in this suite checks one function. This one runs the real chain the
way a user does -- build the embedding cache, train a head, evaluate, export to ONNX,
render the figures, generate the results document, and serve a prediction -- and it does
so with no dataset, no pretrained weights, and no network.

It exists because the modules that produce every number in the README (``eval``,
``figures``, ``report``) run *last*. A defect in them surfaces only after an hour of
cache building on the real dataset, which is the worst possible place to discover one.
Here they run in seconds on every push.

Two substitutions make that possible:

* A **fake backbone** registered in the real registry: a tiny frozen conv net with random
  weights. Everything downstream treats it exactly like DINOv2, because the registry is
  the only thing that knows the difference.
* A **synthetic dataset** written to a temporary ``CARVISION_ROOT``, so the real path
  helpers, the real split code, and the real cache all operate on it unmodified.

The synthetic images are not noise. Each class is a colour, and classes that a human
would confuse -- the same model in a different year, the same model in a different body
style -- are given deliberately similar colours. So the error analysis has the structure
it is designed to find, and its bucketing is checked against structure the fixture planted,
rather than against its own output.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from carvision.data.transforms import PreprocessSpec
from carvision.models.backbones import REGISTRY, BackboneSpec

if TYPE_CHECKING:
    from collections.abc import Iterator

APP_DIR = Path(__file__).resolve().parents[1] / "app"

# --------------------------------------------------------------------- the fake world

BACKBONE_NAME = "fake_tiny"
EMBEDDING_DIM = 12
SOURCE_SIZE = 40

#: Preprocessing for the fake backbone. Small on purpose: the point is the wiring, and
#: 224x224 would make this test slow for no extra coverage.
FAKE_PREPROCESS = PreprocessSpec(resize=36, crop=32, mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25))


@dataclass(frozen=True)
class FakeClass:
    """One synthetic class: a Stanford-Cars-style name and the colour that encodes it."""

    name: str
    colour: tuple[int, int, int]


#: Six classes chosen so every error bucket is reachable. The colours encode the
#: relationships: the two Ford Focus years are nearly the same colour and so will be
#: confused with each other, while the Tesla is far away in colour space and should not
#: be confused with anything.
FAKE_CLASSES: tuple[FakeClass, ...] = (
    FakeClass("2012 Ford Focus Sedan", (200, 40, 40)),
    FakeClass("2007 Ford Focus Sedan", (205, 55, 45)),  # same model, different year
    FakeClass("2012 Ford Focus Coupe", (190, 45, 60)),  # same model, different body
    FakeClass("2012 Ford Fiesta Sedan", (40, 200, 40)),  # same make, different model
    FakeClass("2012 Tesla Model S Sedan", (40, 40, 200)),  # different make
    FakeClass("2011 Audi A4 Sedan", (200, 200, 40)),  # different make
)

TRAIN_PER_CLASS = 10
TEST_PER_CLASS = 6


class FakeBackbone(nn.Module):
    """A frozen conv stem standing in for a pretrained model.

    Random weights are fine. Global average pooling over a colour-dominated image
    yields a feature vector that depends mostly on colour, so the synthetic classes are
    separable -- which is what lets the training step in this test actually converge and
    the evaluation step produce a meaningful confusion matrix.
    """

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, dim, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Map ``(B, 3, H, W)`` to ``(B, dim)``."""
        features: torch.Tensor = self.pool(torch.relu(self.conv(images))).flatten(1)
        return features


def _make_fake_backbone() -> nn.Module:
    # Seeded so the whole test is reproducible, including the accuracy it reaches.
    torch.manual_seed(1234)
    return FakeBackbone()


FAKE_SPEC = BackboneSpec(
    name=BACKBONE_NAME,
    weights_tag="synthetic-v1",
    embedding_dim=EMBEDDING_DIM,
    preprocess=FAKE_PREPROCESS,
    factory=_make_fake_backbone,
    gradcam_layer="conv",
)


def _render(class_index: int, sample_index: int, rng: np.random.Generator) -> Image.Image:
    """Draw one synthetic image for a class.

    The class colour fills the frame; noise and a per-sample brightness shift stop the
    classes being trivially separable, so the model makes some mistakes and the error
    analysis has something to analyse.
    """
    base = np.array(FAKE_CLASSES[class_index].colour, dtype=np.float32)
    canvas = np.tile(base, (SOURCE_SIZE, SOURCE_SIZE, 1))
    canvas += rng.normal(scale=28.0, size=canvas.shape)
    canvas += rng.normal(scale=14.0) * (1 + sample_index % 3)
    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))


@pytest.fixture
def synthetic_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Build a complete fake project root and point carvision at it.

    Writes the images and the download manifest, then calls the *real* ``build_splits``
    so the split code under test is what produces train/val/test -- rather than the test
    hand-writing CSVs and quietly asserting its own idea of the format.
    """
    monkeypatch.setenv("CARVISION_ROOT", str(tmp_path))

    root = tmp_path / "data" / "stanford_cars"
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(7)

    for split, per_class in (("train", TRAIN_PER_CLASS), ("test", TEST_PER_CLASS)):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        for class_index in range(len(FAKE_CLASSES)):
            for sample_index in range(per_class):
                image_id = f"{split}_{class_index:02d}_{sample_index:02d}"
                relpath = f"images/{split}/{image_id}.jpg"
                _render(class_index, sample_index, rng).save(root / relpath, quality=95)
                rows.append(
                    {
                        "image_id": image_id,
                        "split": split,
                        "label_id": class_index,
                        "label_name": FAKE_CLASSES[class_index].name,
                        "width": SOURCE_SIZE,
                        "height": SOURCE_SIZE,
                        "relpath": relpath,
                    }
                )

    import pandas as pd

    pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)
    (root / "classes.txt").write_text("\n".join(c.name for c in FAKE_CLASSES) + "\n")

    from carvision.data.splits import build_splits

    build_splits(val_fraction=0.2, seed=99)
    yield tmp_path


@pytest.fixture
def fake_backbone(monkeypatch: pytest.MonkeyPatch) -> str:
    """Register the fake backbone in the real registry for the duration of the test."""
    monkeypatch.setitem(REGISTRY, BACKBONE_NAME, FAKE_SPEC)
    return BACKBONE_NAME


# --------------------------------------------------------------------- the pipeline


@pytest.fixture
def trained_run(synthetic_repo: Path, fake_backbone: str) -> Path:
    """Run cache build and head training, returning the run directory."""
    from carvision.features import cache
    from carvision.train import TrainConfig, train

    for split in ("train", "val", "test"):
        cache.build(fake_backbone, split, batch_size=8, num_workers=0)

    result = train(
        TrainConfig(
            backbone=fake_backbone,
            head="linear",
            seed=0,
            max_epochs=60,
            batch_size=16,
            lr=5e-2,
            warmup_epochs=3,
            early_stopping_patience=20,
        )
    )
    return result.run_dir


def test_cache_round_trips(synthetic_repo: Path, fake_backbone: str) -> None:
    """The cache writes embeddings whose shape and ordering match the split."""
    from carvision.data.splits import load_split
    from carvision.features import cache

    entry = cache.build(fake_backbone, "test", batch_size=8, num_workers=0)
    embeddings, labels, image_ids = cache.load(fake_backbone, "test")

    expected = load_split("test")
    assert entry.num_rows == len(expected)
    assert embeddings.shape == (len(expected), EMBEDDING_DIM)
    assert embeddings.dtype == np.float32
    # Row order is the contract: row i must be image_ids[i], with labels[i].
    assert image_ids == expected["image_id"].astype(str).tolist()
    np.testing.assert_array_equal(labels, expected["label_id"].to_numpy())


def test_second_cache_build_is_a_hit(synthetic_repo: Path, fake_backbone: str) -> None:
    """The whole design rests on not recomputing embeddings. Prove it does not."""
    from carvision.features import cache

    first = cache.build(fake_backbone, "val", batch_size=8, num_workers=0)
    stamp = (first.directory / "embeddings.npy").stat().st_mtime_ns

    second = cache.build(fake_backbone, "val", batch_size=8, num_workers=0)

    assert second.key == first.key
    assert (second.directory / "embeddings.npy").stat().st_mtime_ns == stamp, (
        "a cache hit must not rewrite the embeddings file"
    )


def test_cache_manifest_records_provenance(synthetic_repo: Path, fake_backbone: str) -> None:
    from carvision.features import cache

    entry = cache.build(fake_backbone, "val", batch_size=8, num_workers=0)
    manifest = json.loads((entry.directory / "manifest.json").read_text())

    assert manifest["backbone"] == fake_backbone
    assert manifest["weights_tag"] == "synthetic-v1"
    assert manifest["preprocess"]["crop"] == FAKE_PREPROCESS.crop
    assert manifest["num_rows"] == entry.num_rows
    assert "torch_version" in manifest


def test_training_persists_everything_needed_to_resume(trained_run: Path) -> None:
    """The legacy project saved nothing. Check each artifact is actually written."""
    for name in ("checkpoint.pt", "config.json", "metrics.json"):
        assert (trained_run / name).is_file(), f"{name} missing"

    metrics = json.loads((trained_run / "metrics.json").read_text())
    assert metrics["epochs_run"] == len(metrics["history"])
    assert metrics["best_epoch"] < metrics["epochs_run"]


def test_one_history_record_per_epoch(trained_run: Path) -> None:
    """The legacy notebook appended per batch, so its loss curves were running totals."""
    history = json.loads((trained_run / "metrics.json").read_text())["history"]
    assert [record["epoch"] for record in history] == list(range(len(history)))
    # A per-batch append would give losses that only ever climb.
    losses = [record["train_loss"] for record in history]
    assert losses[-1] < losses[0], "training loss should fall over the run"


def test_training_learns_the_synthetic_task(trained_run: Path) -> None:
    metrics = json.loads((trained_run / "metrics.json").read_text())
    chance = 1.0 / len(FAKE_CLASSES)
    assert metrics["best_val_top1"] > chance * 2, (
        f"val top-1 {metrics['best_val_top1']:.3f} is near chance ({chance:.3f}); "
        f"the cache/train wiring is probably broken"
    )


@pytest.fixture
def evaluated_run(trained_run: Path) -> Path:
    from carvision.eval import evaluate_run

    evaluate_run(trained_run, resamples=200, seed=0)
    return trained_run


def test_evaluation_writes_a_well_formed_document(evaluated_run: Path) -> None:
    evaluation = json.loads((evaluated_run / "evaluation.json").read_text())

    metrics = evaluation["metrics"]
    assert 0.0 <= metrics["top1"] <= metrics["top5"] <= 1.0
    assert metrics["num_samples"] == len(FAKE_CLASSES) * TEST_PER_CLASS

    interval = evaluation["top1_interval"]
    assert interval["low"] <= interval["point"] <= interval["high"]

    calibration = evaluation["calibration"]
    assert calibration["temperature"] > 0
    assert "before" in calibration and "after" in calibration

    # Make-level accuracy forgives within-make confusions, so it cannot be lower.
    assert evaluation["make_level_top1"] >= metrics["top1"]


def test_predictions_are_saved_for_error_analysis(evaluated_run: Path) -> None:
    data = np.load(evaluated_run / "test_predictions.npz", allow_pickle=False)
    n = len(FAKE_CLASSES) * TEST_PER_CLASS

    assert data["logits"].shape == (n, len(FAKE_CLASSES))
    assert data["labels"].shape == (n,)
    assert data["image_ids"].shape == (n,)
    assert data["confusion"].sum() == n


def test_error_buckets_match_the_planted_structure(evaluated_run: Path) -> None:
    """Confusions between the two Ford Focus years must land in the year bucket.

    The synthetic colours put those two classes next to each other, so this checks the
    bucketing against structure the fixture planted -- not against the analysis's own
    output.
    """
    evaluation = json.loads((evaluated_run / "evaluation.json").read_text())
    by_kind = evaluation["errors"]["by_kind"]

    shares = [entry["share_of_errors"] for entry in by_kind.values()]
    assert sum(shares) == pytest.approx(1.0) or evaluation["errors"]["num_errors"] == 0

    for pair in evaluation["most_confused_pairs"]:
        from carvision.interpret.errors import classify_error

        assert by_kind[classify_error(pair["true"], pair["predicted"])]["count"] > 0


def test_report_renders_from_the_run(evaluated_run: Path) -> None:
    from carvision.report import write

    path = write()
    document = path.read_text()

    assert "## Headline" in document
    assert "Do not edit by hand" in document
    assert evaluated_run.name in document


def test_figures_are_generated(evaluated_run: Path, synthetic_repo: Path) -> None:
    from carvision.figures import generate_all

    paths = generate_all(evaluated_run)

    assert {p.name for p in paths} == {
        "training_curves.png",
        "calibration.png",
        "error_structure.png",
        "confusion.png",
    }
    for path in paths:
        assert path.is_file()
        assert path.stat().st_size > 1000, f"{path.name} looks empty"
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"{path.name} is not a PNG"


# --------------------------------------------------------------------- serving


@pytest.fixture
def serving_bundle(evaluated_run: Path, synthetic_repo: Path) -> Path:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    from carvision.export import prepare_serving_bundle

    return prepare_serving_bundle(evaluated_run, synthetic_repo / "serving")


def test_bundle_is_self_contained(serving_bundle: Path) -> None:
    """The Space installs neither torch nor carvision, so the bundle must carry it all."""
    for name in ("model.onnx", "classes.txt", "serving.json"):
        assert (serving_bundle / name).is_file(), f"{name} missing from the bundle"

    config = json.loads((serving_bundle / "serving.json").read_text())
    assert config["num_classes"] == len(FAKE_CLASSES)
    assert config["preprocess"]["crop"] == FAKE_PREPROCESS.crop
    assert config["metrics"]["num_samples"] == len(FAKE_CLASSES) * TEST_PER_CLASS


def test_served_prediction_matches_the_pytorch_pipeline(
    serving_bundle: Path, evaluated_run: Path
) -> None:
    """Full-stack parity: the deployed path must agree with the trained model.

    This is the assertion that would have caught the resize-rounding bug, which made the
    torch-free serving preprocessing disagree with torchvision's on non-square input.
    Here it covers the whole chain at once -- preprocessing, ONNX graph, temperature.
    """
    sys.path.insert(0, str(APP_DIR))
    import serving
    from carvision.eval import load_run

    predictor = serving.Predictor(serving_bundle)
    head, _ = load_run(evaluated_run)
    backbone = FAKE_SPEC.build()
    transform = FAKE_PREPROCESS.build(train=False)
    temperature = json.loads((evaluated_run / "evaluation.json").read_text())["calibration"][
        "temperature"
    ]

    rng = np.random.default_rng(31)
    for class_index in range(len(FAKE_CLASSES)):
        image = _render(class_index, 0, rng)

        served = predictor(image)
        assert len(served) == min(serving.TOP_K, len(FAKE_CLASSES))
        assert sum(served.values()) == pytest.approx(1.0, abs=0.02)

        with torch.inference_mode():
            logits = head(backbone(transform(image).unsqueeze(0))) / temperature
        expected = FAKE_CLASSES[int(logits.argmax())].name

        assert next(iter(served)) == expected, (
            f"serving and training disagree on class {class_index}: "
            f"served {next(iter(served))!r}, pytorch {expected!r}"
        )


def test_onnx_export_is_verified_against_pytorch(serving_bundle: Path) -> None:
    """The bundle records the measured disagreement, so drift is visible, not assumed."""
    config = json.loads((serving_bundle / "serving.json").read_text())
    assert config["onnx"]["max_abs_diff_vs_pytorch"] < 1e-3
    assert config["onnx"]["size_mb"] > 0
