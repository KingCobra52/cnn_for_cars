"""Regression tests for the second batch of review findings.

Each test reproduces the specific failure the fix addresses, so a revert is caught
rather than merely made less likely.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

# ------------------------------------------------------------------ splits validated first


def test_a_leaking_split_never_reaches_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validation must happen before the write.

    Checking afterwards left a leaking split on disk, where the --overwrite guard then
    protected it from being regenerated -- so the bad file was both written and hard to
    replace.
    """
    import pandas as pd

    monkeypatch.setenv("CARVISION_ROOT", str(tmp_path))
    from carvision.data.splits import SplitError, build_splits
    from carvision.utils.paths import data_dir, splits_dir

    root = data_dir() / "stanford_cars"
    root.mkdir(parents=True)
    # The same image id in both the official train and test splits: guaranteed leakage.
    pd.DataFrame(
        [
            {"image_id": "dup", "split": "train", "label_id": 0, "label_name": "a", "relpath": "x"},
            {"image_id": "t2", "split": "train", "label_id": 0, "label_name": "a", "relpath": "x"},
            {"image_id": "t3", "split": "train", "label_id": 1, "label_name": "b", "relpath": "x"},
            {"image_id": "t4", "split": "train", "label_id": 1, "label_name": "b", "relpath": "x"},
            {"image_id": "dup", "split": "test", "label_id": 0, "label_name": "a", "relpath": "x"},
        ]
    ).to_csv(root / "manifest.csv", index=False)

    with pytest.raises(SplitError, match="both"):
        build_splits()

    written = list(splits_dir().glob("*.csv")) if splits_dir().exists() else []
    assert written == [], f"a leaking split was written anyway: {written}"


# ------------------------------------------------------------------ cache.clear


def _make_entry(directory: Path, *, interrupted: bool) -> None:
    directory.mkdir(parents=True)
    (directory / "embeddings.npy").write_bytes(b"")
    if interrupted:
        (directory / "progress.json").write_text('{"rows_done": 64}')
    else:
        (directory / "manifest.json").write_text('{"key": "k", "num_rows": 1, "dim": 1}')


def test_clear_removes_interrupted_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An interrupted entry has no manifest, so globbing for one skipped it.

    Left behind, `clear` reported "0 removed" and the next non-forced build silently
    resumed the very entry the user had asked to delete.
    """
    monkeypatch.setenv("CARVISION_ROOT", str(tmp_path))
    from carvision.features.cache import clear
    from carvision.utils.paths import cache_dir

    _make_entry(cache_dir() / "resnet50" / "train-aaaa", interrupted=False)
    _make_entry(cache_dir() / "resnet50" / "val-bbbb", interrupted=True)

    assert clear("resnet50") == 2
    assert not (cache_dir() / "resnet50" / "val-bbbb").exists()
    assert not (cache_dir() / "resnet50" / "train-aaaa").exists()


def test_clear_is_scoped_to_one_backbone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CARVISION_ROOT", str(tmp_path))
    from carvision.features.cache import clear
    from carvision.utils.paths import cache_dir

    _make_entry(cache_dir() / "resnet50" / "train-aaaa", interrupted=True)
    _make_entry(cache_dir() / "clip_vitb32" / "train-cccc", interrupted=False)

    assert clear("resnet50") == 1
    assert (cache_dir() / "clip_vitb32" / "train-cccc").exists()


def test_clear_on_an_empty_cache_is_a_no_op(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CARVISION_ROOT", str(tmp_path))
    from carvision.features.cache import clear

    assert clear() == 0


# ------------------------------------------------------------------ num_classes


def test_num_classes_comes_from_the_declared_class_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A class absent from train+val must not silently narrow the head.

    Inferring max(label)+1 would build a 3-way head while classes.txt, the confusion
    matrix and the served class names all still assume 5.
    """
    monkeypatch.setenv("CARVISION_ROOT", str(tmp_path))
    root = tmp_path / "data" / "stanford_cars"
    root.mkdir(parents=True)
    (root / "classes.txt").write_text("a\nb\nc\nd\ne\n")

    from carvision.train import resolve_num_classes

    # Labels only reach 2, but five classes are declared.
    assert resolve_num_classes(np.array([0, 1, 2]), np.array([0, 1])) == 5


def test_num_classes_falls_back_when_there_is_no_class_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Synthetic fixtures have no classes.txt; inference is the right answer there."""
    monkeypatch.setenv("CARVISION_ROOT", str(tmp_path))
    from carvision.train import resolve_num_classes

    assert resolve_num_classes(np.array([0, 1, 2])) == 3


def test_num_classes_rejects_labels_beyond_the_class_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Labels outside the declared set mean the split and class list disagree."""
    monkeypatch.setenv("CARVISION_ROOT", str(tmp_path))
    root = tmp_path / "data" / "stanford_cars"
    root.mkdir(parents=True)
    (root / "classes.txt").write_text("a\nb\n")

    from carvision.train import TrainingError, resolve_num_classes

    with pytest.raises(TrainingError, match="disagree"):
        resolve_num_classes(np.array([0, 1, 7]))


# ------------------------------------------------------------------ empty best_state


def test_a_run_with_no_improving_epoch_fails_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """max_epochs=0 previously died on load_state_dict({}) with "Missing key(s)"."""
    monkeypatch.setenv("CARVISION_ROOT", str(tmp_path))

    from carvision.features import cache
    from carvision.train import TrainConfig, TrainingError, train

    embeddings = np.random.default_rng(0).normal(size=(8, 4)).astype(np.float32)
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    def fake_load(backbone: str, split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        return embeddings, labels, [f"{split}_{i}" for i in range(len(labels))]

    monkeypatch.setattr(cache, "load", fake_load)

    class FakeSpec:
        embedding_dim = 4

        @staticmethod
        def cache_key() -> str:
            return "fake@v1"

    monkeypatch.setattr("carvision.train.get_backbone", lambda _name: FakeSpec())
    input_file = tmp_path / "training-input"
    input_file.write_text("fixed")

    def fake_provenance(config, _spec):
        from carvision.utils.artifacts import dependencies

        record = dependencies([input_file], "carvision train")
        record.update(
            {
                "backbone": config.backbone,
                "backbone_signature": "fake@v1",
                "classes": [],
                "config": config.__dict__,
            }
        )
        return record

    monkeypatch.setattr("carvision.utils.artifacts.training_provenance", fake_provenance)

    with pytest.raises(TrainingError, match="No epoch improved"):
        train(TrainConfig(backbone="fake", head="linear", max_epochs=0))


# ------------------------------------------------------------------ stale artifacts


def test_retraining_clears_artifacts_derived_from_the_old_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run directories do not encode hyperparameters, so retraining reuses them.

    Leaving the previous evaluation.json in place let `report` and `carvision export`
    pair a new checkpoint with old metrics and an old calibration temperature.
    """
    monkeypatch.setenv("CARVISION_ROOT", str(tmp_path))

    from carvision.features import cache
    from carvision.train import TrainConfig, train
    from carvision.utils.paths import runs_dir

    rng = np.random.default_rng(0)
    centres = rng.normal(scale=5.0, size=(2, 4))
    labels = np.repeat([0, 1], 8)
    embeddings = (centres[labels] + rng.normal(scale=0.1, size=(16, 4))).astype(np.float32)

    def fake_load(backbone: str, split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        return embeddings, labels, [f"{split}_{i}" for i in range(len(labels))]

    monkeypatch.setattr(cache, "load", fake_load)

    class FakeSpec:
        embedding_dim = 4

        @staticmethod
        def cache_key() -> str:
            return "fake@v1"

    monkeypatch.setattr("carvision.train.get_backbone", lambda _name: FakeSpec())
    input_file = tmp_path / "training-input"
    input_file.write_text("fixed")

    def fake_provenance(config, _spec):
        from carvision.utils.artifacts import dependencies

        record = dependencies([input_file], "carvision train")
        record.update(
            {
                "backbone": config.backbone,
                "backbone_signature": "fake@v1",
                "classes": [],
                "config": config.__dict__,
            }
        )
        return record

    monkeypatch.setattr("carvision.utils.artifacts.training_provenance", fake_provenance)

    config = TrainConfig(backbone="fake", head="linear", max_epochs=5, warmup_epochs=1)
    run_dir = train(config).run_dir

    # Stand in for a previous evaluation of a now-replaced model.
    (run_dir / "evaluation.json").write_text('{"stale": true}')
    (run_dir / "test_predictions.npz").write_bytes(b"stale")
    (run_dir / "latency.json").write_text("[]")

    second = train(TrainConfig(**{**config.__dict__, "lr": 1e-1})).run_dir

    assert second == run_dir, "the fixture assumes the directory is reused"
    assert not (run_dir / "evaluation.json").exists()
    assert not (run_dir / "test_predictions.npz").exists()
    assert not (run_dir / "latency.json").exists()
    assert (run_dir / "checkpoint.pt").exists()
    assert runs_dir().exists()


# ------------------------------------------------------------------ serving temperature


def test_bundle_records_the_temperature_actually_folded_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under a --temperature override the metadata used to describe a different model."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    monkeypatch.setenv("CARVISION_ROOT", str(tmp_path))

    from carvision.data.transforms import PreprocessSpec
    from carvision.export import prepare_serving_bundle
    from carvision.models.backbones import REGISTRY, BackboneSpec

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 3, stride=2, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out: torch.Tensor = self.pool(torch.relu(self.conv(x))).flatten(1)
            return out

    spec = BackboneSpec(
        name="tiny",
        weights_tag="v1",
        embedding_dim=4,
        preprocess=PreprocessSpec(resize=18, crop=16, mean=(0.5,) * 3, std=(0.25,) * 3),
        factory=Tiny,
    )
    monkeypatch.setitem(REGISTRY, "tiny", spec)

    data_root = tmp_path / "data" / "stanford_cars"
    data_root.mkdir(parents=True)
    (data_root / "classes.txt").write_text("a\nb\n")

    run_dir = tmp_path / "artifacts" / "runs" / "tiny-linear-seed0"
    run_dir.mkdir(parents=True)
    from carvision.models.heads import build_head

    torch.save(
        {
            # Build the real head so the keys match what load_run reconstructs.
            "state_dict": build_head("linear", 4, 2).state_dict(),
            "head_kind": "linear",
            "backbone": "tiny",
            "num_classes": 2,
            "embedding_dim": 4,
        },
        run_dir / "checkpoint.pt",
    )
    (run_dir / "config.json").write_text('{"backbone": "tiny"}')
    (run_dir / "metrics.json").write_text('{"best_val_top1": 0.5}')
    from carvision.utils.artifacts import dependencies, fingerprint

    training = dependencies([data_root / "classes.txt"], "carvision train")
    training.update(
        {
            "backbone": "tiny",
            "backbone_signature": spec.cache_key(),
            "classes": ["a", "b"],
            "config": {"backbone": "tiny"},
            "outputs": {
                name: fingerprint(run_dir / name)
                for name in ("checkpoint.pt", "config.json", "metrics.json")
            },
        }
    )
    (run_dir / "training_provenance.json").write_text(json.dumps(training))
    (run_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "metrics": {"top1": 0.5, "top5": 1.0, "num_samples": 10},
                "top1_interval": {"low": 0.4, "high": 0.6},
                "calibration": {"temperature": 1.8},
            }
        )
    )

    np.savez(run_dir / "test_predictions.npz", logits=np.zeros((1, 2)))
    evaluation = json.loads((run_dir / "evaluation.json").read_text())
    evaluation["provenance"] = dependencies(
        [run_dir / "checkpoint.pt", run_dir / "test_predictions.npz"], "carvision eval"
    )
    evaluation["provenance"]["predictions_sha256"] = fingerprint(run_dir / "test_predictions.npz")
    from carvision.utils.artifacts import seal_evaluation

    seal_evaluation(evaluation)
    (run_dir / "evaluation.json").write_text(json.dumps(evaluation))
    bundle = prepare_serving_bundle(run_dir, tmp_path / "serving", temperature=3.5)
    config = json.loads((bundle / "serving.json").read_text())

    assert config["temperature"] == pytest.approx(3.5), "must record what was folded in"
    assert config["temperature_from_evaluation"] == pytest.approx(1.8)
