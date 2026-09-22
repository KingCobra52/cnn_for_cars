"""End-to-end serving verification for the existing real-image pilot."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from carvision.data.download import load_class_names
from carvision.data.splits import load_split, validate_splits
from carvision.eval import find_best_run, load_run
from carvision.export import PARITY_ATOL, PARITY_RTOL, ServingModel, prepare_serving_bundle
from carvision.models.backbones import get_backbone
from carvision.utils.artifacts import (
    fingerprint,
    validate,
    validate_evaluation,
    validate_training,
)
from carvision.utils.paths import data_dir, repo_root

APP_DIR = repo_root() / "app"
sys.path.insert(0, str(APP_DIR))

import serving  # noqa: E402


def _pilot_images(count_per_orientation: int = 6) -> list[tuple[str, Path]]:
    """Choose the first image IDs in each orientation, independent of row order."""
    selected: dict[str, list[tuple[str, Path]]] = {"portrait": [], "landscape": []}
    root = data_dir() / "stanford_cars"
    for row in load_split("test").sort_values("image_id").itertuples(index=False):
        path = root / str(row.relpath)
        with Image.open(path) as image:
            orientation = "portrait" if image.height > image.width else "landscape"
        if len(selected[orientation]) < count_per_orientation:
            selected[orientation].append((str(row.image_id), path))
        if all(len(items) == count_per_orientation for items in selected.values()):
            break
    assert all(len(items) == count_per_orientation for items in selected.values())
    return sorted(selected["portrait"] + selected["landscape"])


def _verify_real_images(run: Path, bundle: Path) -> dict[str, object]:
    """Verify one run and serving bundle against deterministic real images."""
    validate_splits()
    validate_training(run)
    evaluation = validate_evaluation(run)
    assert evaluation["run"] == run.name
    assert evaluation["metrics"]["num_samples"] == len(load_split("test"))

    metadata = json.loads((bundle / "serving.json").read_text())
    validate(metadata["provenance"])
    assert metadata["run"] == run.name
    assert metadata["temperature"] == evaluation["calibration"]["temperature"]
    assert metadata["temperature_from_evaluation"] == evaluation["calibration"]["temperature"]
    assert fingerprint(bundle / "model.onnx") == metadata["onnx"]["sha256"]

    head, payload = load_run(run)
    assert metadata["backbone"] == payload["backbone"]
    assert metadata["head"] == payload["head_kind"]
    spec = get_backbone(str(payload["backbone"]))
    model = ServingModel(spec.build(), head, float(metadata["temperature"])).eval()
    training_transform = spec.preprocess.build(train=False)

    selected = _pilot_images()
    training_batches: list[np.ndarray] = []
    max_preprocess_diff = 0.0
    for _, path in selected:
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        expected = training_transform(image).numpy()[None]
        actual = serving.preprocess(
            image,
            resize=spec.preprocess.resize,
            crop=spec.preprocess.crop,
            mean=spec.preprocess.mean,
            std=spec.preprocess.std,
            interpolation=spec.preprocess.interpolation,
        )
        max_preprocess_diff = max(max_preprocess_diff, float(np.abs(expected - actual).max()))
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
        training_batches.append(expected)

    predictor = serving.Predictor(bundle)
    assert predictor.class_names == load_class_names() == evaluation["class_names"]
    batch = np.concatenate(training_batches)
    with torch.inference_mode():
        pytorch_logits = model(torch.from_numpy(batch)).numpy()
    (onnx_logits,) = predictor.session.run(None, {"images": batch})
    max_logit_diff = float(np.abs(pytorch_logits - onnx_logits).max())
    np.testing.assert_allclose(onnx_logits, pytorch_logits, rtol=PARITY_RTOL, atol=PARITY_ATOL)
    np.testing.assert_array_equal(onnx_logits.argmax(1), pytorch_logits.argmax(1))

    for index, (_, path) in enumerate(selected):
        with Image.open(path) as handle:
            predictions = predictor(handle.convert("RGB"), top_k=len(predictor.class_names))
        probabilities = np.array(list(predictions.values()))
        assert np.isfinite(probabilities).all()
        assert (probabilities >= 0).all()
        assert probabilities.sum() == pytest.approx(1.0, abs=1e-6)
        expected_name = evaluation["class_names"][int(onnx_logits[index].argmax())]
        assert next(iter(predictions)) == expected_name

    return {
        "image_ids": [image_id for image_id, _ in selected],
        "max_preprocess_diff": max_preprocess_diff,
        "max_logit_diff": max_logit_diff,
        "checkpoint_sha256": fingerprint(run / "checkpoint.pt"),
        "evaluation_sha256": fingerprint(run / "evaluation.json"),
        "onnx_sha256": fingerprint(bundle / "model.onnx"),
        "temperature": float(metadata["temperature"]),
    }


@pytest.mark.slow
def test_resnet50_pilot_real_image_serving(tmp_path, record_property) -> None:
    """Prove preprocessing, graph, probabilities, and labels agree on real images."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    run = repo_root() / "artifacts/runs/resnet50-linear-seed0"
    bundle = prepare_serving_bundle(run, tmp_path / "serving")
    result = _verify_real_images(run, bundle)
    for key, value in result.items():
        record_property(key, value)
    print("pilot verification:", result)


@pytest.mark.slow
def test_selected_model_real_image_serving(record_property) -> None:
    """Prove the selected run agrees with its existing published serving bundle."""
    run = find_best_run()
    assert run.name == "dinov2_vits14-mlp-seed3"
    result = _verify_real_images(run, repo_root() / "artifacts/serving")
    for key, value in result.items():
        record_property(key, value)
    print("selected-model verification:", result)
