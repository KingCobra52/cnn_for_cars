"""Export correctness.

The parity check is the substance here: an ONNX graph that quietly computes something
else is a realistic failure mode, and the only defence is to compare the two runtimes on
real input.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from carvision.export import OPSET, ExportError, ServingModel, export, verify


class TinyBackbone(nn.Module):
    """A conv stem with global pooling, standing in for a real backbone."""

    def __init__(self, dim: int = 12) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, dim, 3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(torch.relu(self.conv(x))).flatten(1)


def register_tiny_backbone(monkeypatch, name: str) -> str:
    """Register the test backbone and return its name."""
    from carvision.data.transforms import PreprocessSpec
    from carvision.models.backbones import REGISTRY, BackboneSpec

    spec = BackboneSpec(
        name=name,
        weights_tag="test",
        embedding_dim=12,
        preprocess=PreprocessSpec(resize=32, crop=32, mean=(0.5,) * 3, std=(0.5,) * 3),
        factory=TinyBackbone,
    )
    monkeypatch.setitem(REGISTRY, name, spec)
    return name


def write_external_legacy_model(path: Path) -> tuple[bytes, bytes]:
    """Write a valid ONNX graph whose initializers live in ``.onnx.data``."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    graph = helper.make_graph(
        [helper.make_node("Gemm", ["images", "weight", "bias"], ["logits"])],
        "legacy",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [None, 3])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, 2])],
        [
            numpy_helper.from_array(np.arange(6, dtype=np.float32).reshape(3, 2), "weight"),
            numpy_helper.from_array(np.array([0.25, -0.25], dtype=np.float32), "bias"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.save_model(
        model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{path.name}.data",
        size_threshold=0,
    )
    onnx.checker.check_model(onnx.load(path, load_external_data=True))
    sidecar = Path(f"{path}.data")
    return path.read_bytes(), sidecar.read_bytes()


@pytest.fixture
def serving_model() -> ServingModel:
    torch.manual_seed(0)
    return ServingModel(TinyBackbone(), nn.Linear(12, 7), temperature=1.4).eval()


# ------------------------------------------------------------------ ServingModel


def test_serving_model_applies_temperature() -> None:
    """The graph must divide by the temperature, not ignore it."""
    torch.manual_seed(0)
    backbone, head = TinyBackbone(), nn.Linear(12, 7)
    images = torch.randn(2, 3, 32, 32)

    plain = ServingModel(backbone, head, temperature=1.0).eval()
    scaled = ServingModel(backbone, head, temperature=2.0).eval()

    with torch.inference_mode():
        torch.testing.assert_close(scaled(images), plain(images) / 2.0)


def test_serving_model_rejects_a_non_positive_temperature() -> None:
    with pytest.raises(ValueError, match="positive"):
        ServingModel(TinyBackbone(), nn.Linear(12, 7), temperature=0.0)


def test_temperature_does_not_change_the_prediction(serving_model: ServingModel) -> None:
    images = torch.randn(4, 3, 32, 32)
    plain = ServingModel(serving_model.backbone, serving_model.head, 1.0).eval()
    with torch.inference_mode():
        assert torch.equal(serving_model(images).argmax(1), plain(images).argmax(1))


# ------------------------------------------------------------------ export + parity


@pytest.fixture
def exported(tmp_path, serving_model: ServingModel):
    onnx = pytest.importorskip("onnx")  # noqa: F841
    pytest.importorskip("onnxruntime")

    path = tmp_path / "model.onnx"
    example = torch.randn(3, 3, 32, 32)
    torch.onnx.export(
        serving_model,
        (example,),
        str(path),
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    return path, example


def test_export_matches_pytorch(serving_model: ServingModel, exported) -> None:
    path, example = exported
    result = verify(serving_model, path, example)

    assert result.max_abs_diff < 1e-4
    assert result.size_mb > 0
    assert result.input_shape == (3, 3, 32, 32)


def test_verify_rejects_a_mismatched_model(serving_model: ServingModel, exported) -> None:
    """Verifying a *different* model against the graph must fail, or the check is inert."""
    path, example = exported
    torch.manual_seed(99)
    other = ServingModel(TinyBackbone(), nn.Linear(12, 7), temperature=1.4).eval()

    with pytest.raises(ExportError, match="disagrees"):
        verify(other, path, example)


def test_exported_graph_accepts_a_different_batch_size(exported) -> None:
    """The dynamic batch axis is what lets one graph serve 1 image or 100."""
    ort = pytest.importorskip("onnxruntime")
    path, _ = exported

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for batch in (1, 5):
        (logits,) = session.run(
            None, {"images": np.random.randn(batch, 3, 32, 32).astype("float32")}
        )
        assert logits.shape == (batch, 7)


def test_export_writes_one_self_contained_model_file(tmp_path, monkeypatch) -> None:
    """Large production graphs must not hide untracked weights in a sidecar file."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    name = register_tiny_backbone(monkeypatch, "tiny_single_file")
    path = tmp_path / "model.onnx"
    write_external_legacy_model(path)

    result = export(name, nn.Linear(12, 7), path, image_size=32, verify_batch=1)

    assert path.is_file()
    assert not Path(f"{path}.data").exists()
    assert result.path == path
    onnx = pytest.importorskip("onnx")
    model = onnx.load(path, load_external_data=False)
    assert all(not tensor.external_data for tensor in model.graph.initializer)
    assert next(item.version for item in model.opset_import if item.domain == "") == OPSET


@pytest.mark.parametrize("failure", ["exporter", "parity", "replacement"])
def test_failed_export_preserves_external_legacy_model(tmp_path, monkeypatch, failure) -> None:
    """Every pre-publication failure preserves a loadable legacy graph byte-for-byte."""
    name = register_tiny_backbone(monkeypatch, f"tiny_failed_{failure}")
    path = tmp_path / "model.onnx"
    sidecar = Path(f"{path}.data")
    old_graph, old_weights = write_external_legacy_model(path)
    export_module = importlib.import_module("carvision.export")

    def fail_export(*args, **kwargs) -> None:
        Path(args[2]).write_bytes(b"partial replacement")
        raise RuntimeError("injected exporter failure")

    def fail_parity(*args, **kwargs) -> None:
        raise ExportError("injected parity failure")

    def fail_replacement(*args, **kwargs) -> None:
        raise OSError("injected replacement failure")

    if failure == "exporter":
        monkeypatch.setattr(torch.onnx, "export", fail_export)
    elif failure == "parity":
        monkeypatch.setattr(export_module, "verify", fail_parity)
    else:
        monkeypatch.setattr(Path, "replace", fail_replacement)

    with pytest.raises((RuntimeError, ExportError, OSError), match=f"injected {failure}"):
        export(name, nn.Linear(12, 7), path, image_size=32, verify_batch=1)

    assert path.read_bytes() == old_graph
    assert sidecar.read_bytes() == old_weights
    onnx = pytest.importorskip("onnx")
    onnx.checker.check_model(onnx.load(path, load_external_data=True))
    assert set(tmp_path.iterdir()) == {path, sidecar}


@pytest.mark.parametrize("failure", ["exporter", "parity", "replacement"])
def test_failed_first_export_leaves_no_files(tmp_path, monkeypatch, failure) -> None:
    """A first export is all-or-nothing and cleans its private staging directory."""
    name = register_tiny_backbone(monkeypatch, f"tiny_first_{failure}")
    path = tmp_path / "model.onnx"
    export_module = importlib.import_module("carvision.export")

    def fail_export(*args, **kwargs) -> None:
        raise RuntimeError("injected exporter failure")

    def fail_parity(*args, **kwargs) -> None:
        raise ExportError("injected parity failure")

    def fail_replacement(*args, **kwargs) -> None:
        raise OSError("injected replacement failure")

    if failure == "exporter":
        monkeypatch.setattr(torch.onnx, "export", fail_export)
    elif failure == "parity":
        monkeypatch.setattr(export_module, "verify", fail_parity)
    else:
        monkeypatch.setattr(Path, "replace", fail_replacement)

    with pytest.raises((RuntimeError, ExportError, OSError), match=f"injected {failure}"):
        export(name, nn.Linear(12, 7), path, image_size=32, verify_batch=1)

    assert list(tmp_path.iterdir()) == []


def test_sidecar_cleanup_failure_keeps_new_graph(tmp_path, monkeypatch, caplog) -> None:
    """Obsolete-weight cleanup is non-fatal after the graph has been published."""
    name = register_tiny_backbone(monkeypatch, "tiny_cleanup_warning")
    path = tmp_path / "model.onnx"
    write_external_legacy_model(path)
    sidecar = Path(f"{path}.data")
    original_unlink = Path.unlink

    def fail_sidecar_unlink(self, *args, **kwargs):
        if self == sidecar:
            raise PermissionError("injected cleanup failure")
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_sidecar_unlink)
    result = export(name, nn.Linear(12, 7), path, image_size=32, verify_batch=1)

    assert result.path == path
    assert sidecar.exists()
    assert "Could not remove legacy ONNX sidecar" in caplog.text
    onnx = pytest.importorskip("onnx")
    onnx.checker.check_model(onnx.load(path, load_external_data=False))
