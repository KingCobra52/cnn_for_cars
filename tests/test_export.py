"""Export correctness.

The parity check is the substance here: an ONNX graph that quietly computes something
else is a realistic failure mode, and the only defence is to compare the two runtimes on
real input.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from carvision.export import ExportError, ServingModel, verify


class TinyBackbone(nn.Module):
    """A conv stem with global pooling, standing in for a real backbone."""

    def __init__(self, dim: int = 12) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, dim, 3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(torch.relu(self.conv(x))).flatten(1)


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
