"""Export a trained model to ONNX, and prove the export is faithful.

Training and serving are separate concerns here: training reads cached embeddings, while
serving must take a raw image. So the exported graph is the backbone and the head fused
into one module, with temperature folded in, giving a single file that maps a
preprocessed image tensor to calibrated logits.

Export is only worth anything if the exported graph computes the same function as the
model it came from. Operator coverage differs between PyTorch and ONNX Runtime, and a
silently wrong export is a realistic failure. So :func:`verify` is not optional -- every
export runs it and refuses to report success without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from carvision.models.backbones import get_backbone
from carvision.utils.logging import get_logger
from carvision.utils.paths import ensure_dir

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger(__name__)

#: ONNX opset. 17 covers LayerNorm natively, which the heads use.
OPSET = 17

#: Tolerance for PyTorch/ONNX agreement. Float32 accumulation order differs between the
#: two runtimes, so exact equality is not a reasonable bar; 1e-3 relative is.
PARITY_RTOL = 1e-3
PARITY_ATOL = 1e-4


class ExportError(RuntimeError):
    """Raised when an export cannot be produced or fails verification."""


class ServingModel(nn.Module):
    """Backbone, head and temperature as one module.

    Everything the served graph needs, so inference is a single call on a preprocessed
    image rather than an orchestration of three objects.
    """

    def __init__(self, backbone: nn.Module, head: nn.Module, temperature: float = 1.0) -> None:
        """Initialise.

        Args:
            backbone: The frozen feature extractor.
            head: The trained classifier head.
            temperature: Calibration temperature to fold into the graph, so the served
                probabilities are the calibrated ones without a separate step.

        Raises:
            ValueError: If the temperature is not positive.
        """
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        self.backbone = backbone
        self.head = head
        # A buffer rather than a Python float, so it is captured in the traced graph.
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Map ``(B, 3, H, W)`` preprocessed images to calibrated logits."""
        logits: torch.Tensor = self.head(self.backbone(images)) / self.temperature
        return logits


@dataclass(frozen=True)
class ExportResult:
    """A verified ONNX export.

    Attributes:
        path: Where the graph was written.
        max_abs_diff: Largest absolute disagreement with PyTorch over the probe batch.
        max_rel_diff: Largest relative disagreement.
        size_mb: File size in megabytes.
        input_shape: The probe input shape.
    """

    path: Path
    max_abs_diff: float
    max_rel_diff: float
    size_mb: float
    input_shape: tuple[int, ...]


def export(
    backbone_name: str,
    head: nn.Module,
    output_path: Path,
    *,
    temperature: float = 1.0,
    image_size: int | None = None,
    verify_batch: int = 4,
) -> ExportResult:
    """Export a backbone-plus-head to ONNX and verify it against PyTorch.

    Args:
        backbone_name: A registered backbone name.
        head: The trained head.
        output_path: Destination ``.onnx`` file.
        temperature: Calibration temperature to fold in.
        image_size: Input side length. Defaults to the backbone's own crop size.
        verify_batch: Batch size used for the parity check.

    Returns:
        The verified export.

    Raises:
        ExportError: If the exported graph disagrees with PyTorch beyond tolerance.
    """
    spec = get_backbone(backbone_name)
    size = image_size or spec.preprocess.crop

    model = ServingModel(spec.build(), head, temperature).eval()
    example = torch.randn(verify_batch, 3, size, size)

    ensure_dir(output_path.parent)
    logger.info("Exporting %s + %s -> %s", backbone_name, type(head).__name__, output_path)

    torch.onnx.export(
        model,
        (example,),
        str(output_path),
        input_names=["images"],
        output_names=["logits"],
        # A dynamic batch axis lets the same graph serve one image or a hundred.
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=OPSET,
        do_constant_folding=True,
    )

    result = verify(model, output_path, example)
    logger.info(
        "Export verified: max abs diff %.2e, max rel diff %.2e, %.1f MB",
        result.max_abs_diff,
        result.max_rel_diff,
        result.size_mb,
    )
    return result


def verify(
    model: nn.Module,
    onnx_path: Path,
    example: torch.Tensor,
    *,
    rtol: float = PARITY_RTOL,
    atol: float = PARITY_ATOL,
) -> ExportResult:
    """Check that the ONNX graph matches PyTorch on a probe batch.

    Args:
        model: The PyTorch module that was exported.
        onnx_path: The exported graph.
        example: Probe input.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Returns:
        The export result, including the observed disagreement.

    Raises:
        ExportError: If the outputs differ beyond tolerance.
    """
    import onnxruntime as ort

    with torch.inference_mode():
        expected = model(example).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    (actual,) = session.run(None, {"images": example.numpy()})

    if expected.shape != actual.shape:
        raise ExportError(f"Shape mismatch: PyTorch {expected.shape} vs ONNX {actual.shape}.")

    absolute = np.abs(expected - actual)
    relative = absolute / np.maximum(np.abs(expected), 1e-8)
    max_abs, max_rel = float(absolute.max()), float(relative.max())

    if not np.allclose(expected, actual, rtol=rtol, atol=atol):
        raise ExportError(
            f"ONNX output disagrees with PyTorch: max abs diff {max_abs:.3e}, max rel "
            f"diff {max_rel:.3e} (tolerance rtol={rtol}, atol={atol}). The exported "
            f"graph does not compute the same function; do not serve it."
        )

    return ExportResult(
        path=onnx_path,
        max_abs_diff=max_abs,
        max_rel_diff=max_rel,
        size_mb=onnx_path.stat().st_size / 1024**2,
        input_shape=tuple(example.shape),
    )


@dataclass(frozen=True)
class LatencyStats:
    """Wall-clock latency for one runtime.

    Attributes:
        runtime: ``"pytorch"`` or ``"onnxruntime"``.
        batch_size: Images per call.
        p50_ms: Median latency.
        p95_ms: 95th percentile, which is what a user actually feels.
        mean_ms: Mean latency.
        runs: Number of timed calls.
    """

    runtime: str
    batch_size: int
    p50_ms: float
    p95_ms: float
    mean_ms: float
    runs: int

    def as_dict(self) -> dict[str, float | int | str]:
        """Return the stats as a plain dict, ready for JSON."""
        return {
            "runtime": self.runtime,
            "batch_size": self.batch_size,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "mean_ms": self.mean_ms,
            "runs": self.runs,
        }


def _summarise(runtime: str, batch_size: int, timings_ms: Sequence[float]) -> LatencyStats:
    array = np.asarray(timings_ms)
    return LatencyStats(
        runtime=runtime,
        batch_size=batch_size,
        p50_ms=float(np.percentile(array, 50)),
        p95_ms=float(np.percentile(array, 95)),
        mean_ms=float(array.mean()),
        runs=len(array),
    )


def benchmark_pytorch(
    model: nn.Module,
    example: torch.Tensor,
    *,
    runs: int = 100,
    warmup: int = 10,
) -> LatencyStats:
    """Time PyTorch inference.

    Args:
        model: The module to time, already in eval mode.
        example: A representative input batch.
        runs: Timed iterations.
        warmup: Untimed iterations first. These matter: the first calls pay for lazy
            kernel selection and allocator warm-up, and including them would report a
            latency no steady-state user ever sees.

    Returns:
        The latency summary.
    """
    import time

    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(example)

        timings = []
        for _ in range(runs):
            start = time.perf_counter()
            model(example)
            timings.append((time.perf_counter() - start) * 1000)

    return _summarise("pytorch", len(example), timings)


def benchmark_onnx(
    onnx_path: Path,
    example: torch.Tensor,
    *,
    runs: int = 100,
    warmup: int = 10,
) -> LatencyStats:
    """Time ONNX Runtime inference.

    Args:
        onnx_path: The exported graph.
        example: A representative input batch.
        runs: Timed iterations.
        warmup: Untimed iterations first.

    Returns:
        The latency summary.
    """
    import time

    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = {"images": example.numpy()}

    for _ in range(warmup):
        session.run(None, inputs)

    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, inputs)
        timings.append((time.perf_counter() - start) * 1000)

    return _summarise("onnxruntime", len(example), timings)


def prepare_serving_bundle(
    run_dir: Path,
    output_dir: Path,
    *,
    temperature: float | None = None,
) -> Path:
    """Assemble everything the demo and the API need into one directory.

    The bundle is self-contained on purpose: the Space that serves it has neither torch
    nor this package installed, so the graph, the class names and the preprocessing
    parameters all have to travel together.

    Args:
        run_dir: A trained, evaluated run directory.
        output_dir: Destination directory for the bundle.
        temperature: Calibration temperature. Defaults to the one in the run's
            ``evaluation.json``.

    Returns:
        The bundle directory.

    Raises:
        ExportError: If the run has not been evaluated, so no temperature or metrics
            exist to serve alongside the graph.
    """
    import json as json_module
    from dataclasses import asdict

    from carvision.data.download import load_class_names
    from carvision.eval import load_run

    evaluation_path = run_dir / "evaluation.json"
    if not evaluation_path.exists():
        raise ExportError(
            f"{run_dir.name} has not been evaluated. Run `carvision eval` first -- the "
            f"bundle needs its calibration temperature and headline metrics."
        )
    evaluation = json_module.loads(evaluation_path.read_text())

    head, payload = load_run(run_dir)
    backbone_name = str(payload["backbone"])
    spec = get_backbone(backbone_name)

    ensure_dir(output_dir)
    result = export(
        backbone_name,
        head,
        output_dir / "model.onnx",
        temperature=(
            temperature
            if temperature is not None
            else float(evaluation["calibration"]["temperature"])
        ),
    )

    (output_dir / "classes.txt").write_text("\n".join(load_class_names()) + "\n")
    (output_dir / "serving.json").write_text(
        json_module.dumps(
            {
                "run": run_dir.name,
                "backbone": backbone_name,
                "head": payload["head_kind"],
                "num_classes": payload["num_classes"],
                "preprocess": asdict(spec.preprocess),
                "temperature": evaluation["calibration"]["temperature"],
                "metrics": {
                    "top1": evaluation["metrics"]["top1"],
                    "top5": evaluation["metrics"]["top5"],
                    "top1_low": evaluation["top1_interval"]["low"],
                    "top1_high": evaluation["top1_interval"]["high"],
                    "num_samples": evaluation["metrics"]["num_samples"],
                },
                "onnx": {
                    "opset": OPSET,
                    "size_mb": round(result.size_mb, 2),
                    "max_abs_diff_vs_pytorch": result.max_abs_diff,
                },
            },
            indent=2,
        )
        + "\n"
    )

    logger.info("Serving bundle ready at %s (%.1f MB)", output_dir, result.size_mb)
    return output_dir
