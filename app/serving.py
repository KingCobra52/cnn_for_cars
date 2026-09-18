"""Torch-free inference for the deployed demo.

This file is deliberately self-contained and imports neither torch nor torchvision nor
the ``carvision`` package. The Hugging Face Space and the Docker image install only
``onnxruntime``, ``pillow`` and ``numpy``, which is the difference between a ~150 MB
image and a ~2 GB one, and it is what keeps cold starts on a free CPU Space bearable.

The cost of that independence is a second implementation of the preprocessing, which is
exactly the kind of duplication that silently drifts. So it does not go unchecked:
``tests/test_serving_parity.py`` asserts that this implementation and the torchvision one
in ``carvision.data.transforms`` produce numerically equivalent tensors, and that this
file imports no torch. If they ever diverge, CI fails.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

TOP_K = 5

#: Default bundle location, overridable with CARVISION_MODEL_DIR.
MODEL_DIR_DEFAULT = Path(os.environ.get("CARVISION_MODEL_DIR", "artifacts/serving"))


def resize_shorter_side(image: Image.Image, size: int) -> Image.Image:
    """Resize so the shorter side is ``size``, preserving aspect ratio.

    Matches ``torchvision.transforms.v2.Resize(size)`` with an int argument, including
    its rounding: torchvision **truncates** the computed long side rather than rounding
    it. That one-pixel difference shifts the subsequent centre crop and yields a
    completely different tensor, so it is not cosmetic -- see
    ``tests/test_serving_parity.py``, which caught exactly this.

    Args:
        image: The input image.
        size: Target length of the shorter side.

    Returns:
        The resized image.
    """
    width, height = image.size
    if width <= height:
        new_width, new_height = size, int(size * height / width)
    else:
        new_width, new_height = int(size * width / height), size
    # BICUBIC with antialiasing, matching the transform spec's default.
    return image.resize((new_width, new_height), Image.BICUBIC)


def center_crop(image: Image.Image, size: int) -> Image.Image:
    """Crop a centred square of side ``size``.

    Matches ``torchvision.transforms.v2.CenterCrop(size)``, including its rounding:
    torchvision computes the top-left corner as ``round((dim - size) / 2.0)``.

    Args:
        image: The input image.
        size: Side length of the crop.

    Returns:
        The cropped image.
    """
    width, height = image.size
    left = round((width - size) / 2.0)
    top = round((height - size) / 2.0)
    return image.crop((left, top, left + size, top + size))


def preprocess(
    image: Image.Image,
    *,
    resize: int,
    crop: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> np.ndarray:
    """Turn a PIL image into a normalised NCHW float32 batch of one.

    Args:
        image: The input image, converted to RGB internally.
        resize: Shorter-side length before cropping.
        crop: Final square crop size.
        mean: Per-channel normalisation mean.
        std: Per-channel normalisation standard deviation.

    Returns:
        A ``(1, 3, crop, crop)`` float32 array.
    """
    rgb = center_crop(resize_shorter_side(image.convert("RGB"), resize), crop)

    array = np.asarray(rgb, dtype=np.float32) / 255.0
    array = (array - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    # HWC -> CHW, then add the batch axis.
    return np.ascontiguousarray(array.transpose(2, 0, 1))[None]


def softmax(scores: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D score vector."""
    shifted = scores - scores.max()
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum()


class Predictor:
    """Loads the exported ONNX graph once and serves predictions."""

    def __init__(self, model_dir: Path) -> None:
        """Initialise from a serving bundle.

        Args:
            model_dir: Directory holding ``model.onnx``, ``classes.txt`` and
                ``serving.json``, as written by ``carvision export``.

        Raises:
            FileNotFoundError: If the bundle is missing, with instructions to build it.
        """
        import onnxruntime as ort

        onnx_path = model_dir / "model.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"No exported model at {onnx_path}. Run `make export` first, or point "
                f"CARVISION_MODEL_DIR at a directory containing one."
            )

        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.class_names = (model_dir / "classes.txt").read_text().splitlines()
        self.config: dict[str, Any] = json.loads((model_dir / "serving.json").read_text())

        spec = self.config["preprocess"]
        self._preprocess_kwargs = {
            "resize": int(spec["resize"]),
            "crop": int(spec["crop"]),
            "mean": tuple(spec["mean"]),
            "std": tuple(spec["std"]),
        }

    @property
    def backbone(self) -> str:
        """Name of the frozen backbone that produced the features."""
        return str(self.config.get("backbone", "unknown"))

    def __call__(self, image: Image.Image | None, *, top_k: int = TOP_K) -> dict[str, float]:
        """Classify one image.

        Args:
            image: The image to classify. None yields an empty result, which is what
                Gradio sends before anything is uploaded.
            top_k: How many candidates to return.

        Returns:
            A mapping from class name to calibrated probability, best first. This is the
            shape Gradio's Label component expects.
        """
        if image is None:
            return {}

        batch = preprocess(image, **self._preprocess_kwargs)  # type: ignore[arg-type]
        (logits,) = self.session.run(None, {"images": batch})

        # The calibration temperature is folded into the graph, so this is calibrated.
        probabilities = softmax(logits[0])
        top = np.argsort(-probabilities)[:top_k]
        return {self.class_names[int(index)]: float(probabilities[index]) for index in top}
