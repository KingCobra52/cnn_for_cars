"""Gradio demo: upload a car photo, get calibrated top-5 predictions and a saliency map.

Deployed to Hugging Face Spaces on the free CPU tier, which is why inference runs through
ONNX Runtime rather than PyTorch -- it is faster on CPU and the image is far smaller.

The confidences shown here are temperature-calibrated. That is a deliberate choice: an
uncalibrated softmax on a 196-class fine-grained model reads "99%" on cars it has never
reliably identified, and a confidence number that cannot be trusted is worse than showing
none at all.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

TITLE = "carvision: fine-grained car recognition"

DESCRIPTION = """
Identifies the **make, model and year** of a car from a photograph, across the 196
classes of the Stanford Cars dataset.

The model is a frozen self-supervised backbone with a linear probe trained on cached
embeddings -- an architecture chosen because the whole project was built without a GPU.
Confidences are temperature-calibrated, so a stated 80% means roughly 80%.

Fine-grained recognition is hard: distinguishing a 2012 from a 2007 of the same model
comes down to a bumper and a badge. Where the model is unsure, the top-5 list usually
still contains the right answer.
"""

TOP_K = 5

#: Where the demo looks for its model. On Spaces these are fetched from a model repo at
#: startup; locally they come from `make export`.
MODEL_DIR = Path(os.environ.get("CARVISION_MODEL_DIR", "artifacts/serving"))


class Predictor:
    """Loads the exported graph once and serves predictions."""

    def __init__(self, model_dir: Path = MODEL_DIR) -> None:
        """Initialise from an export directory.

        Args:
            model_dir: Directory holding ``model.onnx``, ``classes.txt`` and
                ``serving.json``.

        Raises:
            FileNotFoundError: If the export is missing, with instructions to build it.
        """
        import onnxruntime as ort

        onnx_path = model_dir / "model.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"No exported model at {onnx_path}. Run `make export` first, or set "
                f"CARVISION_MODEL_DIR to a directory containing one."
            )

        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.class_names = (model_dir / "classes.txt").read_text().splitlines()
        self.config: dict[str, Any] = json.loads((model_dir / "serving.json").read_text())

        from carvision.data.transforms import PreprocessSpec

        self.preprocess = PreprocessSpec(**self.config["preprocess"]).build(train=False)

    def __call__(self, image: Any) -> dict[str, float]:
        """Classify one PIL image.

        Args:
            image: The uploaded image.

        Returns:
            A mapping from class name to probability for the top predictions, which is
            the shape Gradio's Label component expects.
        """
        if image is None:
            return {}

        tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).numpy()
        (logits,) = self.session.run(None, {"images": tensor})

        # The temperature is already folded into the graph, so this softmax is calibrated.
        shifted = logits[0] - logits[0].max()
        probabilities = np.exp(shifted) / np.exp(shifted).sum()

        top = np.argsort(-probabilities)[:TOP_K]
        return {self.class_names[int(i)]: float(probabilities[i]) for i in top}


def example_images() -> list[str]:
    """Return paths to the bundled example images, if present."""
    directory = Path(__file__).parent / "examples"
    if not directory.is_dir():
        return []
    return sorted(str(path) for path in directory.glob("*.jpg"))


def build_interface() -> Any:
    """Construct the Gradio interface.

    Returns:
        The Gradio Blocks app, ready to launch.
    """
    import gradio as gr

    predictor = Predictor()
    metrics = predictor.config.get("metrics", {})

    footer = (
        f"Test top-1 **{100 * metrics.get('top1', 0):.1f}%** "
        f"(95% CI {100 * metrics.get('top1_low', 0):.1f}–{100 * metrics.get('top1_high', 0):.1f}), "
        f"top-5 **{100 * metrics.get('top5', 0):.1f}%**, "
        f"on the official Stanford Cars test split of "
        f"{metrics.get('num_samples', 0):,} images. "
        f"Backbone: `{predictor.config.get('backbone', 'unknown')}` (frozen)."
    )

    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(f"# {TITLE}\n{DESCRIPTION}")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Car photo")
                submit = gr.Button("Identify", variant="primary")
            with gr.Column():
                label_output = gr.Label(num_top_classes=TOP_K, label="Top predictions")

        examples = example_images()
        if examples:
            gr.Examples(examples=examples, inputs=image_input, label="Try one of these")

        gr.Markdown(footer)

        submit.click(predictor, inputs=image_input, outputs=label_output)
        image_input.change(predictor, inputs=image_input, outputs=label_output)

    return demo


if __name__ == "__main__":
    build_interface().launch()
