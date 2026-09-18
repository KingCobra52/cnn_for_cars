"""Gradio demo: upload a car photo, get calibrated top-5 predictions.

Deployed to Hugging Face Spaces on the free CPU tier. Inference runs through ONNX Runtime
rather than PyTorch -- see ``serving.py`` for why the whole demo path is torch-free.

The confidences shown are temperature-calibrated. That is deliberate: an uncalibrated
196-class softmax reads "99%" on cars the model cannot reliably identify, and a
confidence number that cannot be trusted is worse than showing none at all.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from serving import TOP_K, Predictor

TITLE = "carvision: fine-grained car recognition"

DESCRIPTION = """
Identifies the **make, model and year** of a car from a photograph, across the 196
classes of the Stanford Cars dataset.

The model is a frozen self-supervised backbone with a classifier head trained on cached
embeddings -- an architecture chosen because the whole project was built without a GPU.
Confidences are temperature-calibrated, so a stated 80% means roughly 80%.

Fine-grained recognition is hard: telling a 2012 from a 2007 of the same model comes down
to a bumper and a badge. Where the model is unsure, the top-5 list usually still contains
the right answer.
"""

#: On Spaces this is fetched from a model repo at startup; locally it comes from
#: `make export`.
MODEL_DIR = Path(os.environ.get("CARVISION_MODEL_DIR", "artifacts/serving"))


def example_images() -> list[str]:
    """Return paths to the bundled example images, if any are present."""
    directory = Path(__file__).parent / "examples"
    if not directory.is_dir():
        return []
    return sorted(str(path) for path in directory.glob("*.jpg"))


def footer_text(predictor: Predictor) -> str:
    """Render the results line shown under the demo, from the bundle's own metrics."""
    metrics = predictor.config.get("metrics", {})
    return (
        f"Test top-1 **{100 * metrics.get('top1', 0):.1f}%** "
        f"(95% CI {100 * metrics.get('top1_low', 0):.1f}-"
        f"{100 * metrics.get('top1_high', 0):.1f}), "
        f"top-5 **{100 * metrics.get('top5', 0):.1f}%**, on the official Stanford Cars "
        f"test split of {metrics.get('num_samples', 0):,} images. "
        f"Backbone: `{predictor.backbone}` (frozen). "
        f"[Source](https://github.com/KingCobra52/cnn_for_cars)"
    )


def build_interface(model_dir: Path = MODEL_DIR) -> Any:
    """Construct the Gradio app.

    Args:
        model_dir: Directory holding the serving bundle.

    Returns:
        The Gradio Blocks app, ready to launch.
    """
    import gradio as gr

    predictor = Predictor(model_dir)

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

        gr.Markdown(footer_text(predictor))

        submit.click(predictor, inputs=image_input, outputs=label_output)
        image_input.change(predictor, inputs=image_input, outputs=label_output)

    return demo


if __name__ == "__main__":
    build_interface().launch()
