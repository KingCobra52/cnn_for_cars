"""CLIP zero-shot classification: a baseline that trains nothing.

CLIP was trained to align images with their captions, so a class can be specified by
writing its name into a sentence rather than by fitting weights. Encoding 196 prompts
costs one forward pass through the text tower; after that, classification is a dot
product against cached image embeddings.

This is worth having for two reasons. It is a floor that required no training data at
all, which makes the trained probe's improvement legible. And on Stanford Cars it is a
surprisingly strong floor, because class names like ``2012 Tesla Model S Sedan`` carry
real information that a randomly initialised head has to learn from forty images.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import torch

from carvision.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger(__name__)

#: Prompt ensemble. Averaging several phrasings is consistently worth a point or two
#: over any single template, because it averages away the quirks of each wording.
PROMPT_TEMPLATES: tuple[str, ...] = (
    "a photo of a {}.",
    "a photo of the {}.",
    "a photo of my {}.",
    "a photo of a {}, a type of car.",
    "a close-up photo of a {}.",
    "a bright photo of a {}.",
    "a cropped photo of a {}.",
    "a photo of a {} on the road.",
)

#: Stanford Cars class names look like "2012 Tesla Model S Sedan": make, model, body
#: style, and a year that ends the name.
_YEAR = re.compile(r"\b(19|20)\d{2}\b")


def parse_class_name(name: str) -> dict[str, str]:
    """Split a Stanford Cars class name into its parts.

    Used both for prompting and, in error analysis, for grouping mistakes by make and by
    body style -- which is where the interesting structure in this dataset lives.

    Args:
        name: A class name such as ``"2012 Tesla Model S Sedan"``.

    Returns:
        A mapping with ``make``, ``model``, ``body``, ``year`` and the original ``full``
        name. Missing parts come back as empty strings.
    """
    year_match = _YEAR.search(name)
    year = year_match.group(0) if year_match else ""
    without_year = _YEAR.sub("", name).strip()

    tokens = without_year.split()
    make = tokens[0] if tokens else ""

    # Body styles that appear as the trailing token in this dataset's naming scheme.
    body_styles = {
        "sedan", "coupe", "convertible", "hatchback", "wagon", "suv", "van",
        "minivan", "cab", "crew", "extended", "regular", "supercab", "club",
    }
    body = tokens[-1] if tokens and tokens[-1].lower() in body_styles else ""

    model_tokens = tokens[1:-1] if body else tokens[1:]
    return {
        "full": name,
        "make": make,
        "model": " ".join(model_tokens),
        "body": body,
        "year": year,
    }


@torch.inference_mode()
def build_text_classifier(
    class_names: Sequence[str],
    *,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    templates: Sequence[str] = PROMPT_TEMPLATES,
) -> np.ndarray:
    """Encode the class names into a zero-shot classifier matrix.

    Args:
        class_names: The 196 class names, ordered by class id.
        model_name: open_clip model name.
        pretrained: open_clip pretrained tag.
        templates: Prompt templates; each is formatted with the class name and the
            resulting embeddings are averaged.

    Returns:
        An L2-normalised ``(num_classes, D)`` matrix. Classifying an image is a matrix
        multiply against it.
    """
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    weights = []
    for name in class_names:
        prompts = [template.format(name) for template in templates]
        embeddings = model.encode_text(tokenizer(prompts)).float()
        embeddings /= embeddings.norm(dim=-1, keepdim=True)
        # Average the ensemble, then renormalise so every class vector is unit length.
        averaged = embeddings.mean(dim=0)
        weights.append(averaged / averaged.norm())

    logger.info("Built zero-shot classifier: %d classes x %d templates", len(class_names), len(templates))
    return torch.stack(weights).numpy().astype(np.float32)


def predict(image_embeddings: np.ndarray, text_classifier: np.ndarray) -> np.ndarray:
    """Score cached image embeddings against the zero-shot classifier.

    Args:
        image_embeddings: ``(N, D)`` CLIP image embeddings, not necessarily normalised.
        text_classifier: ``(C, D)`` matrix from :func:`build_text_classifier`.

    Returns:
        ``(N, C)`` cosine similarities, usable as logits.

    Raises:
        ValueError: If the embedding widths disagree, which means the image embeddings
            came from a different backbone than the text tower.
    """
    if image_embeddings.shape[1] != text_classifier.shape[1]:
        raise ValueError(
            f"Image embeddings are {image_embeddings.shape[1]}-d but the text classifier "
            f"is {text_classifier.shape[1]}-d. Zero-shot needs both towers of the same "
            f"CLIP model; check that --backbone is the CLIP one."
        )
    normalised = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    return (normalised @ text_classifier.T).astype(np.float32)
