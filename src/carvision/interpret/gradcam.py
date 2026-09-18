"""Grad-CAM saliency for the convolutional path.

Grad-CAM answers "which pixels drove this prediction" by weighting a convolutional
feature map by the gradient of the target logit with respect to it. The result is coarse
-- 7x7 for a ResNet at 224px -- but it is enough to distinguish a model looking at the
grille from a model looking at the parking lot behind the car.

This is the check worth running on the confidently-wrong cases from
:mod:`carvision.interpret.errors`. A saliency map centred on background is evidence of a
spurious cue; one centred on the car means the model is simply losing a hard
fine-grained distinction, which is a different problem with a different fix.

The backbone is frozen elsewhere in this project, so gradients are enabled locally here
and restored on exit.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Iterator


class GradCAMError(RuntimeError):
    """Raised when a layer cannot be resolved or produces no usable activation."""


def resolve_layer(module: nn.Module, path: str) -> nn.Module:
    """Resolve a dotted attribute path to a submodule.

    Args:
        module: Root module.
        path: Dotted path such as ``"layer4"`` or ``"features.28"``.

    Returns:
        The submodule.

    Raises:
        GradCAMError: If any component of the path is missing.
    """
    current: nn.Module = module
    for part in path.split("."):
        if part.isdigit() and isinstance(current, nn.Sequential):
            current = current[int(part)]
            continue
        if not hasattr(current, part):
            raise GradCAMError(
                f"Cannot resolve {path!r}: {type(current).__name__} has no attribute {part!r}."
            )
        current = getattr(current, part)
    return current


@contextmanager
def _grads_enabled(module: nn.Module) -> Iterator[None]:
    """Temporarily re-enable gradients on a frozen module, restoring them after."""
    previous = {name: parameter.requires_grad for name, parameter in module.named_parameters()}
    module.requires_grad_(True)
    try:
        yield
    finally:
        for name, parameter in module.named_parameters():
            parameter.requires_grad_(previous[name])


class GradCAM:
    """Grad-CAM over one convolutional layer of a model.

    Usage is one call: construct with the model and target layer, then
    :meth:`__call__` with a preprocessed image batch.
    """

    def __init__(self, model: nn.Module, layer: str) -> None:
        """Initialise.

        Args:
            model: The model to explain. Must return ``(B, C)`` logits or ``(B, D)``
                features.
            layer: Dotted path to the convolutional layer to read activations from.
        """
        self.model = model
        self.layer_path = layer
        self.layer = resolve_layer(model, layer)

    def __call__(
        self,
        images: torch.Tensor,
        target: int | torch.Tensor | None = None,
        *,
        head: nn.Module | None = None,
    ) -> np.ndarray:
        """Produce a saliency map per image.

        Args:
            images: ``(B, 3, H, W)`` preprocessed batch.
            target: Class index to explain. An int applies to every image, a tensor gives
                one index per image, and None uses each image's own argmax.
            head: Optional classifier head applied to the backbone output, for the
                cached-embedding setup where the backbone alone emits features rather
                than class logits.

        Returns:
            ``(B, h, w)`` maps in ``[0, 1]``, where ``h`` and ``w`` are the target
            layer's spatial dimensions.

        Raises:
            GradCAMError: If the target layer produced no gradient, which means it is not
                on the path between the input and the explained output.
        """
        activations: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []

        def forward_hook(_module: nn.Module, _inputs: object, output: torch.Tensor) -> None:
            activations.append(output)
            # retain_grad is simpler and more robust than a full backward hook here,
            # which fires inconsistently across module types.
            output.retain_grad()

        handle = self.layer.register_forward_hook(forward_hook)

        try:
            with _grads_enabled(self.model), torch.enable_grad():
                features = self.model(images)
                logits = head(features) if head is not None else features

                if target is None:
                    indices = logits.argmax(dim=1)
                elif isinstance(target, int):
                    indices = torch.full((len(images),), target, dtype=torch.long)
                else:
                    indices = target.long()

                score = logits.gather(1, indices[:, None]).sum()
                self.model.zero_grad(set_to_none=True)
                score.backward()

                activation = activations[0]
                if activation.grad is None:
                    raise GradCAMError(
                        f"Layer {self.layer_path!r} received no gradient, so it is not on "
                        f"the path to the explained output. Pick a layer inside the "
                        f"forward pass."
                    )
                gradients.append(activation.grad)
        finally:
            handle.remove()

        return _to_maps(activations[0].detach(), gradients[0].detach())


def _to_maps(activation: torch.Tensor, gradient: torch.Tensor) -> np.ndarray:
    """Combine activations and gradients into normalised saliency maps.

    Args:
        activation: ``(B, C, h, w)`` feature map.
        gradient: Gradient of the target score with respect to ``activation``.

    Returns:
        ``(B, h, w)`` maps, each scaled to ``[0, 1]``.
    """
    # Channel importance is the spatially averaged gradient (Grad-CAM's key move).
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activation).sum(dim=1)

    # ReLU: only evidence *for* the class is of interest, not against it.
    cam = torch.relu(cam)

    flat = cam.flatten(1)
    minimum = flat.min(dim=1).values[:, None, None]
    maximum = flat.max(dim=1).values[:, None, None]
    # An all-zero map (no positive evidence) would divide by zero; leave it at zero.
    scale = (maximum - minimum).clamp(min=1e-8)
    return ((cam - minimum) / scale).cpu().numpy()


def overlay(
    image: np.ndarray,
    cam: np.ndarray,
    *,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """Blend a saliency map over an image for display.

    Args:
        image: ``(H, W, 3)`` uint8 or float image in ``[0, 1]``.
        cam: ``(h, w)`` saliency map in ``[0, 1]``; upsampled to the image size.
        alpha: Weight of the heatmap in the blend.
        colormap: Matplotlib colormap name.

    Returns:
        An ``(H, W, 3)`` uint8 overlay.
    """
    import matplotlib.cm as cm
    from PIL import Image

    if image.dtype != np.uint8:
        image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)

    height, width = image.shape[:2]
    resized = np.asarray(
        Image.fromarray((cam * 255).astype(np.uint8)).resize((width, height), Image.BILINEAR)
    )

    heatmap = (cm.get_cmap(colormap)(resized / 255.0)[..., :3] * 255).astype(np.uint8)
    blended = (1 - alpha) * image.astype(np.float32) + alpha * heatmap.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)
