"""Preprocessing, owned by the backbone rather than by the caller.

The legacy project's most consequential bug was a mismatch between model and
preprocessing: a pretrained ImageNet VGG16 was fed 32x32 tensors normalised with
CIFAR-10 statistics. It still scored 85% because the task was trivial, which is exactly
what made the bug invisible.

The fix is structural, not a matter of care. A transform is never constructed by hand
here; it is obtained from the same weights object that supplies the pretrained
parameters, so input size and normalisation statistics cannot drift apart. Where a
backbone ships no transform (DINOv2 via torch.hub), the constants live next to that
backbone's definition and are covered by a test.
"""

from __future__ import annotations

from dataclasses import dataclass

from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: OpenAI CLIP was trained with its own statistics, not ImageNet's.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class PreprocessSpec:
    """A fully specified preprocessing pipeline.

    Keeping this as data rather than a bare callable means it can be hashed into the
    embedding cache key, so changing the resize or the normalisation statistics
    invalidates cached embeddings automatically.

    Attributes:
        resize: Shorter-side length before cropping.
        crop: Final square crop size fed to the model.
        mean: Per-channel normalisation mean.
        std: Per-channel normalisation standard deviation.
        interpolation: Resampling mode name, as understood by torchvision.
    """

    resize: int
    crop: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    interpolation: str = "bicubic"

    def cache_key(self) -> str:
        """Return a stable string identifying this preprocessing exactly."""
        return (
            f"resize={self.resize};crop={self.crop};"
            f"mean={','.join(f'{m:.6f}' for m in self.mean)};"
            f"std={','.join(f'{s:.6f}' for s in self.std)};"
            f"interp={self.interpolation}"
        )

    def _interpolation_mode(self) -> v2.InterpolationMode:
        return {
            "bicubic": v2.InterpolationMode.BICUBIC,
            "bilinear": v2.InterpolationMode.BILINEAR,
            "nearest": v2.InterpolationMode.NEAREST,
        }[self.interpolation]

    def build(self, *, train: bool = False) -> v2.Transform:
        """Construct the torchvision transform.

        Args:
            train: If true, add light augmentation (random resized crop and horizontal
                flip). Only meaningful when fine-tuning end to end; the cached-embedding
                path always uses ``train=False``, because augmenting inputs whose
                embeddings are computed once would defeat the cache.

        Returns:
            A composed transform mapping a PIL image to a normalised float tensor.
        """
        interpolation = self._interpolation_mode()

        if train:
            geometry: list[v2.Transform] = [
                v2.RandomResizedCrop(
                    self.crop, scale=(0.7, 1.0), interpolation=interpolation, antialias=True
                ),
                v2.RandomHorizontalFlip(),
            ]
        else:
            geometry = [
                v2.Resize(self.resize, interpolation=interpolation, antialias=True),
                v2.CenterCrop(self.crop),
            ]

        return v2.Compose(
            [
                *geometry,
                v2.ToImage(),
                v2.ToDtype(dtype_scale_placeholder(), scale=True),
                v2.Normalize(mean=list(self.mean), std=list(self.std)),
            ]
        )


def dtype_scale_placeholder() -> object:
    """Return the float dtype used for model input.

    Isolated in a function so the torch import stays lazy for callers that only want to
    inspect a :class:`PreprocessSpec` without loading torch.
    """
    import torch

    return torch.float32


#: Preprocessing for backbones whose weights object does not supply a transform.
IMAGENET_224 = PreprocessSpec(resize=256, crop=224, mean=IMAGENET_MEAN, std=IMAGENET_STD)
CLIP_224 = PreprocessSpec(resize=224, crop=224, mean=CLIP_MEAN, std=CLIP_STD)

#: DINOv2 uses a patch size of 14, so its input side must be a multiple of 14.
#: 224 = 16 x 14 works; 256/224 resize-then-crop matches the reference evaluation.
DINOV2_224 = PreprocessSpec(resize=256, crop=224, mean=IMAGENET_MEAN, std=IMAGENET_STD)
