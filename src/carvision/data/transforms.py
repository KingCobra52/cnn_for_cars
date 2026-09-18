"""Preprocessing, owned by the backbone rather than by the caller.

The legacy project's most consequential bug was a mismatch between model and
preprocessing: a pretrained ImageNet VGG16 was fed 32x32 tensors normalised with
CIFAR-10 statistics. It still scored 85% because the task was trivial, which is exactly
what made the bug invisible.

The fix is structural, not a matter of care: a :class:`PreprocessSpec` is bound to a
backbone's weights in the registry, so selecting a backbone selects its preprocessing
and the two cannot drift apart at a call site.

The specs are written out as constants rather than read from the weights object at
runtime, because they are hashed into the embedding cache key and so must be stable,
inspectable values rather than whatever the installed torchvision happens to return.
That makes them a claim about the weights, and a claim needs checking: for every
torchvision backbone, ``tests/test_transforms.py`` asserts the constants here match
``Weights.transforms()`` exactly, so a torchvision change or a transcription error fails
CI instead of quietly costing accuracy.

That check earned itself immediately. ``resnet50`` was configured at 256/bicubic; the
IMAGENET1K_V2 weights actually want 232/bilinear.
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


#: Generic ImageNet preprocessing: the classic 256/224 crop, bilinear.
#: Matches ``ResNet50_Weights.IMAGENET1K_V1.transforms()``.
IMAGENET_224 = PreprocessSpec(
    resize=256, crop=224, mean=IMAGENET_MEAN, std=IMAGENET_STD, interpolation="bilinear"
)

#: Preprocessing for ``ResNet50_Weights.IMAGENET1K_V2``. The V2 weights were trained with
#: a different recipe and want a 232-pixel shorter side, not 256. Feeding them the V1
#: transform costs roughly a point of ImageNet top-1, and correspondingly degrades the
#: features this project extracts from them.
RESNET50_V2 = PreprocessSpec(
    resize=232, crop=224, mean=IMAGENET_MEAN, std=IMAGENET_STD, interpolation="bilinear"
)
CLIP_224 = PreprocessSpec(resize=224, crop=224, mean=CLIP_MEAN, std=CLIP_STD)

#: DINOv2 uses a patch size of 14, so its input side must be a multiple of 14.
#: 224 = 16 x 14 works; 256/224 resize-then-crop matches the reference evaluation.
DINOV2_224 = PreprocessSpec(resize=256, crop=224, mean=IMAGENET_MEAN, std=IMAGENET_STD)
