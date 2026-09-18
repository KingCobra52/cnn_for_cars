"""carvision: fine-grained car classification under a CPU-only constraint.

The package is organised around one idea: running a frozen pretrained backbone
over the dataset is expensive and happens once, while training a classifier head
on the resulting embeddings is cheap and happens hundreds of times. See
``carvision.features.cache``.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
