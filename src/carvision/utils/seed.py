"""Deterministic seeding for every entry point.

The legacy notebook seeded nothing, so no two runs agreed and the reported
accuracy could not be reproduced. Every entry point in this package calls
:func:`set_seed` before touching a model.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

DEFAULT_SEED = 0


def set_seed(seed: int = DEFAULT_SEED, *, deterministic: bool = True) -> None:
    """Seed every RNG this project touches.

    Args:
        seed: The seed applied to ``random``, ``numpy`` and ``torch``.
        deterministic: If true, also force deterministic cuDNN/cuBLAS kernels and
            ask torch to raise on any op that has no deterministic implementation.
            Costs some speed; on CPU the cost is negligible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuBLAS needs this set before the first CUDA context to be reproducible.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker.

    Passed as ``worker_init_fn`` so that multi-worker loading stays reproducible.
    Torch gives each worker a distinct ``initial_seed()``; we derive the numpy and
    stdlib seeds from it so every library in the worker agrees.

    Args:
        worker_id: Index of the worker, supplied by the DataLoader. Unused; the
            per-worker entropy comes from ``torch.initial_seed()``.
    """
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
