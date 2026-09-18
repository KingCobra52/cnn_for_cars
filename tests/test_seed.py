"""Seeding must actually make runs reproducible -- the legacy project's core failure."""

from __future__ import annotations

import random

import numpy as np
import torch

from carvision.utils.seed import set_seed


def _sample() -> tuple[float, float, float]:
    return random.random(), float(np.random.rand()), float(torch.rand(1).item())


def test_same_seed_gives_identical_draws() -> None:
    set_seed(7)
    first = _sample()
    set_seed(7)
    second = _sample()
    assert first == second


def test_different_seeds_diverge() -> None:
    set_seed(7)
    first = _sample()
    set_seed(8)
    second = _sample()
    assert first != second


def test_model_init_is_reproducible() -> None:
    set_seed(3)
    a = torch.nn.Linear(32, 8).weight.detach().clone()
    set_seed(3)
    b = torch.nn.Linear(32, 8).weight.detach().clone()
    torch.testing.assert_close(a, b)
