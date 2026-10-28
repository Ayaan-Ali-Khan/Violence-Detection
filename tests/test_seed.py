"""Tests for experiment seeding utilities."""
import random
import numpy as np
import torch
from violence_detection.utils.seed import make_generator, set_seed


def test_set_seed_repeats_python_numpy_and_torch_random_values() -> None:
    set_seed(123)
    first = (random.random(), float(np.random.rand()), float(torch.rand(1).item()))

    set_seed(123)
    second = (random.random(), float(np.random.rand()), float(torch.rand(1).item()))

    assert first == second


def test_make_generator_is_reproducible() -> None:
    first = torch.rand(5, generator=make_generator(17))
    second = torch.rand(5, generator=make_generator(17))

    assert torch.equal(first, second)