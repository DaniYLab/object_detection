from __future__ import annotations

import random

import numpy as np
import torch

from src.training.reproducibility import (
    capture_rng_state,
    make_generator,
    restore_rng_state,
    seed_everything,
)


def test_rng_state_round_trip_including_data_generator() -> None:
    seed_everything(1337)
    generator = make_generator(7331)
    state = capture_rng_state(generator)

    expected = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
        torch.rand(3, generator=generator),
    )

    restore_rng_state(state, generator)
    actual = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
        torch.rand(3, generator=generator),
    )

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
    assert torch.equal(actual[3], expected[3])


def test_seed_everything_repeats_cpu_initialization() -> None:
    seed_everything(42)
    first = torch.nn.Linear(4, 2).weight.detach().clone()
    seed_everything(42)
    second = torch.nn.Linear(4, 2).weight.detach().clone()

    assert torch.equal(first, second)
