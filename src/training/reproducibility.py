"""Deterministic seeding and RNG state capture helpers."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch without starting any computation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker from PyTorch's worker-specific initial seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def capture_rng_state(
    data_generator: torch.Generator | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    if data_generator is not None:
        state["data_generator"] = data_generator.get_state()
    return state


def restore_rng_state(
    state: dict[str, Any],
    data_generator: torch.Generator | None = None,
) -> None:
    required = {"python", "numpy", "torch_cpu"}
    missing = required.difference(state)
    if missing:
        raise ValueError(f"RNG state is incomplete; missing {sorted(missing)}")

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])

    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if data_generator is not None and "data_generator" in state:
        data_generator.set_state(state["data_generator"])
