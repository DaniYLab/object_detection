"""Stroke tokenization for the dual-pathway (image + vector) architecture.

Every FloorPlanCAD SVG path element contains exactly one primitive — an
``M → L`` line or an ``M → A`` arc (verified over all 15,663 drawings in
``docs/stroke_audit.md``). This module converts those primitives into the
fixed 12-dim superset layout the vector encoder consumes:

    [x0, y0, x1, y1, cx, cy, r, cos(t0), sin(t0), cos(t1), sin(t1), large]

- dims 0–3: primitive endpoints, shared by both types (the token backbone).
- Line: dims 4–11 are zero.
- Arc: center/radius plus start/end angles encoded as ``(cos, sin)`` pairs so
  that 359.9° and 0.1° stay numerically close; the large-arc flag is dim 11.
- Coordinates are stored in pixels and normalized to ``[0, 1]`` at load time
  by :func:`normalize_strokes`, matching the raster branch's resize frame.

Type ids (0 = line, 1 = arc) feed a learned TypeEmbedding in the model; they
are never mixed into the projected 12-dim feature itself.
"""

from __future__ import annotations

import math
import random
from typing import Any, Mapping, Sequence

import torch

STROKE_FEATURE_DIM = 12
LINE_TYPE_ID = 0
ARC_TYPE_ID = 1


def normalize_strokes(
    strokes: Sequence[Sequence[float]],
    image_size: Sequence[int],
) -> torch.Tensor:
    """Convert stored pixel-space strokes into normalized ``[N, 12]`` tokens.

    Endpoint and center coordinates are divided by ``(width, height)``; the
    radius is divided by ``sqrt(width * height)`` so it stays in a comparable
    range for non-square images. Angle features are already unit-free.
    """

    if len(image_size) != 2:
        raise ValueError("image_size must be (width, height)")
    width, height = float(image_size[0]), float(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if not strokes:
        return torch.zeros((0, STROKE_FEATURE_DIM), dtype=torch.float32)
    tensor = torch.as_tensor(strokes, dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[1] != STROKE_FEATURE_DIM:
        raise ValueError(
            f"strokes must have shape [N, {STROKE_FEATURE_DIM}], got {tuple(tensor.shape)}"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("strokes contain non-finite values")
    scale = torch.tensor(
        [width, height, width, height, width, height, math.sqrt(width * height)]
        + [1.0] * 5,
        dtype=torch.float32,
    )
    return tensor / scale


def stroke_type_ids(strokes: Sequence[Sequence[float]]) -> torch.Tensor:
    """Return ``[N]`` long tensor of type ids (0=line, 1=arc).

    An arc is identified by a non-zero radius slot (dim 6); lines store zeros
    in every arc slot by construction.
    """

    if not strokes:
        return torch.zeros((0,), dtype=torch.long)
    tensor = torch.as_tensor(strokes, dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[1] != STROKE_FEATURE_DIM:
        raise ValueError(
            f"strokes must have shape [N, {STROKE_FEATURE_DIM}], got {tuple(tensor.shape)}"
        )
    return (tensor[:, 6] != 0).to(dtype=torch.long)


def sample_strokes(
    tokens: torch.Tensor,
    n_max: int,
    generator: random.Random | None = None,
) -> torch.Tensor:
    """Randomly subsample stroke tokens to at most ``n_max`` rows.

    Drawings above ``n_max`` primitives are uniformly subsampled without
    replacement — each epoch sees a different subset of the same drawing,
    which acts as free augmentation and avoids a deterministic information
    cut. Deterministic callers pass a seeded ``random.Random``.
    """

    if n_max <= 0:
        raise ValueError("n_max must be positive")
    if tokens.ndim != 2:
        raise ValueError(f"tokens must have shape [N, D], got {tuple(tokens.shape)}")
    count = tokens.shape[0]
    if count <= n_max:
        return tokens
    rng = generator or random
    keep = sorted(rng.sample(range(count), n_max))
    return tokens.index_select(0, torch.tensor(keep, dtype=torch.long))


def pad_stroke_batch(
    token_list: Sequence[torch.Tensor],
    n_max: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of ``[N_i, 12]`` token tensors into a batch.

    Returns ``(tokens, valid_mask)`` where tokens has shape ``[B, n_max, 12]``
    and ``valid_mask`` has shape ``[B, n_max]`` with ``True`` marking real
    (non-pad) positions — the padding mask consumed by attention. Samples
    exceeding ``n_max`` are subsampled first; empty samples produce a single
    padding-only row so the mask's any() guard in fusion stays well-defined.
    """

    if n_max <= 0:
        raise ValueError("n_max must be positive")
    batch = len(token_list)
    tokens_out = torch.zeros((batch, n_max, STROKE_FEATURE_DIM), dtype=torch.float32)
    valid_out = torch.zeros((batch, n_max), dtype=torch.bool)
    for index, tokens in enumerate(token_list):
        if tokens.numel() == 0:
            continue
        if tokens.ndim != 2 or tokens.shape[1] != STROKE_FEATURE_DIM:
            raise ValueError(
                f"tokens must have shape [N, {STROKE_FEATURE_DIM}], got {tuple(tokens.shape)}"
            )
        count = min(tokens.shape[0], n_max)
        if count < tokens.shape[0]:
            keep = sorted(random.sample(range(tokens.shape[0]), count))
            tokens = tokens.index_select(0, torch.tensor(keep, dtype=torch.long))
        tokens_out[index, :count] = tokens[:count]
        valid_out[index, :count] = True
    return tokens_out, valid_out


def metadata_strokes(metadata: Mapping[str, Any]) -> list[list[float]]:
    """Read the ``strokes`` array from schema-v3 metadata (empty for v2)."""

    strokes = metadata.get("strokes")
    if not isinstance(strokes, list):
        return []
    result: list[list[float]] = []
    for stroke in strokes:
        if isinstance(stroke, (list, tuple)) and len(stroke) == STROKE_FEATURE_DIM:
            result.append([float(value) for value in stroke])
    return result
