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


def transform_normalized_strokes(
    tokens: torch.Tensor,
    coordinate_matrix: torch.Tensor,
    input_size: Sequence[int],
    output_size: Sequence[int],
) -> torch.Tensor:
    """Apply an image transform's affine map to normalized stroke tokens.

    Endpoints, arc centres, radii, and arc direction vectors are updated so the
    vector pathway stays aligned with the augmented raster image. Coordinates
    outside a crop remain outside ``[0, 1]`` intentionally; this retains lines
    crossing a crop boundary instead of inventing clipped SVG primitives.
    """

    if tokens.ndim != 2 or tokens.shape[1] != STROKE_FEATURE_DIM:
        raise ValueError(
            f"tokens must have shape [N, {STROKE_FEATURE_DIM}], got {tuple(tokens.shape)}"
        )
    if tokens.numel() == 0:
        return tokens.clone()
    if coordinate_matrix.shape != (3, 3):
        raise ValueError("coordinate_matrix must have shape [3, 3]")
    in_w, in_h = float(input_size[0]), float(input_size[1])
    out_w, out_h = float(output_size[0]), float(output_size[1])
    if min(in_w, in_h, out_w, out_h) <= 0:
        raise ValueError("input and output dimensions must be positive")

    result = tokens.clone()
    matrix = coordinate_matrix.to(dtype=result.dtype, device=result.device)
    arcs = result[:, 6] != 0

    def mapped_points(values: torch.Tensor) -> torch.Tensor:
        pixels = values * result.new_tensor([in_w, in_h])
        homogeneous = torch.cat(
            [
                pixels,
                torch.ones(
                    (pixels.shape[0], 1),
                    dtype=result.dtype,
                    device=result.device,
                ),
            ],
            dim=1,
        )
        return (homogeneous @ matrix.T)[:, :2] / result.new_tensor([out_w, out_h])

    result[:, 0:2] = mapped_points(result[:, 0:2])
    result[:, 2:4] = mapped_points(result[:, 2:4])
    if bool(arcs.any()):
        result[arcs, 4:6] = mapped_points(result[arcs, 4:6])
        linear = matrix[:2, :2]
        radius_scale = torch.sqrt(torch.abs(torch.linalg.det(linear)))
        result[arcs, 6] *= (
            math.sqrt(in_w * in_h) * radius_scale / math.sqrt(out_w * out_h)
        )
        for start in (7, 9):
            directions = result[arcs, start : start + 2] @ linear.T
            directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
            result[arcs, start : start + 2] = directions
    return result


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
    """Subsample stroke tokens to at most ``n_max`` rows, length-weighted.

    Drawings above ``n_max`` primitives are subsampled without replacement with
    probability proportional to the stroke's endpoint diagonal — long strokes
    carry object identity (walls, symbol outlines) while short hatch/dot marks
    are noise-dense (see ``docs/vector_branch_analysis.md``). Weights get a
    floor of 2% of the max weight so short strokes are never fully excluded,
    and each epoch still sees a different subset of the same drawing (free
    augmentation). Deterministic callers pass a seeded ``random.Random``.
    """

    if n_max <= 0:
        raise ValueError("n_max must be positive")
    if tokens.ndim != 2:
        raise ValueError(f"tokens must have shape [N, D], got {tuple(tokens.shape)}")
    count = tokens.shape[0]
    if count <= n_max:
        return tokens
    rng = generator or random
    # Endpoint diagonal in normalized coords: dims 0-3 are (x0, y0, x1, y1).
    dx = (tokens[:, 2] - tokens[:, 0]).abs()
    dy = (tokens[:, 3] - tokens[:, 1]).abs()
    lengths = torch.sqrt(dx * dx + dy * dy)
    max_length = float(lengths.max()) if count else 1.0
    if max_length <= 0:
        keep = sorted(rng.sample(range(count), n_max))
        return tokens.index_select(0, torch.tensor(keep, dtype=torch.long))
    # Floor keeps short strokes in the lottery; arcs with tiny endpoint
    # distance (near-full circles) get the radius as a length proxy.
    radius = tokens[:, 6]
    effective = torch.maximum(lengths, radius)
    weights = (effective / max_length).clamp(min=0.02).tolist()
    keep = sorted(rng.choices(range(count), weights=weights, k=n_max))
    # rng.choices samples with replacement — deduplicate then fill from the rest.
    unique_keep = list(dict.fromkeys(keep))
    if len(unique_keep) < n_max:
        remaining = sorted(set(range(count)) - set(unique_keep))
        rng.shuffle(remaining)
        unique_keep.extend(remaining[: n_max - len(unique_keep)])
    return tokens.index_select(0, torch.tensor(sorted(unique_keep), dtype=torch.long))


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
