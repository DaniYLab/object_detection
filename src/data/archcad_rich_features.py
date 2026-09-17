"""Scale-aware input features for the ArchCAD primitive baseline.

Motivation (measured, see `docs/experiments/archcad_failure_localization_20260917.md`):

- The 12-dimensional geometry-only input carries real instance signal, but a scale-blind
  pairwise rule caps at recall 0.44 because instances range from one primitive to dozens.
- Missed instances are the ones packed against a different instance of the same class
  (median feature distance 0.285 versus 0.540 for covered instances), and coverage falls
  from 42.7% to 25.4% as a class repeats within a drawing.
- Allowing an oracle radius per instance, 57.6% of instances are separable; the model
  covers only 39.5% of those, so scale-aware input has headroom for about a third of the
  problem.

The added dimensions give the model the *local packing scale* and the primitive's size
relative to it, which a fixed distance threshold cannot express:

  polyline length   length of the 4-point polyline, a size cue for every primitive type
  extent            largest pairwise distance among the 4 points
  d1, d4, d8        distance to the nearest / mean of 4 nearest / mean of 8 nearest
                    primitive centres - the local packing scale
  size ratio        log1p(polyline length / d4), the scale-normalised size

A structural cue such as junction degree is deliberately left out of this version: it
needs endpoint geometry, which the cached item does not carry. It is the natural next
addition if this one moves validation at all.

Nothing here changes the frozen dataset, model or decoder; the class is new and the
augmentation is a pure function of an already-cached item.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from src.models.archcad_primitive_batched import ArchCADPrimitiveBatched

FEATURE_NAMES = ('polyline_length', 'extent', 'd1', 'd4', 'd8', 'size_ratio')
RICH_WIDTH = 12 + len(FEATURE_NAMES)


def rich_features(item):
    """Return `[N, RICH_WIDTH]` features: the frozen 12 plus the scale extras."""
    base = item['features']
    if base.ndim != 2 or base.shape[1] != 12:
        raise ValueError('expected the frozen [N,12] features')
    points = base[:, :8].reshape(-1, 4, 2)
    centres = item['centers']
    if centres.shape[0] != points.shape[0]:
        raise ValueError('centres must align with primitives')
    steps = (points[:, 1:] - points[:, :-1]).norm(dim=-1)
    polyline = steps.sum(dim=1)
    extent = torch.cdist(points, points).amax(dim=(1, 2))
    # Local packing scale over centre distances; the self-distance is excluded.
    distances = torch.cdist(centres, centres)
    n = distances.shape[0]
    if n > 1:
        distances = distances + torch.eye(n, dtype=distances.dtype) * 1e6
        ordered = distances.sort(dim=1).values
        d1 = ordered[:, 0]
        d4 = ordered[:, :min(4, n - 1)].mean(dim=1)
        d8 = ordered[:, :min(8, n - 1)].mean(dim=1)
    else:
        zero = torch.zeros(1, dtype=distances.dtype)
        d1 = d4 = d8 = zero
    ratio = torch.log1p(polyline / d4.clamp_min(1e-6))
    return torch.cat([base, polyline[:, None], extent[:, None], d1[:, None], d4[:, None],
                      d8[:, None], ratio[:, None]], dim=1).float()


def augment_items(items):
    """Replace each item's features with the rich version; returns the items."""
    for item in items:
        item['features'] = rich_features(item)
    return items


class ArchCADPrimitiveRich(ArchCADPrimitiveBatched):
    """`ArchCADPrimitiveBatched` with the wider input layer."""

    def __init__(self, dim=64, num_queries=64, heads=4, encoder_layers=1, decoder_layers=2):
        super().__init__(dim=dim, num_queries=num_queries, heads=heads,
                         encoder_layers=encoder_layers, decoder_layers=decoder_layers)
        self.embedding = nn.Sequential(nn.Linear(RICH_WIDTH, dim), nn.GELU(), nn.LayerNorm(dim))
        # Recorded in run config so a checkpoint states its own input contract.
        self.config['input_features'] = RICH_WIDTH

    def forward(self, features, padding_mask=None):
        if features.ndim != 3 or features.shape[2] != RICH_WIDTH or not features.shape[1]:
            raise ValueError('expected non-empty [B,N,{}] rich primitive features'.format(RICH_WIDTH))
        if padding_mask is not None and padding_mask.shape != features.shape[:2]:
            raise ValueError('padding mask must be [B,N]')
        encoded = self.encoder(self.embedding(features), src_key_padding_mask=padding_mask)
        queries = self.queries[None].expand(features.shape[0], -1, -1)
        decoded = self.decoder(queries, encoded, memory_key_padding_mask=padding_mask)
        return {'class_logits': self.classes(decoded),
                'mask_logits': self.mask_queries(decoded) @ self.mask_features(encoded).transpose(1, 2)
                / math.sqrt(decoded.shape[-1])}
