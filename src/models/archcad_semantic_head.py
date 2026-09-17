"""Per-primitive semantic head for ArchCAD: classify each primitive, do not group it.

Motivation, measured in `docs/experiments/archcad_semantics_bottleneck_20260917.md`:

- A learning-free geometric grouper given **oracle** semantics reaches pooled PQ 0.808 on
  the 512-image validation set. The same grouper given the query/mask model's semantics
  reaches 0.170, and that model end to end reaches 0.149.
- So ~93% of the lost PQ is per-primitive semantic error, and the model produces its
  semantics as a byproduct of a mask vote: `argmax_c (class_probability_c · mask).sum`.
  When the mask head emits blobs that are not instance-aware — measured separately — that
  vote is close to arbitrary, and per-primitive accuracy lands at 22–32%.

This module drops the entanglement. It predicts one label per primitive with a plain
weighted cross-entropy, and instances are formed by the geometric grouper. The task it
solves is a per-primitive classification, which the 8-image overfit test already showed is
learnable to macro F1 0.93, and whose irreducible ambiguity floor is 0.66% of training
primitives.

The `ignore` label (raw semantic 100, remapped to 30) contributes no loss and no metric
weight, and padding positions are excluded from attention and from the loss.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.data.archcad_constants import ARCHCAD_IGNORE_LABEL, NUM_ARCHCAD_SEMANTIC_CLASSES


class ArchCADPrimitiveSemantic(nn.Module):
    """Transformer encoder over primitive tokens with a 30-way head per primitive."""

    def __init__(self, in_features=12, dim=64, heads=4, encoder_layers=2,
                 classes=NUM_ARCHCAD_SEMANTIC_CLASSES):
        super().__init__()
        self.config = dict(in_features=in_features, dim=dim, heads=heads,
                           encoder_layers=encoder_layers, classes=classes,
                           architecture='per_primitive_semantic')
        self.embedding = nn.Sequential(nn.Linear(in_features, dim), nn.GELU(), nn.LayerNorm(dim))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout=0., batch_first=True),
            encoder_layers, enable_nested_tensor=False)
        self.classifier = nn.Linear(dim, classes)

    def forward(self, features, padding_mask=None):
        if features.ndim != 3 or features.shape[2] != self.config['in_features'] or not features.shape[1]:
            raise ValueError('expected non-empty [B,N,{}] features'.format(self.config['in_features']))
        if padding_mask is not None and padding_mask.shape != features.shape[:2]:
            raise ValueError('padding mask must be [B,N]')
        encoded = self.encoder(self.embedding(features), src_key_padding_mask=padding_mask)
        return {'semantic_logits': self.classifier(encoded)}


def pad_semantic_batch(items, device='cpu'):
    """Right-pad primitive features, semantic labels and length weights for one step."""
    if not items:
        raise ValueError('empty batch')
    sizes = [len(item['features']) for item in items]
    if min(sizes) <= 0:
        raise ValueError('every image must carry at least one primitive')
    widths = {int(item['features'].shape[1]) for item in items}
    if len(widths) != 1:
        raise ValueError('all items in a batch must share the feature width')
    feature_width = widths.pop()
    width, batch = max(sizes), len(items)
    features = torch.zeros(batch, width, feature_width, dtype=torch.float32)
    labels = torch.full((batch, width), ARCHCAD_IGNORE_LABEL, dtype=torch.long)
    weights = torch.zeros(batch, width, dtype=torch.float32)
    padding_mask = torch.ones(batch, width, dtype=torch.bool)
    for index, item in enumerate(items):
        n = sizes[index]
        features[index, :n] = item['features']
        labels[index, :n] = item['targets']['semantic_labels'].long()
        weights[index, :n] = item['targets']['length_weights'].float()
        padding_mask[index, :n] = False
    return {'features': features.to(device), 'labels': labels.to(device),
            'weights': weights.to(device), 'padding_mask': padding_mask.to(device), 'sizes': sizes}


def semantic_classification_loss(output, batch):
    """Length-weighted cross-entropy over non-ignore primitives.

    Weighting by the same `round(log1p(length), 3)` used by the evaluator aligns the
    training objective with the length-weighted semantic metric.
    """
    logits = output['semantic_logits'].reshape(-1, output['semantic_logits'].shape[-1])
    labels = batch['labels'].reshape(-1)
    weights = batch['weights'].reshape(-1)
    valid = labels != ARCHCAD_IGNORE_LABEL
    if not bool(valid.any()):
        return logits.sum() * 0
    losses = F.cross_entropy(logits[valid], labels[valid], reduction='none')
    kept = weights[valid]
    return (losses * kept).sum() / kept.sum().clamp_min(1e-6)
