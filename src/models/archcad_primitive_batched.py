"""Batched primitive-only panoptic baseline: same model, many images per step.

Measured on this machine, one training step costs ~10.5 ms of wall clock while the
arithmetic is ~1.7 us (0.016% of the step). The step is host/launch bound, so the
only way to use the GPU is to make each step carry more images.

Design constraints that keep this comparable to the frozen pilot:

- `ArchCADPrimitiveBatched` subclasses `ArchCADPrimitiveBaseline` and only overrides
  `forward`, so the parameter set and `state_dict` keys are identical and weights
  transfer between the two in either direction.
- The per-image loss is the frozen `primitive_mask_loss`, applied to a slice of the
  batched output. The loss therefore changes nothing about what is optimized; only
  the number of images that share one optimizer step changes.
- Padding is right-aligned, so a slice `[:, :n]` of the batched mask logits is
  exactly the unpadded column set. The frozen decoder and evaluator consume that
  slice unchanged, which keeps every reported metric on the frozen path.

Padded positions are excluded from attention with masks. They are still embedded and
encoded (their outputs are never read), and LayerNorm/FFN are per-token, so real
tokens cannot be influenced by pad content.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from src.models.archcad_primitive_baseline import ArchCADPrimitiveBaseline, primitive_mask_loss


class ArchCADPrimitiveBatched(ArchCADPrimitiveBaseline):
    """`ArchCADPrimitiveBaseline` forward for `[B, N, 12]` inputs with padding."""

    def forward(self, features, padding_mask=None):
        if features.ndim != 3 or features.shape[2] != 12 or not features.shape[1]:
            raise ValueError('expected non-empty [B,N,12] primitive features')
        if padding_mask is not None and padding_mask.shape != features.shape[:2]:
            raise ValueError('padding mask must be [B,N]')
        encoded = self.encoder(self.embedding(features), src_key_padding_mask=padding_mask)
        queries = self.queries[None].expand(features.shape[0], -1, -1)
        decoded = self.decoder(queries, encoded, memory_key_padding_mask=padding_mask)
        return {'class_logits': self.classes(decoded),
                'mask_logits': self.mask_queries(decoded) @ self.mask_features(encoded).transpose(1, 2)
                / math.sqrt(decoded.shape[-1])}


def pad_primitive_batch(items, device='cpu'):
    """Right-pad primitive features and per-segment masks for one optimizer step.

    Returns batched tensors plus `sizes` (real primitive counts) and `padding_mask`
    (True marks a pad position).
    """
    if not items:
        raise ValueError('empty batch')
    sizes = [len(item['features']) for item in items]
    if min(sizes) <= 0:
        raise ValueError('every image must carry at least one primitive')
    width = max(sizes)
    batch = len(items)
    features = torch.zeros(batch, width, 12, dtype=torch.float32)
    padding_mask = torch.ones(batch, width, dtype=torch.bool)
    target_masks, labels, widths = [], [], []
    for index, item in enumerate(items):
        n = sizes[index]
        features[index, :n] = item['features']
        padding_mask[index, :n] = False
        target_masks.append(item['targets']['masks'])
        labels.append(item['targets']['labels'])
        widths.append(int(item['targets']['masks'].shape[1]))
    if widths != sizes:
        raise ValueError('target masks must align with primitive order and count')
    segments = max(int(mask.shape[0]) for mask in target_masks)
    masks = torch.zeros(batch, segments, width, dtype=torch.bool)
    for index, mask in enumerate(target_masks):
        masks[index, :mask.shape[0], :sizes[index]] = mask
    return {'features': features.to(device), 'padding_mask': padding_mask.to(device),
            'masks': masks.to(device), 'labels': labels, 'sizes': sizes}


def batched_primitive_mask_loss(output, batch):
    """Mean of the frozen per-image loss over the batch.

    The frozen loss is used per image so that batching cannot change what each
    image contributes; only the optimizer-step granularity differs. The segment
    axis is padded to the batch maximum, so each image is sliced back to its own
    segment and primitive counts before the frozen alignment check runs.
    """
    losses = []
    for index, size in enumerate(batch['sizes']):
        label_count = len(batch['labels'][index])
        losses.append(primitive_mask_loss(
            {'class_logits': output['class_logits'][index],
             'mask_logits': output['mask_logits'][index][:, :size]},
            {'labels': batch['labels'][index],
             'masks': batch['masks'][index][:label_count, :size]})['loss'])
    return torch.stack(losses).mean()


def batch_items(items, size):
    """Chunk a list of per-image items into batches of at most `size`."""
    if size <= 0:
        raise ValueError('batch size must be positive')
    return [items[start:start + size] for start in range(0, len(items), size)]


def padded_fraction(items):
    """Share of primitive slots in a batch that are padding rather than data."""
    sizes = [len(item['features']) for item in items]
    width = max(sizes)
    return 1 - sum(sizes) / (width * len(sizes))
