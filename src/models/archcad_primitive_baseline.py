"""Compact primitive-only query/mask baseline for the public-40K pilot.

This baseline establishes a trainable control. It is not the pretrained
HRNet/PointTransformerV2 DPSS architecture.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

from src.data.archcad_constants import ARCHCAD_STUFF_SEMANTIC_IDS


class ArchCADPrimitiveBaseline(nn.Module):
    def __init__(self, dim=64, num_queries=64, heads=4, encoder_layers=1, decoder_layers=2):
        super().__init__()
        self.config = dict(dim=dim, num_queries=num_queries, heads=heads,
                           encoder_layers=encoder_layers, decoder_layers=decoder_layers)
        self.embedding = nn.Sequential(nn.Linear(12, dim), nn.GELU(), nn.LayerNorm(dim))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout=0., batch_first=True),
            encoder_layers, enable_nested_tensor=False)
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * .02)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(dim, heads, dim * 2, dropout=0., batch_first=True), decoder_layers)
        self.classes = nn.Linear(dim, 31)
        self.mask_queries = nn.Linear(dim, dim)
        self.mask_features = nn.Linear(dim, dim)

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != 12 or not len(features):
            raise ValueError('expected non-empty [N,12] primitive features')
        encoded = self.encoder(self.embedding(features)[None])
        decoded = self.decoder(self.queries[None], encoded)[0]
        return {'class_logits': self.classes(decoded),
                'mask_logits': self.mask_queries(decoded) @ self.mask_features(encoded[0]).T / math.sqrt(decoded.shape[-1])}


def primitive_mask_loss(output, target):
    """Hungarian class/BCE/Dice matching; input masks are [K,N]."""
    classes, masks = output['class_logits'], output['mask_logits']
    labels = target['labels'].to(classes.device)
    truth = target['masks'].to(masks.device, dtype=masks.dtype)
    q, n = masks.shape
    if len(labels) > q:
        raise ValueError('more target segments than queries; increase num_queries')
    if truth.shape != (len(labels), n):
        raise ValueError('target masks must align with class labels and primitive order')
    probabilities = masks.sigmoid()
    with torch.no_grad():
        bce_cost = F.softplus(masks).mean(1)[:, None] - masks @ truth.T / n
        dice_cost = 1 - (2 * probabilities @ truth.T + 1) / (probabilities.sum(1)[:, None] + truth.sum(1)[None] + 1)
        cost = -2 * classes.softmax(-1)[:, labels] + 7 * bce_cost + 5 * dice_cost
        rows, columns = linear_sum_assignment(cost.cpu().numpy())
        rows = torch.as_tensor(rows, device=classes.device)
        columns = torch.as_tensor(columns, device=classes.device)
    class_targets = torch.full((q,), 30, dtype=torch.long, device=classes.device)
    class_targets[rows] = labels[columns]
    weights = classes.new_ones(31)
    weights[-1] = .05
    classification = F.cross_entropy(classes, class_targets, weight=weights)
    if len(rows):
        matched, expected = masks[rows], truth[columns]
        bce = F.binary_cross_entropy_with_logits(matched, expected)
        p = matched.sigmoid()
        dice = (1 - (2 * (p * expected).sum(1) + 1) / (p.sum(1) + expected.sum(1) + 1)).mean()
    else:
        bce = dice = masks.sum() * 0
    return {'loss': 2 * classification + 7 * bce + 5 * dice,
            'class_ce': classification, 'mask_bce': bce, 'mask_dice': dice}


@torch.no_grad()
def decode_primitive_masks(output, image_id, score_threshold=.1, mask_threshold=.5,
                           consistent_semantic_thresholds=False):
    """Emit semantic labels and disjoint panoptic masks with merged stuff.

    The instance path keeps only queries that pass the no-object and score gates,
    with a binarised mask. The semantic map does not, by default: it lets every
    query vote with its raw mask probability. `consistent_semantic_thresholds`
    applies the same gates to that vote and marks primitives with no surviving
    vote as ignore (30). It is opt-in because the default is the frozen decoder of
    the 2026-09-16/17 runs, and it can only move the semantic F1/wF1: instances
    are built from query class scores, never from the semantic map.
    """
    cls = output['class_logits'].softmax(-1)
    masks = output['mask_logits'].sigmoid()
    scores, labels = cls.max(-1)
    keep = (labels != 30) & (scores >= score_threshold)
    if consistent_semantic_thresholds:
        votes = cls[keep][:, :30].T @ (masks[keep] >= mask_threshold).to(cls.dtype)
        semantic = votes.argmax(0).cpu().numpy()
        semantic[votes.sum(0).cpu().numpy() == 0] = 30
    else:
        semantic = (cls[:, :30].T @ masks).argmax(0).cpu().numpy()
    scores, labels, masks = scores[keep], labels[keep], masks[keep]
    instances = []
    if len(scores):
        assigned = (scores[:, None] * masks).argmax(0)
        stuff = {}
        for i, (score, label, mask) in enumerate(zip(scores, labels, masks)):
            selected = ((assigned == i) & (mask >= mask_threshold)).cpu().numpy()
            if not selected.any():
                continue
            label = int(label)
            if label in ARCHCAD_STUFF_SEMANTIC_IDS and label in stuff:
                record = instances[stuff[label]]
                record['mask'] |= selected
                record['score'] = max(record['score'], float(score))
            else:
                if label in ARCHCAD_STUFF_SEMANTIC_IDS:
                    stuff[label] = len(instances)
                instances.append({'semantic_id': label, 'score': float(score), 'mask': selected})
    return {'image_id': image_id, 'semantic_labels': semantic, 'instances': instances}
