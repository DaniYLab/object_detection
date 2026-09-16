"""Raw-entity ArchCAD dataset for local panoptic baseline experiments.

Features and targets share an explicit entity index. This is a documented local
representation, not a bit-for-bit reconstruction of the missing DPSS loader.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.data.archcad_panoptic import parse_archcad_panoptic_annotation

REPRESENTATION = 'archcad_entity_four_points_v1'
PRIMITIVE_TYPES = ('LINE', 'ARC', 'CIRCLE', 'ELLIPSE')


def primitive_points(geometry):
    """Sample four ordered points in source (x, y) coordinates."""
    kind = geometry['type'].upper()
    if kind == 'LINE':
        start, end = np.asarray(geometry['start']), np.asarray(geometry['end'])
        result = start + np.linspace(0, 1, 4)[:, None] * (end - start)
    elif kind in {'ARC', 'CIRCLE'}:
        if kind == 'CIRCLE':
            angles = np.arange(4) * math.pi / 2
        else:
            sign = 1 if geometry.get('direction', 0) == 0 else -1
            begin, end = float(geometry['start_angle']), float(geometry['end_angle'])
            sweep = (sign * (end - begin)) % 360
            angles = np.deg2rad(begin + sign * np.linspace(0, sweep, 4))
        result = np.asarray(geometry['center']) + float(geometry['radius']) * np.column_stack((np.cos(angles), np.sin(angles)))
    elif kind == 'ELLIPSE':
        axis = np.asarray(geometry['major_axis'], dtype=float)
        other = np.array([-axis[1], axis[0]]) * float(geometry['ratio'])
        begin = float(geometry.get('start_param', 0))
        end = float(geometry.get('end_param', 2 * math.pi))
        sign = 1 if geometry.get('is_ccw', True) else -1
        full = math.isclose(abs(end - begin), 2 * math.pi, abs_tol=1e-9)
        sweep = 2 * math.pi if full else (sign * (end - begin)) % (2 * math.pi)
        angles = begin + sign * np.linspace(0, sweep, 4, endpoint=not full)
        result = np.asarray(geometry['center']) + np.cos(angles)[:, None] * axis + np.sin(angles)[:, None] * other
    else:
        raise ValueError(f'unsupported primitive type {kind!r}')
    result = np.asarray(result, dtype=np.float64)
    if result.shape != (4, 2) or not np.isfinite(result).all():
        raise ValueError('primitive points must be finite [4,2]')
    return result


class ArchCADPanopticDataset(Dataset):
    """Full PNG and all raw entities; ragged masks retained without truncation."""

    def __init__(self, data_root='data/ArchCAD', *, manifest_path=None, split='train', max_primitives=None):
        self.root = Path(data_root)
        self.manifest_path = Path(manifest_path) if manifest_path else self.root / 'splits.json'
        manifest = json.loads(self.manifest_path.read_text(encoding='utf-8'))
        if manifest.get('strategy') != 'sha256_group_partition':
            raise ValueError('expected ArchCAD public-40K sha256_group_partition manifest')
        splits = manifest['splits']
        ids = {role: splits[role]['images'] for role in ('train', 'val', 'test')}
        flat = [image_id for role in ids.values() for image_id in role]
        if len(set(flat)) != len(flat):
            raise ValueError('split IDs are duplicated or cross split boundaries')
        if any(not isinstance(i, str) or not i or Path(i).name != i or '/' in i or '\\' in i or i in {'.', '..'} for i in flat):
            raise ValueError('invalid sample ID')
        payload = {'seed': manifest['seed'], **{role: sorted(values) for role, values in ids.items()}}
        self.split_fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if self.split_fingerprint != manifest['fingerprint']:
            raise ValueError('manifest fingerprint mismatch')
        if split not in ids:
            raise ValueError('split must be train, val, or test')
        self.image_ids = ids[split]
        self.split = split
        self.max_primitives = max_primitives

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        raw = (self.root / 'json' / f'{image_id}.json').read_bytes()
        annotation = json.loads(raw)
        n = len(annotation['entities'])
        if self.max_primitives is not None and n > self.max_primitives:
            raise ValueError(f'{image_id}: {n} primitives exceeds declared budget {self.max_primitives}; no implicit truncation')
        sample = parse_archcad_panoptic_annotation(annotation, image_id=image_id)
        with Image.open(self.root / 'png' / f'{image_id}.png') as opened:
            image = np.array(opened.convert('RGB'), copy=True)
        h, w = image.shape[:2]
        points = np.stack([primitive_points(g) for g in sample.geometries]) if n else np.empty((0, 4, 2))
        points = points / np.array([w, h])
        type_ids = np.array([PRIMITIVE_TYPES.index(g['type'].upper()) for g in sample.geometries], dtype=np.int64)
        features = np.concatenate((points.reshape(n, 8), np.eye(4)[type_ids]), axis=1).astype(np.float32)
        # Geometric features contain no semantic, instance, layer or RGB priors.
        centers = points.mean(axis=1).astype(np.float32)
        masks = np.stack([s.primitive_mask for s in sample.segments]) if sample.segments else np.empty((0, n), dtype=bool)
        return {
            'image_id': image_id, 'image': torch.from_numpy(image).permute(2, 0, 1).float().div(255),
            'features': torch.from_numpy(features), 'centers': torch.from_numpy(centers),
            'entity_indices': torch.arange(n), 'primitive_types': torch.from_numpy(type_ids),
            'targets': {'semantic_labels': torch.from_numpy(sample.semantic_labels),
                        'instance_ids': torch.from_numpy(sample.instance_ids),
                        'labels': torch.tensor([s.semantic_id for s in sample.segments], dtype=torch.long),
                        'masks': torch.from_numpy(masks),
                        'lengths': torch.from_numpy(sample.primitive_lengths),
                        'length_weights': torch.from_numpy(sample.length_weights)},
            'provenance': {'representation': REPRESENTATION, 'raw_sha256': hashlib.sha256(raw).hexdigest(),
                           'split_fingerprint': self.split_fingerprint, 'split': self.split,
                           'image_size': [w, h], 'paper_input_parity': False},
            'issues': sample.issues,
        }


def archcad_panoptic_collate(batch):
    """Packed primitive features + per-image masks (no cross-image instances)."""
    if not batch:
        raise ValueError('empty batch')
    if len({tuple(s['image'].shape) for s in batch}) != 1:
        raise ValueError('image dimensions differ; explicitly configure resizing before batching')
    sizes = [len(s['features']) for s in batch]
    return {
        'images': torch.stack([s['image'] for s in batch]),
        'features': torch.cat([s['features'] for s in batch]),
        'centers': torch.cat([s['centers'] for s in batch]),
        'entity_indices': torch.cat([s['entity_indices'] for s in batch]),
        'offsets': torch.tensor(np.cumsum(sizes), dtype=torch.long),
        'targets': [s['targets'] for s in batch],
        'image_ids': [s['image_id'] for s in batch],
        'provenance': [s['provenance'] for s in batch],
    }
