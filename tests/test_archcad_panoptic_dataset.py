import hashlib
import json
import math

import numpy as np
from PIL import Image
import pytest
import torch

from src.data.archcad_panoptic_dataset import ArchCADPanopticDataset, archcad_panoptic_collate, primitive_points


def dataset_root(tmp_path):
    (tmp_path / 'json').mkdir()
    (tmp_path / 'png').mkdir()
    source = {'entities': [
        {'type': 'LINE', 'start': [0, 0], 'end': [10, 10], 'semantic': 1, 'instance': 'a'},
        {'type': 'CIRCLE', 'center': [8, 8], 'radius': 2, 'semantic': 20},
    ]}
    for name in ['a', 'b']:
        (tmp_path / 'json' / f'{name}.json').write_text(json.dumps(source), encoding='utf-8')
        Image.new('RGB', (16, 16), 'white').save(tmp_path / 'png' / f'{name}.png')
    ids = {'train': ['a', 'b'], 'val': [], 'test': []}
    fingerprint = hashlib.sha256(json.dumps({'seed': 1, **ids}, sort_keys=True).encode()).hexdigest()
    (tmp_path / 'splits.json').write_text(json.dumps({'seed': 1, 'strategy': 'sha256_group_partition', 'fingerprint': fingerprint,
                                                    'splits': {k: {'images': v} for k, v in ids.items()}}))
    return tmp_path


def test_label_changes_cannot_change_model_inputs(tmp_path):
    root = dataset_root(tmp_path)
    ds = ArchCADPanopticDataset(root)
    first = ds[0]
    path = root / 'json/a.json'
    raw = json.loads(path.read_text())
    raw['entities'][0].update(semantic=2, instance='different', rgb=[1, 2, 3], layer='label-bearing')
    path.write_text(json.dumps(raw))
    second = ds[0]
    assert torch.equal(first['features'], second['features'])
    assert torch.equal(first['centers'], second['centers'])
    assert not torch.equal(first['targets']['semantic_labels'], second['targets']['semantic_labels'])
    assert first['provenance']['raw_sha256'] != second['provenance']['raw_sha256']


def test_collate_keeps_instance_masks_in_their_own_images(tmp_path):
    ds = ArchCADPanopticDataset(dataset_root(tmp_path))
    batch = archcad_panoptic_collate([ds[1], ds[0]])
    assert batch['image_ids'] == ['b', 'a']
    assert batch['offsets'].tolist() == [2, 4]
    assert batch['entity_indices'].tolist() == [0, 1, 0, 1]
    assert batch['features'].shape == (4, 12)
    assert [t['masks'].shape for t in batch['targets']] == [(2, 2), (2, 2)]
    assert torch.all(batch['targets'][0]['masks'].sum(0) == 1)


def test_primitive_budget_fails_explicitly(tmp_path):
    ds = ArchCADPanopticDataset(dataset_root(tmp_path), max_primitives=1)
    with pytest.raises(ValueError, match='no implicit truncation'):
        ds[0]


def test_manifest_tampering_is_rejected(tmp_path):
    root = dataset_root(tmp_path)
    p = root / 'splits.json'
    raw = json.loads(p.read_text())
    raw['splits']['train']['images'] = ['a']
    p.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='fingerprint'):
        ArchCADPanopticDataset(root)


def test_points_follow_arc_direction_and_partial_ellipse():
    arc = primitive_points({'type': 'ARC', 'center': [0, 0], 'radius': 1, 'start_angle': 350, 'end_angle': 10, 'direction': 0})
    assert np.all(arc[:, 0] > .98)
    ellipse = primitive_points({'type': 'ELLIPSE', 'center': [0, 0], 'major_axis': [0, 10], 'ratio': .5,
                                'start_param': 0., 'end_param': math.pi / 2, 'is_ccw': True})
    np.testing.assert_allclose(ellipse[0], [0, 10], atol=1e-8)
    np.testing.assert_allclose(ellipse[-1], [-5, 0], atol=1e-8)
