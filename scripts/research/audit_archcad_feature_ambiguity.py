"""Audit identical model-input tokens assigned different non-ignore targets.

A geometry-only, permutation-equivariant model cannot separate identical input
tokens within the same drawing. Two irreducible cases exist and are counted
separately: different target-mask signatures (the mask head cannot split them)
and different semantic labels (the class head cannot distinguish them). This
diagnostic quantifies that limitation; it does not change labels or infer which
annotation should be kept.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.archcad_constants import ARCHCAD_IGNORE_LABEL
from src.data.archcad_panoptic_dataset import ArchCADPanopticDataset


def conflicting_groups(features, labels, masks):
    """Split identical-input primitive groups into mask conflicts and label conflicts.

    Returns (mask_conflicts, label_conflicts, both) as lists of entity-index arrays,
    each containing only groups with more than one non-ignore primitive.
    """
    _, inverse, counts = np.unique(features, axis=0, return_inverse=True, return_counts=True)
    mask_conflicts, label_conflicts, both = [], [], []
    for group in np.flatnonzero(counts > 1):
        indices = np.flatnonzero((inverse == group) & (labels != ARCHCAD_IGNORE_LABEL))
        if len(indices) <= 1:
            continue
        mask_clash = len(np.unique(masks[:, indices].T, axis=0)) > 1
        label_clash = len(np.unique(labels[indices])) > 1
        if mask_clash:
            mask_conflicts.append((indices, labels[indices]))
        if label_clash:
            label_conflicts.append((indices, labels[indices]))
        if mask_clash or label_clash:
            both.append(indices)
    return mask_conflicts, label_conflicts, both


def audit_config(config_path, split):
    config = json.loads(config_path.read_text(encoding='utf-8'))
    ds = ArchCADPanopticDataset(config['data_root'], split=split)
    if ds.split_fingerprint != config['split_fingerprint']:
        raise ValueError('split fingerprint drift')
    lookup = {v: i for i, v in enumerate(ds.image_ids)}
    rows, totals = [], Counter()
    for reference in config[split]:
        sample = ds[lookup[reference['image_id']]]
        if sample['provenance']['raw_sha256'] != reference['raw_sha256']:
            raise ValueError('annotation source drift')
        features = sample['features'].numpy()
        labels = sample['targets']['semantic_labels'].numpy()
        masks = sample['targets']['masks'].numpy()
        _, _, counts = np.unique(features, axis=0, return_inverse=True, return_counts=True)
        mask_conflicts, label_conflicts, ambiguous = conflicting_groups(features, labels, masks)
        pairs = dict.fromkeys([(tuple(i.tolist()), tuple(s.tolist())) for i, s in mask_conflicts + label_conflicts])
        conflicts = [{'entity_indices': list(indices), 'semantic_ids': list(semantic_ids),
                      'distinct_target_signatures': len(np.unique(masks[:, list(indices)].T, axis=0)),
                      'distinct_labels': len(set(semantic_ids))}
                     for indices, semantic_ids in pairs]
        row = {'image_id': sample['image_id'], 'primitive_count': len(features),
               'duplicate_feature_groups': int((counts > 1).sum()),
               'conflicting_mask_groups': len(mask_conflicts),
               'conflicting_label_groups': len(label_conflicts),
               'ambiguous_groups': len(ambiguous),
               'conflicting_primitives': int(sum(len(indices) for indices in ambiguous)),
               'conflicts': conflicts}
        rows.append(row)
        for key in ('primitive_count', 'duplicate_feature_groups', 'conflicting_mask_groups',
                    'conflicting_label_groups', 'ambiguous_groups', 'conflicting_primitives'):
            totals[key] += row[key]
    return {'schema_version': 'archcad-feature-ambiguity-v1', 'config': str(config_path), 'split': split,
            'image_count': len(rows), 'totals': dict(totals), 'records': rows,
            'interpretation': 'exact float32 model-input collisions only; not near-duplicate detection or a bound on PQ'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config', type=Path)
    p.add_argument('--split', choices=['train', 'val'], default='train')
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    report = audit_config(a.config, a.split)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
