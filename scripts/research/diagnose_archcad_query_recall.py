"""Separate raw-mask coverage, class errors and postprocessing misses.

Coverage is an optimistic per-GT upper bound (queries may cover multiple GTs),
not one-to-one detection recall. No thresholds or weights are changed.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.archcad_constants import ARCHCAD_THING_SEMANTIC_IDS
from src.data.archcad_panoptic_dataset import ArchCADPanopticDataset
from src.models.archcad_primitive_baseline import ArchCADPrimitiveBaseline, decode_primitive_masks


@torch.no_grad()
def diagnose(checkpoint, device='cpu', split='val'):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    cfg = saved['run_config']
    if split not in {'train', 'val'}:
        raise ValueError('diagnostic split must be train or val')
    ds = ArchCADPanopticDataset(cfg['data_root'], split=split)
    if ds.split_fingerprint != cfg['split_fingerprint']:
        raise ValueError('split fingerprint drift')
    lookup = {image_id: i for i, image_id in enumerate(ds.image_ids)}
    model = ArchCADPrimitiveBaseline(**saved['model_config']).to(device).eval()
    model.load_state_dict(saved['model'])
    torch.set_num_threads(4)
    counts = Counter()
    for reference in cfg[split]:
        item = ds[lookup[reference['image_id']]]
        if item['provenance']['raw_sha256'] != reference['raw_sha256']:
            raise ValueError('source annotation drift')
        output = {k: v.cpu() for k, v in model(item['features'].to(device)).items()}
        score, cls = output['class_logits'].softmax(-1).max(-1)
        masks = output['mask_logits'].sigmoid() >= .5
        target = item['targets']
        weights = target['length_weights'] * (target['semantic_labels'] != 30)
        gt = target['masks'].double()
        intersection = (gt * weights) @ masks.double().T
        union = (gt * weights).sum(1)[:, None] + (masks * weights).sum(1)[None] - intersection
        iou = intersection / union.clamp_min(1e-12)
        for i, label in enumerate(target['labels'].tolist()):
            if label not in ARCHCAD_THING_SEMANTIC_IDS or (gt[i] * weights).sum() <= 0:
                continue
            counts['thing_gt'] += 1
            overlap = iou[i] > .5
            counts['raw_mask_class_agnostic_covered'] += int(overlap.any())
            counts['raw_mask_correct_class_covered'] += int((overlap & (cls == label)).any())
            counts['raw_mask_correct_class_score_kept_covered'] += int((overlap & (cls == label) & (score >= .1)).any())
        final = decode_primitive_masks(output, item['image_id'])
        counts['final_predicted_thing_segments'] += sum(p['semantic_id'] in ARCHCAD_THING_SEMANTIC_IDS for p in final['instances'])
        counts['final_scorable_thing_segments'] += sum(
            p['semantic_id'] in ARCHCAD_THING_SEMANTIC_IDS and float(weights[p['mask']].sum()) > 0
            for p in final['instances'])
    return {'schema_version': 'archcad-query-recall-diagnostic-v1', 'checkpoint': str(checkpoint),
            'seed': cfg['seed'], 'device': device, 'split': split, 'image_count': len(cfg[split]), 'counts': dict(counts),
            'coverage': {k: v / counts['thing_gt'] for k, v in counts.items() if k.endswith('_covered')},
            'interpretation': 'optimistic raw-query per-GT coverage, not one-to-one recall; fixed IoU>0.5, score>=0.1, mask>=0.5'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--split', choices=['train', 'val'], default='val')
    args = p.parse_args()
    report = diagnose(args.checkpoint, args.device, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
