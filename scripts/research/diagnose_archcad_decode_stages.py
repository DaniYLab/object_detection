"""Decompose the gap between raw-query coverage and final one-to-one thing matches.

`diagnose_archcad_query_recall.py` reports three optimistic coverage tiers and the
final saved evaluation. This script inserts the missing stage — the decoder's own
disjoint assignment and mask binarisation — so the postprocessing loss can be
attributed to a specific step rather than to "postprocessing" as a whole.

The semantic map is measured separately: instances are built from query class
scores, so a semantic-map threshold policy cannot move the thing-recall ladder,
it can only move the semantic F1/wF1. Both policies are evaluated here on the
same dump.

Nothing here tunes thresholds or changes the frozen decoder default.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.archcad_constants import ARCHCAD_THING_SEMANTIC_IDS
from src.data.archcad_panoptic import parse_archcad_panoptic_annotation
from src.data.archcad_panoptic_dataset import ArchCADPanopticDataset
from src.evaluation.archcad_panoptic import evaluate_archcad_panoptic
from src.models.archcad_primitive_baseline import ArchCADPrimitiveBaseline, decode_primitive_masks

SCHEMA = 'archcad-decode-stage-diagnostic-v1'
IOU_GATE = .5
SCORE_GATE = .1
MASK_GATE = .5


def _truth(data_root, reference):
    raw_bytes = (data_root / 'json' / f"{reference['image_id']}.json").read_bytes()
    if hashlib.sha256(raw_bytes).hexdigest() != reference['raw_sha256']:
        raise ValueError('source annotation drift')
    return parse_archcad_panoptic_annotation(json.loads(raw_bytes.decode('utf-8')),
                                             image_id=reference['image_id'])


@torch.no_grad()
def diagnose(checkpoint, device='cpu'):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    cfg = saved['run_config']
    data_root = Path(cfg['data_root'])
    ds = ArchCADPanopticDataset(data_root, split='val')
    if ds.split_fingerprint != cfg['split_fingerprint']:
        raise ValueError('split fingerprint drift')
    lookup = {image_id: i for i, image_id in enumerate(ds.image_ids)}
    model = ArchCADPrimitiveBaseline(**saved['model_config']).to(device).eval()
    model.load_state_dict(saved['model'])
    torch.set_num_threads(4)

    counts = Counter()
    truths, default_predictions, consistent_predictions = [], [], []
    semantic_digests = []
    for reference in cfg['val']:
        item = ds[lookup[reference['image_id']]]
        truth = _truth(data_root, reference)
        truths.append(truth)
        output = {k: v.cpu() for k, v in model(item['features'].to(device)).items()}
        target = item['targets']
        weights = target['length_weights'] * (target['semantic_labels'] != 30)
        gt = target['masks'].double()

        probability = output['mask_logits'].sigmoid()
        classes = output['class_logits'].softmax(-1)
        score, label = classes.max(-1)

        # Stage 1-3: raw query masks, before any disjoint assignment or binarisation.
        raw_iou = (gt * weights) @ (probability >= MASK_GATE).double().T / (
            (gt * weights).sum(1)[:, None] + ((probability >= MASK_GATE).double() * weights).sum(1)[None]
            - (gt * weights) @ (probability >= MASK_GATE).double().T).clamp_min(1e-12)
        # Stage 4: the decoder's own kept-query assignment and mask binarisation,
        # still allowing several GT segments to be covered by one query.
        keep = (label != 30) & (score >= SCORE_GATE)
        kept_scores, kept_labels, kept_masks = score[keep], label[keep], probability[keep]
        if len(kept_scores):
            assigned = (kept_scores[:, None] * kept_masks).argmax(0)
            spans = torch.stack([((assigned == i) & (kept_masks[i] >= MASK_GATE)) for i in range(len(kept_scores))])
            span_iou = (gt * weights) @ spans.double().T / (
                (gt * weights).sum(1)[:, None] + (spans.double() * weights).sum(1)[None]
                - (gt * weights) @ spans.double().T).clamp_min(1e-12)
        for i, thing_label in enumerate(target['labels'].tolist()):
            if thing_label not in ARCHCAD_THING_SEMANTIC_IDS or (gt[i] * weights).sum() <= 0:
                continue
            counts['thing_gt'] += 1
            overlap = raw_iou[i] > IOU_GATE
            counts['s1_raw_class_agnostic'] += int(overlap.any())
            counts['s2_raw_correct_class'] += int((overlap & (label == thing_label)).any())
            counts['s3_raw_class_and_score'] += int((overlap & (label == thing_label) & (score >= SCORE_GATE)).any())
            if len(kept_scores):
                counts['s4_disjoint_mask_correct_class'] += int(
                    ((span_iou[i] > IOU_GATE) & (kept_labels == thing_label)).any())

        prediction = decode_primitive_masks(output, item['image_id'])
        default_predictions.append(prediction)
        consistent_predictions.append(
            decode_primitive_masks(output, item['image_id'], consistent_semantic_thresholds=True))
        semantic_digests.append(hashlib.sha256(json.dumps(
            [int(x) for x in prediction['semantic_labels']]).encode()).hexdigest())
        # Scorable means the prediction covers at least one primitive the evaluator weights.
        for record in prediction['instances']:
            if record['semantic_id'] in ARCHCAD_THING_SEMANTIC_IDS:
                counts['predicted_thing_segments'] += 1
                counts['scorable_thing_segments'] += int(float(weights[record['mask']].sum()) > 0)

    default_metrics = evaluate_archcad_panoptic(truths, default_predictions)
    consistent_metrics = evaluate_archcad_panoptic(truths, consistent_predictions)
    thing = default_metrics['panoptic']['thing']
    counts['s5_one_to_one_thing_tp'] = int(thing['tp'])
    counts['s5_one_to_one_thing_fn'] = int(thing['fn'])
    changed = sum(
        int(a != b) for q, r in zip(default_predictions, consistent_predictions)
        for a, b in zip(q['semantic_labels'], r['semantic_labels']))
    primitives = sum(len(q['semantic_labels']) for q in default_predictions)
    return {
        'schema_version': SCHEMA, 'checkpoint': str(checkpoint), 'split': 'val',
        'seed': cfg['seed'], 'device': device, 'image_count': len(truths),
        'gates': {'iou': IOU_GATE, 'score': SCORE_GATE, 'mask': MASK_GATE},
        'counts': dict(counts),
        'coverage': {k: counts[k] / counts['thing_gt'] for k in (
            's1_raw_class_agnostic', 's2_raw_correct_class',
            's3_raw_class_and_score', 's4_disjoint_mask_correct_class')},
        'thing_recall': counts['s5_one_to_one_thing_tp'] / (
            counts['s5_one_to_one_thing_tp'] + counts['s5_one_to_one_thing_fn']),
        'semantic_map': {
            'primitives': primitives,
            'labels_changed_by_consistent_thresholds': changed,
            'default': {k: v for k, v in default_metrics['semantic'].items() if k != 'per_class'},
            'consistent_thresholds': {k: v for k, v in consistent_metrics['semantic'].items() if k != 'per_class'},
            'default_prediction_digest': hashlib.sha256(''.join(semantic_digests).encode()).hexdigest(),
        },
        'interpretation': (
            's1-s3 are optimistic per-segment coverage; s4 adds the decoder disjoint assignment and '
            'mask binarisation while still allowing many-to-one; s5 is the class-conditional '
            'one-to-one panoptic match. The semantic map cannot move s1-s5 because instances come '
            'from query class scores; it is reported separately against the semantic F1/wF1.'),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    report = diagnose(args.checkpoint, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
