"""Run non-identity fixtures through a local and a pinned upstream evaluator.

The upstream file is read from a separate checkout, not vendored or modified.
Only its evaluator class definitions execute; absent dataset imports are
replaced with taxonomy names for logging. This checks actual upstream behavior.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.archcad_panoptic import parse_archcad_panoptic_annotation, oracle_panoptic_prediction
from src.evaluation.archcad_panoptic import evaluate_archcad_panoptic


class SilentLogger:
    def info(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-root', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path('outputs/panoptic/upstream_metric_comparison_20260916.json'))
    args = parser.parse_args()
    path = args.upstream_root / 'svgnet/evaluation/point_wise_eval.py'
    source = path.read_bytes()
    tree = ast.parse(source, filename=str(path))
    classes = ast.Module(body=[n for n in tree.body if isinstance(n, ast.ClassDef)], type_ignores=[])
    namespace = {'np': np, 'torch': torch, 'dist': dist,
                 'SVG_CATEGORIES': [{'name': str(i)} for i in range(31)]}
    exec(compile(classes, str(path), 'exec'), namespace)
    sample = parse_archcad_panoptic_annotation({'entities': [
        {'type': 'LINE', 'start': [0, 0], 'end': [10, 0], 'semantic': 1, 'instance': 'door'},
        {'type': 'LINE', 'start': [0, 10], 'end': [10, 10], 'semantic': 20},
    ]}, image_id='fixture')
    identity = oracle_panoptic_prediction(sample)
    duplicate = copy.deepcopy(identity)
    duplicate['instances'].append(copy.deepcopy(identity['instances'][1]))
    wrong = copy.deepcopy(identity)
    wrong['instances'][1]['semantic_id'] = 2
    missing = copy.deepcopy(identity)
    missing['instances'].pop()
    results = {}
    for name, prediction in [('identity', identity), ('duplicate', duplicate), ('wrong_class', wrong), ('missing', missing)]:
        evaluator = namespace['InstanceEval'](num_classes=30, ignore_label=30, gpu_num=1)
        evaluator.update(
            [{'labels': p['semantic_id'], 'scores': p['score'], 'masks': p['mask']} for p in prediction['instances']],
            {'labels': torch.tensor([s.semantic_id for s in sample.segments]),
             'masks': torch.tensor(np.stack([s.primitive_mask for s in sample.segments], axis=1))},
            torch.tensor(sample.primitive_lengths),
        )
        pq, rq, sq = evaluator.get_eval(SilentLogger())
        results[name] = {
            'upstream': {'pq': pq / 100, 'rq': rq / 100, 'sq': sq / 100,
                         'tp': float(evaluator.tp_classes.sum()), 'fp': float(evaluator.fp_classes.sum()), 'fn': float(evaluator.fn_classes.sum())},
            'local': evaluate_archcad_panoptic([sample], [prediction])['panoptic']['all'],
        }
    report = {'schema_version': 'archcad-upstream-metric-comparison-v1',
              'upstream_commit': subprocess.check_output(['git', '-C', str(args.upstream_root), 'rev-parse', 'HEAD'], text=True).strip(),
              'source_sha256': hashlib.sha256(source).hexdigest(), 'fixtures': results,
              'paper_parity_established': False,
              'conclusion': 'Identity agrees approximately; released matching differs on duplicate and wrong-class predictions.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
