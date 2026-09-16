"""Aggregate completed seeds only if data, model and implementation match."""
import argparse
import json
from pathlib import Path
import statistics


def comparable_signature(config):
    return {k: config[k] for k in (
        'selection_seed', 'train', 'val', 'model', 'representation', 'split_fingerprint',
        'epochs', 'lr', 'max_primitives', 'queries', 'training', 'decoder',
        'metric_profile', 'implementation_sha256')}


def summarize(reports):
    if len(reports) < 2:
        raise ValueError('at least two completed seeds required')
    reference = comparable_signature(reports[0]['config'])
    if any(comparable_signature(r['config']) != reference for r in reports[1:]):
        raise ValueError('replication confounded: data, model, budget or code differs')
    seeds = [r['config']['seed'] for r in reports]
    if len(set(seeds)) != len(seeds):
        raise ValueError('duplicate model seed')
    rows = []
    for report in reports:
        m = report['final_val']['metrics']
        thing = m['panoptic']['thing']
        rows.append({'seed': report['config']['seed'], 'val_loss': report['final_val']['loss'],
                     'semantic_f1_macro': m['semantic']['f1_macro'],
                     'weighted_semantic_f1_macro': m['semantic']['weighted_f1_macro'],
                     'instance_ap50': m['instance']['ap50'], 'instance_map': m['instance']['map_50_95'],
                     'pooled_pq': m['panoptic']['all']['pq'], 'thing_pq': thing['pq'],
                     'stuff_pq': m['panoptic']['stuff']['pq'],
                     'thing_recall': thing['tp'] / (thing['tp'] + thing['fn']) if thing['tp'] + thing['fn'] else 0.,
                     'matched_thing_segments': thing['tp'], 'missed_thing_segments': thing['fn'],
                     'elapsed_seconds': report['elapsed_seconds']})
    keys = [k for k in rows[0] if k != 'seed']
    return {'schema_version': 'archcad-primitive-replication-v1', 'seeds': seeds,
            'comparability_checked': True, 'per_seed': rows,
            'aggregate': {k: {'mean': statistics.mean(r[k] for r in rows),
                              'sample_std': statistics.stdev(r[k] for r in rows)} for k in keys},
            'train_images': len(reports[0]['config']['train']),
            'val_images': len(reports[0]['config']['val']),
            'interpretation': 'fixed-subset development control; seed variability only; paper parity unverified'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize([json.loads(p.read_text(encoding='utf-8')) for p in args.reports])
    result['source_reports'] = [str(p) for p in args.reports]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
