"""Assemble the 2026-09-17 determinism, control, budget and capacity evidence.

Reads run reports under a run directory (default `outputs/panoptic`) and prints the
tables quoted in `docs/experiments/archcad_optimization_20260917.md`. Read-only: it
never starts a run, selects samples or writes reports. Runs that are absent are
reported as missing rather than skipped silently.

Usage:
    python scripts/research/summarize_archcad_optimization.py
    python scripts/research/summarize_archcad_optimization.py --json report.json
"""
import argparse
import json
from pathlib import Path

FIELDS = ('loss', 'thing_recall', 'pooled_pq', 'thing_pq', 'semantic_f1_macro', 'instance_ap50')
CONTROL = 'det_control_a'
CONTROL_REPEAT = 'det_control_b'
REFERENCE = 'primitive_256_s1337'
NONDETERMINISTIC_PEER = 'budget100_20260917_s1337'
TOLERANCE = 1e-6
CAPACITY_RUNS = ('cap_dim64_200', 'cap_dim64_600', 'cap_dim64_1200', 'cap_dim64_3000',
                 'cap_dim128_200', 'cap_dim128_600', 'cap_dim256_200')
OVERFIT_GATE = {'thing_recall': .95, 'pooled_pq': .90}
BUDGET_GATE = {'thing_recall_gain': .03}


def load(root, name):
    path = Path(root) / name / 'report.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def final_val(report):
    """Final-epoch validation row, from the evaluation series or the legacy fields."""
    if 'evaluations' in report:
        return report['evaluations'][-1]['val'], report['evaluations'][-1]['epoch']
    metrics = report['final_val']['metrics']
    thing = metrics['panoptic']['thing']
    return {'loss': report['final_val']['loss'], 'pooled_pq': metrics['panoptic']['all']['pq'],
            'thing_pq': thing['pq'],
            'thing_recall': thing['tp'] / (thing['tp'] + thing['fn']) if thing['tp'] + thing['fn'] else 0.,
            'semantic_f1_macro': metrics['semantic']['f1_macro'],
            'instance_ap50': metrics['instance']['ap50']}, report['config']['epochs']


def val_at(report, epoch):
    return [e for e in report['evaluations'] if e['epoch'] == epoch][0]['val']


def final_train(report):
    return report['evaluations'][-1]['train']


def components(root, name, epoch):
    """PQ components and counts for the saved full evaluation at `epoch`, if present."""
    path = Path(root) / name / f'evaluation_{epoch:03d}.json'
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding='utf-8'))['train']['metrics']['panoptic']
    return {'thing_rq': record['thing']['rq'], 'thing_sq': record['thing']['sq'],
            'thing_tp': record['thing']['tp'], 'thing_fn': record['thing']['fn'],
            'fp': record['all']['fp']}


def collect(root, names):
    out = {}
    for name in names:
        report = load(root, name)
        if report is None:
            out[name] = None
            continue
        value, epoch = final_val(report)
        out[name] = {'report': report, 'final_val': value, 'epoch': epoch,
                     'final_train': final_train(report) if 'evaluations' in report else None,
                     'components': components(root, name, epoch)}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('outputs/panoptic'))
    p.add_argument('--json', type=Path, default=None)
    args = p.parse_args()
    runs = collect(args.root, (REFERENCE, NONDETERMINISTIC_PEER, CONTROL, CONTROL_REPEAT,
                              'det_budget100') + CAPACITY_RUNS)
    summary = {}

    print('A. seed-1337 control reproducibility (256 train / 128 val, 20 epochs)')
    for name in (REFERENCE, NONDETERMINISTIC_PEER, CONTROL, CONTROL_REPEAT):
        run = runs[name]
        if run is None:
            print(f'   {name:34s} MISSING')
            continue
        label = f'{name}{" (deterministic)" if name.startswith("det_") else ""}'
        print(f'   {label:34s} ' + ' '.join(f'{k}={run["final_val"][k]:.6f}' for k in FIELDS))

    if runs[CONTROL] and runs[CONTROL_REPEAT]:
        deltas = {k: runs[CONTROL]['final_val'][k] - runs[CONTROL_REPEAT]['final_val'][k] for k in FIELDS}
        summary['deterministic_repeat_bit_identical'] = all(d == 0. for d in deltas.values())
        print(f'   A1 repeated deterministic run bit-identical: {summary["deterministic_repeat_bit_identical"]}')
    if runs[CONTROL] and runs[REFERENCE]:
        deltas = {k: runs[CONTROL]['final_val'][k] - runs[REFERENCE]['final_val'][k] for k in FIELDS}
        summary['control_within_tolerance'] = all(abs(d) <= TOLERANCE for d in deltas.values())
        summary['control_deltas'] = deltas
        print(f'   A2 control vs saved {REFERENCE} within {TOLERANCE:g}: {summary["control_within_tolerance"]}')
        for k, v in deltas.items():
            print(f'        {k:20s} delta={v:+.6f}')

    # The nondeterministic peer is the budget run, so compare it at epoch 20, the
    # endpoint of the saved reference, not at its own longer-budget endpoint.
    peer = runs[NONDETERMINISTIC_PEER]
    peer_row = val_at(peer['report'], 20) if peer and 'evaluations' in peer['report'] else None
    peers = [r for r in (runs[REFERENCE], {'final_val': peer_row} if peer_row else None, runs[CONTROL]) if r]
    if len(peers) > 1:
        summary['identical_protocol_spread'] = {
            k: max(r['final_val'][k] for r in peers) - min(r['final_val'][k] for r in peers)
            for k in ('thing_recall', 'pooled_pq', 'semantic_f1_macro')}
        print('   A3 spread across identical-protocol seed-1337 runs (all at epoch 20): '
              + ' '.join(f'{k}={v:.6f}' for k, v in summary['identical_protocol_spread'].items()))

    print()
    print('B. H2 budget gate (deterministic, epochs 20 vs final)')
    budget = runs['det_budget100']
    if budget is None:
        print('   det_budget100 MISSING')
    else:
        for entry in budget['report']['evaluations']:
            print(f'   epoch {entry["epoch"]:3d} ' + ' '.join(f'{k}={entry["val"][k]:.6f}' for k in FIELDS))
        first, last = val_at(budget['report'], 20), budget['final_val']
        gain = last['thing_recall'] - first['thing_recall']
        pq_delta = last['pooled_pq'] - first['pooled_pq']
        summary['budget_gate'] = {'thing_recall_gain': gain, 'pooled_pq_delta': pq_delta,
                                  'passed': gain >= BUDGET_GATE['thing_recall_gain'] and pq_delta >= 0}
        print(f'   thing_recall gain {gain:+.6f} (needs >= {BUDGET_GATE["thing_recall_gain"]}), '
              f'pooled_pq delta {pq_delta:+.6f} (must not regress) -> passed={summary["budget_gate"]["passed"]}')

    print()
    print('C. H1 overfit gate / capacity sweep (8 train images, seed 1337)')
    print('   run                params  epochs  train_recall  train_pq  train_f1   rq     sq')
    summary['capacity'] = {}
    for name in CAPACITY_RUNS:
        run = runs.get(name)
        if run is None:
            print(f'   {name:18s} MISSING')
            continue
        config, train, comp = run['report']['config'], run['final_train'], run['components'] or {}
        passed = (train['thing_recall'] >= OVERFIT_GATE['thing_recall']
                  and train['pooled_pq'] >= OVERFIT_GATE['pooled_pq'])
        summary['capacity'][name] = {'params': config['parameter_count'], 'epochs': config['epochs'],
                                     'train': train, 'gate_passed': passed, **comp}
        print(f'   {name:18s} {config["parameter_count"]:8,d} {config["epochs"]:6d} '
              f'{train["thing_recall"]:12.4f}  {train["pooled_pq"]:8.4f}  {train["semantic_f1_macro"]:8.4f} '
              f'{comp.get("thing_rq", float("nan")):6.3f} {comp.get("thing_sq", float("nan")):6.3f}'
              f'{"  PASS" if passed else ""}')
    print(f'   gate: thing_recall >= {OVERFIT_GATE["thing_recall"]}, pooled_pq >= {OVERFIT_GATE["pooled_pq"]}')

    if args.json:
        args.json.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        print(f'\nwrote {args.json}')


if __name__ == '__main__':
    main()
