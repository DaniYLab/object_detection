"""Bounded primitive-only panoptic pilot on the frozen public-40K train/val split.

Selection, model, optimizer, decoder and metric conventions are written before
training. This is an optimization feasibility run, not a paper reproduction.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.archcad_panoptic import parse_archcad_panoptic_annotation
from src.data.archcad_panoptic_dataset import ArchCADPanopticDataset, REPRESENTATION
from src.evaluation.archcad_panoptic import evaluate_archcad_panoptic
from src.models.archcad_primitive_baseline import ArchCADPrimitiveBaseline, primitive_mask_loss, decode_primitive_masks


def select_samples(dataset, count, seed, max_primitives, num_queries):
    order = list(range(len(dataset)))
    random.Random(seed).shuffle(order)
    selected, truths, skips = [], [], Counter()
    for index in order:
        image_id = dataset.image_ids[index]
        raw = json.loads((dataset.root / 'json' / f'{image_id}.json').read_text(encoding='utf-8'))
        if not 1 <= len(raw['entities']) <= max_primitives:
            skips['primitive_budget'] += 1
            continue
        truth = parse_archcad_panoptic_annotation(raw, image_id=image_id)
        if not 1 <= len(truth.segments) <= num_queries:
            skips['segment_budget'] += 1
            continue
        item = dataset[index]
        item.pop('image')  # The declared primitive-only control does not consume pixels.
        selected.append(item)
        truths.append(truth)
        if len(selected) == count:
            return selected, truths, dict(skips)
    raise ValueError(f'only {len(selected)} eligible samples; requested {count}')


@torch.no_grad()
def evaluate(model, items, truths, device):
    model.eval()
    predictions, losses = [], []
    for item in items:
        output = model(item['features'].to(device))
        losses.append(float(primitive_mask_loss(output, item['targets'])['loss']))
        predictions.append(decode_primitive_masks(output, item['image_id']))
    return {'loss': sum(losses) / len(losses), 'metrics': evaluate_archcad_panoptic(truths, predictions)}


def compact(result):
    metrics = result['metrics']
    return {'loss': result['loss'], 'semantic_f1_macro': metrics['semantic']['f1_macro'],
            'semantic_wf1_macro': metrics['semantic']['weighted_f1_macro'],
            'instance_ap50': metrics['instance']['ap50'], 'instance_map': metrics['instance']['map_50_95'],
            'pooled_pq': metrics['panoptic']['all']['pq'],
            'tp': metrics['panoptic']['all']['tp'], 'fp': metrics['panoptic']['all']['fp'], 'fn': metrics['panoptic']['all']['fn']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path('data/ArchCAD'))
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--train-images', type=int, default=32)
    p.add_argument('--val-images', type=int, default=16)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--max-primitives', type=int, default=1024)
    p.add_argument('--queries', type=int, default=64)
    p.add_argument('--dim', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    if min(args.train_images, args.val_images, args.epochs, args.max_primitives, args.queries) <= 0:
        raise ValueError('sample budgets, queries and epochs must be positive')
    # A run is immutable once configured; never overwrite an existing run.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    train_ds = ArchCADPanopticDataset(args.data_root, split='train')
    val_ds = ArchCADPanopticDataset(args.data_root, split='val')
    train, train_gt, train_skips = select_samples(train_ds, args.train_images, args.seed, args.max_primitives, args.queries)
    val, val_gt, val_skips = select_samples(val_ds, args.val_images, args.seed + 1, args.max_primitives, args.queries)
    # Check content, not only sample ID disjointness, for this actual pilot.
    from hashlib import sha256
    hashes = lambda items: {sha256((args.data_root / 'png' / (i['image_id'] + '.png')).read_bytes()).hexdigest() for i in items}
    if hashes(train) & hashes(val):
        raise ValueError('pilot train/val share PNG content')
    model = ArchCADPrimitiveBaseline(dim=args.dim, num_queries=args.queries).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    config = {**vars(args), 'data_root': str(args.data_root), 'output_dir': str(args.output_dir),
              'model': model.config, 'representation': REPRESENTATION, 'split_fingerprint': train_ds.split_fingerprint,
              'parameter_count': sum(x.numel() for x in model.parameters()), 'torch_version': str(torch.__version__),
              'device_name': torch.cuda.get_device_name() if args.device.startswith('cuda') else 'cpu',
              'train': [{'image_id': i['image_id'], **i['provenance']} for i in train],
              'val': [{'image_id': i['image_id'], **i['provenance']} for i in val],
              'selection_skips': {'train': train_skips, 'val': val_skips},
              'training': {'batch_images': 1, 'precision': 'fp32', 'gradient_clip': 1., 'optimizer': 'AdamW', 'weight_decay': .01,
                           'scheduler': 'none; fixed pilot budget', 'loss_weights': [2, 7, 5], 'no_object_weight': .05},
              'selection_rule': 'seeded random eligibility under primitive/query caps; no target truncation',
              'metric_profile': 'local reconstruction v1; paper parity unverified', 'paper_reproduction': False}
    (args.output_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    start = time.monotonic()
    initial_train = evaluate(model, train, train_gt, args.device)
    initial_val = evaluate(model, val, val_gt, args.device)
    print(json.dumps({'initial_train': compact(initial_train), 'initial_val': compact(initial_val)}), flush=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train)))
        random.Random(args.seed + epoch).shuffle(order)
        losses = []
        for index in order:
            item = train[index]
            optimizer.zero_grad(set_to_none=True)
            loss = primitive_mask_loss(model(item['features'].to(args.device)), item['targets'])['loss']
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite pilot loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({'epoch': epoch, 'training_loss': sum(losses) / len(losses)})
        print(json.dumps(history[-1]), flush=True)
    final_train = evaluate(model, train, train_gt, args.device)
    final_val = evaluate(model, val, val_gt, args.device)
    report = {'schema_version': 'archcad-primitive-pilot-v1', 'config': config, 'history': history,
              'initial_train': initial_train, 'initial_val': initial_val,
              'final_train': final_train, 'final_val': final_val, 'elapsed_seconds': time.monotonic() - start,
              'interpretation': 'bounded optimization feasibility; not a converged or paper-comparable result'}
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    torch.save({'format': 'archcad-primitive-pilot-v1', 'model_config': model.config, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'epoch': args.epochs, 'run_config': config,
                'torch_rng': torch.get_rng_state()}, args.output_dir / 'last.pt')
    print(json.dumps({'final_train': compact(final_train), 'final_val': compact(final_val), 'elapsed_seconds': report['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
