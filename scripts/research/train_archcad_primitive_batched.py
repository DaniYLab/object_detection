"""Batched primitive-only panoptic pilot: several images per optimizer step.

Motivation and measurements are in `docs/experiments/archcad_batched_step_20260917.md`.
One unbatched step costs ~10.5 ms of wall clock for ~1.7 us of arithmetic, so the
frozen pilot is host/launch bound and leaves the GPU idle. Grouping images into one
step amortizes the launch overhead: measured 2.9x throughput at B=16 with
length-grouped batching, processing the same images under the same per-image loss.

This is a different optimization protocol, not a faster version of the frozen one:
one optimizer step now covers B images, so steps per epoch and the effective
gradient batch change. Results from it must not be mixed with the frozen pilot.

What is deliberately kept identical to the frozen pilot:

- image selection (`select_samples`), eligibility caps and the split fingerprint;
- the decoder and the evaluator, both consumed through the frozen code paths;
- the per-image loss (`primitive_mask_loss`, applied per image by
  `batched_primitive_mask_loss`);
- checkpoint payload schema, so the frozen diagnostics can read these checkpoints
  (the batched model shares `state_dict` keys with the frozen one).

Sanity check that no checkpoint-payload key can silently diverge from the pilot:
`tests/test_archcad_batched_pilot.py` compares both trainers' payloads on the same
synthetic input.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.research.train_archcad_primitive_pilot import (checkpoint_payload, compact,
                                                            enforce_determinism, evaluation_due,
                                                            select_samples)
from src.data.archcad_panoptic_dataset import ArchCADPanopticDataset, REPRESENTATION
from src.data.archcad_rich_features import ArchCADPrimitiveRich, augment_items
from src.evaluation.archcad_panoptic import evaluate_archcad_panoptic
from src.models.archcad_primitive_baseline import decode_primitive_masks, primitive_mask_loss
from src.models.archcad_primitive_batched import (ArchCADPrimitiveBatched, batch_items,
                                                  batched_primitive_mask_loss, pad_primitive_batch,
                                                  padded_fraction)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@torch.no_grad()
def evaluate_batched(model, items, truths, batch_size, device=DEVICE):
    """Frozen evaluation contract through the batched forward.

    Metrics are produced by the frozen decoder and evaluator on a per-image slice,
    so they are identical to the frozen `evaluate`; the loss is the per-image mean,
    which is also identical. Batching here is only a speed knob.
    """
    model.eval()
    predictions, losses = [], []
    for chunk in batch_items(items, batch_size):
        batch = pad_primitive_batch(chunk, device)
        output = model(batch['features'], batch['padding_mask'])
        for index, item in enumerate(chunk):
            size = batch['sizes'][index]
            sliced = {'class_logits': output['class_logits'][index],
                      'mask_logits': output['mask_logits'][index][:, :size]}
            losses.append(float(primitive_mask_loss(sliced, item['targets'])['loss']))
            predictions.append(decode_primitive_masks(sliced, item['image_id']))
    return {'loss': sum(losses) / len(losses), 'metrics': evaluate_archcad_panoptic(truths, predictions)}


def training_order(items, batch_size, seed, epoch, group_by_length):
    """Epoch order, chunked into batches; grouping by length minimizes padding."""
    order = list(range(len(items)))
    random.Random(seed + epoch).shuffle(order)
    if group_by_length:
        order.sort(key=lambda index: len(items[index]['features']))
    return batch_items(order, batch_size)


def batch_padding(items, batches):
    """Share of primitive slots that are padding, weighted by batch size."""
    return sum(padded_fraction([items[index] for index in batch]) * len(batch) for batch in batches) / len(items)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path('data/ArchCAD'))
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--train-images', type=int, default=32)
    p.add_argument('--val-images', type=int, default=16)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--eval-every', type=int, default=0,
                   help='Evaluate all selected train/val images and save checkpoint every N epochs; 0 means final only.')
    p.add_argument('--max-primitives', type=int, default=1024)
    p.add_argument('--queries', type=int, default=64)
    p.add_argument('--dim', type=int, default=64)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--encoder-layers', type=int, default=1)
    p.add_argument('--decoder-layers', type=int, default=2)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--selection-seed', type=int, default=1337,
                   help='Fixed across model seeds so replication uses identical images.')
    p.add_argument('--batch-size', type=int, default=8, help='Images per optimizer step.')
    p.add_argument('--batch-order', choices=['shuffled', 'length-grouped'], default='length-grouped')
    p.add_argument('--eval-batch-size', type=int, default=8)
    p.add_argument('--rich-features', action='store_true',
                   help='Append the scale-aware dimensions from src/data/archcad_rich_features.py.')
    p.add_argument('--deterministic', action='store_true',
                   help='Force reproducible CUDA kernels; changes kernel selection relative to the unbatched runs.')
    p.add_argument('--device', default=DEVICE)
    args = p.parse_args()
    if args.eval_every < 0:
        raise ValueError('eval-every must be non-negative')
    if min(args.train_images, args.val_images, args.epochs, args.max_primitives, args.queries, args.batch_size) <= 0:
        raise ValueError('sample budgets, queries, epochs and batch size must be positive')
    if args.dim % args.heads:
        raise ValueError('dim must be divisible by heads')
    determinism = enforce_determinism() if args.deterministic else {'enabled': False}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    train_ds = ArchCADPanopticDataset(args.data_root, split='train')
    val_ds = ArchCADPanopticDataset(args.data_root, split='val')
    train, train_gt, train_skips = select_samples(train_ds, args.train_images, args.selection_seed, args.max_primitives, args.queries)
    val, val_gt, val_skips = select_samples(val_ds, args.val_images, args.selection_seed + 1, args.max_primitives, args.queries)
    if args.rich_features:
        augment_items(train)
        augment_items(val)
    architecture = ArchCADPrimitiveRich if args.rich_features else ArchCADPrimitiveBatched
    model = architecture(dim=args.dim, num_queries=args.queries, heads=args.heads,
                         encoder_layers=args.encoder_layers,
                         decoder_layers=args.decoder_layers).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    steps_per_epoch = int(np.ceil(len(train) / args.batch_size))
    config = {**vars(args), 'data_root': str(args.data_root), 'output_dir': str(args.output_dir),
              'device': args.device, 'model': model.config, 'representation': REPRESENTATION,
              'split_fingerprint': train_ds.split_fingerprint,
              'parameter_count': sum(x.numel() for x in model.parameters()), 'torch_version': str(torch.__version__),
              'device_name': torch.cuda.get_device_name() if args.device.startswith('cuda') else 'cpu',
              'train': [{'image_id': i['image_id'], **i['provenance']} for i in train],
              'val': [{'image_id': i['image_id'], **i['provenance']} for i in val],
              'selection_skips': {'train': train_skips, 'val': val_skips},
              'training': {'batch_images': args.batch_size, 'steps_per_epoch': steps_per_epoch,
                           'total_steps': steps_per_epoch * args.epochs, 'precision': 'fp32',
                           'gradient_clip': 1., 'optimizer': 'AdamW', 'weight_decay': .01,
                           'scheduler': 'none; fixed pilot budget', 'loss_weights': [2, 7, 5],
                           'no_object_weight': .05},
              'selection_rule': 'seeded random eligibility under primitive/query caps; no target truncation',
              'protocol': 'batched_step_v1; one optimizer step per batch of images'
                          + ('; scale-aware input features' if args.rich_features else ''),
              'representation_detail': ('archcad_entity_four_points_v1' if not args.rich_features
                                        else 'archcad_entity_four_points_v1 + scale-aware extras'),
              'metric_profile': 'local reconstruction v1; paper parity unverified', 'paper_reproduction': False,
              'determinism': determinism}
    config['decoder'] = {'score_threshold': .1, 'mask_threshold': .5,
                         'assignment': 'query-score times mask-probability argmax; merge stuff'}
    hashed = [Path(__file__).resolve(),
              Path('src/models/archcad_primitive_batched.py'),
              Path('src/models/archcad_primitive_baseline.py'),
              Path('src/data/archcad_panoptic_dataset.py'),
              Path('src/data/archcad_panoptic.py'),
              Path('src/evaluation/archcad_panoptic.py')]
    if args.rich_features:
        hashed.append(Path('src/data/archcad_rich_features.py'))
    config['implementation_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                       for path in hashed}
    config['padding_fraction_first_epoch'] = batch_padding(
        train, training_order(train, args.batch_size, args.seed, 1, args.batch_order == 'length-grouped'))
    (args.output_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    start = time.monotonic()
    initial_train = evaluate_batched(model, train, train_gt, args.eval_batch_size, args.device)
    initial_val = evaluate_batched(model, val, val_gt, args.eval_batch_size, args.device)
    (args.output_dir / 'evaluation_000.json').write_text(
        json.dumps({'epoch': 0, 'train': initial_train, 'val': initial_val}, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'initial_train': compact(initial_train), 'initial_val': compact(initial_val),
                      'steps_per_epoch': steps_per_epoch, 'padding_fraction': config['padding_fraction_first_epoch']}), flush=True)
    history = []
    evaluations = [{'epoch': 0, 'train': compact(initial_train), 'val': compact(initial_val)}]
    for epoch in range(1, args.epochs + 1):
        model.train()
        batches = training_order(train, args.batch_size, args.seed, epoch, args.batch_order == 'length-grouped')
        losses, seen = [], 0
        for batch_indices in batches:
            chunk = [train[index] for index in batch_indices]
            batch = pad_primitive_batch(chunk, args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = batched_primitive_mask_loss(model(batch['features'], batch['padding_mask']), batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite batched pilot loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            # Weight each image, not each batch, so the reported loss matches the
            # per-image mean the frozen pilot reports.
            losses.append(float(loss.detach()) * len(chunk))
            seen += len(chunk)
        history.append({'epoch': epoch, 'training_loss': sum(losses) / seen})
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(json.dumps(history[-1]), flush=True)
        if evaluation_due(epoch, args.epochs, args.eval_every):
            final_train = evaluate_batched(model, train, train_gt, args.eval_batch_size, args.device)
            final_val = evaluate_batched(model, val, val_gt, args.eval_batch_size, args.device)
            evaluations.append({'epoch': epoch, 'train': compact(final_train), 'val': compact(final_val)})
            (args.output_dir / f'evaluation_{epoch:03d}.json').write_text(
                json.dumps({'epoch': epoch, 'train': final_train, 'val': final_val}, indent=2) + '\n', encoding='utf-8')
            (args.output_dir / 'progress.json').write_text(
                json.dumps({'history': history, 'evaluations': evaluations}, indent=2) + '\n', encoding='utf-8')
            torch.save(checkpoint_payload(model, optimizer, epoch, config), args.output_dir / f'epoch_{epoch:03d}.pt')
            print(json.dumps({'evaluation': evaluations[-1]}), flush=True)
    report = {'schema_version': 'archcad-primitive-pilot-v1', 'config': config, 'history': history,
              'initial_train': initial_train, 'initial_val': initial_val,
              'final_train': final_train, 'final_val': final_val, 'elapsed_seconds': time.monotonic() - start,
              'evaluations': evaluations,
              'interpretation': 'batched-step optimization feasibility; not a converged or paper-comparable result'}
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    torch.save(checkpoint_payload(model, optimizer, args.epochs, config), args.output_dir / 'last.pt')
    print(json.dumps({'final_train': compact(final_train), 'final_val': compact(final_val),
                      'elapsed_seconds': report['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
