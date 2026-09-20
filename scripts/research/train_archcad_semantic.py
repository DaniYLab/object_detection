"""Train the per-primitive semantic head and score it through the geometric grouper.

The reported pipeline is: predicted per-primitive semantics -> `group_instances` -> the
frozen panoptic evaluator. References at the same regime on 512 validation images, with
grouping constants fitted on the training split (see
`docs/experiments/archcad_semantics_bottleneck_20260917.md`):

  query/mask model end to end                          pooled PQ 0.148511
  its semantics + geometric grouping                   0.169716
  oracle semantics + geometric grouping                0.807876
  its per-primitive semantic accuracy                  31.56%

The head exists to attack the last two lines of that list: semantics carry ~93% of the gap,
and they were never predicted directly before.
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
from scripts.research.train_archcad_primitive_batched import training_order
from scripts.research.train_archcad_primitive_pilot import (checkpoint_payload, enforce_determinism,
                                                            evaluation_due, select_samples)
from src.data.archcad_constants import ARCHCAD_IGNORE_LABEL
from src.data.archcad_geometric_grouping import (DEFAULT_CONNECTIVITY, DEFAULT_PROXIMITY,
                                                 predictions_from_semantics)
from src.data.archcad_panoptic_dataset import ArchCADPanopticDataset
from src.data.archcad_rich_features import augment_items
from src.evaluation.archcad_panoptic import evaluate_archcad_panoptic
from src.models.archcad_semantic_head import (ArchCADPrimitiveNoContext, ArchCADPrimitiveSemantic,
                                              pad_semantic_batch, semantic_classification_loss)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def learning_rate_at(step, total_steps, base_lr, schedule, final_fraction=0.0):
    """Learning rate for a global step index (0-based).

    The 560-epoch run held the rate fixed and its last third oscillated by 0.013 pooled PQ
    between adjacent checkpoints while the train-validation gap widened fivefold, which is what
    a constant rate looks like from the outside. `cosine` anneals to `final_fraction * base_lr`
    over the budget so the endpoint is a settled point rather than a sample from a moving one.
    """
    if schedule == 'constant':
        return base_lr
    progress = min(1.0, max(0.0, step / max(1, total_steps)))
    return base_lr * (final_fraction + (1.0 - final_fraction) * 0.5 * (1.0 + float(np.cos(np.pi * progress))))


@torch.no_grad()
def evaluate_semantic(model, items, truths, batch_size, proximity, connectivity, device=DEVICE):
    """Semantic accuracy, semantic F1/wF1 and panoptic PQ for the full pipeline."""
    model.eval()
    semantics, losses = [], []
    correct = total = 0
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        batch = pad_semantic_batch(chunk, device)
        output = model(batch['features'], batch['padding_mask'])
        losses.append(float(semantic_classification_loss(output, batch)) * len(chunk))
        predicted = output['semantic_logits'].argmax(-1).cpu()
        for index, item in enumerate(chunk):
            n = batch['sizes'][index]
            item_semantics = predicted[index, :n].numpy()
            semantics.append(item_semantics)
            truth = item['targets']['semantic_labels'].numpy()
            keep = truth != ARCHCAD_IGNORE_LABEL
            correct += int((item_semantics[keep] == truth[keep]).sum())
            total += int(keep.sum())
    metrics = evaluate_archcad_panoptic(
        truths, predictions_from_semantics(items, semantics, proximity=proximity,
                                           connectivity=connectivity))
    return {'loss': sum(losses) / len(items), 'metrics': metrics,
            'accuracy': correct / max(total, 1), 'primitives': total}


def capped_evaluation_set(items, truths, cap):
    """Prefix subset for the diagnostic train-side evaluation; cap <= 0 keeps everything.

    A prefix of the seeded selection, so the subset is identical across runs and a train
    metric is comparable between them even when it covers only part of the split.
    """
    if cap <= 0:
        return items, truths
    return items[:cap], truths[:cap]


def compact(result):
    metrics = result['metrics']
    thing = metrics['panoptic']['thing']
    return {'loss': result['loss'], 'accuracy': result['accuracy'],
            'semantic_f1_macro': metrics['semantic']['f1_macro'],
            'semantic_f1_micro': metrics['semantic']['f1_micro'],
            'pooled_pq': metrics['panoptic']['all']['pq'],
            'thing_pq': thing['pq'],
            'thing_recall': thing['tp'] / (thing['tp'] + thing['fn']) if thing['tp'] + thing['fn'] else 0.,
            'tp': metrics['panoptic']['all']['tp'], 'fp': metrics['panoptic']['all']['fp'],
            'fn': metrics['panoptic']['all']['fn']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path('data/ArchCAD'))
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--train-images', type=int, default=4096,
                   help='-1 selects every eligible training image.')
    p.add_argument('--val-images', type=int, default=512)
    p.add_argument('--epochs', type=int, default=28)
    p.add_argument('--eval-every', type=int, default=7)
    p.add_argument('--eval-train-images', type=int, default=2048,
                   help='Cap the diagnostic train-side evaluation to this many selected images; '
                        '0 evaluates every training image. The validation set is never capped.')
    p.add_argument('--max-primitives', type=int, default=1024)
    p.add_argument('--queries', type=int, default=64,
                   help='Only used for selection eligibility, so the subset matches the baseline runs.')
    p.add_argument('--architecture', choices=['transformer', 'no_context'], default='transformer',
                   help='no_context classifies each primitive from its own features alone.')
    p.add_argument('--dim', type=int, default=64)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--width', type=int, default=256, help='Hidden width for --architecture no_context.')
    p.add_argument('--encoder-layers', type=int, default=2)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--lr-schedule', choices=['constant', 'cosine'], default='constant',
                   help='constant reproduces every run up to 2026-09-20; cosine anneals per step '
                        'from --lr to --lr-final over the whole budget.')
    p.add_argument('--lr-final', type=float, default=0.0,
                   help='End learning rate of --lr-schedule cosine, as a fraction of --lr.')
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--selection-seed', type=int, default=1337)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--batch-order', choices=['shuffled', 'length-grouped'], default='length-grouped')
    p.add_argument('--eval-batch-size', type=int, default=8)
    p.add_argument('--rich-features', action='store_true',
                   help='Append the scale-aware dimensions before classification.')
    p.add_argument('--proximity', type=float, default=None,
                   help='Force one proximity radius for every class. Default uses the fitted '
                        'per-class table in src/data/archcad_geometric_grouping.py.')
    p.add_argument('--connectivity', type=float, default=DEFAULT_CONNECTIVITY)
    p.add_argument('--deterministic', action='store_true')
    p.add_argument('--device', default=DEVICE)
    args = p.parse_args()
    if args.eval_every < 0:
        raise ValueError('eval-every must be non-negative')
    if args.train_images == 0:
        raise ValueError('train-images may be -1 (all eligible) or a positive count')
    if min(args.val_images, args.epochs, args.max_primitives, args.batch_size) <= 0:
        raise ValueError('sample budgets, epochs and batch size must be positive')
    if args.architecture != "no_context" and args.dim % args.heads:
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
    train, train_gt, train_skips = select_samples(train_ds, args.train_images, args.selection_seed,
                                                  args.max_primitives, args.queries)
    val, val_gt, val_skips = select_samples(val_ds, args.val_images, args.selection_seed + 1,
                                            args.max_primitives, args.queries)
    if args.rich_features:
        augment_items(train)
        augment_items(val)
    in_features = int(train[0]['features'].shape[1])
    if args.architecture == 'no_context':
        model = ArchCADPrimitiveNoContext(in_features=in_features, width=args.width).to(args.device)
    else:
        model = ArchCADPrimitiveSemantic(in_features=in_features, dim=args.dim, heads=args.heads,
                                         encoder_layers=args.encoder_layers).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    steps_per_epoch = int(np.ceil(len(train) / args.batch_size))
    # The train-side metric is diagnostic; the gate is on validation. Evaluating it over the
    # whole split costs minutes per point because the geometric grouper runs per drawing
    # (measured: 11.3 s per 1,024 drawings, against 0.6 s for the model forward). The cap is
    # a prefix of the same seeded selection, so the subset is identical across runs.
    train_eval, train_eval_gt = capped_evaluation_set(train, train_gt, args.eval_train_images)
    config = {**vars(args), 'data_root': str(args.data_root), 'output_dir': str(args.output_dir),
              'device': args.device, 'model': model.config, 'split_fingerprint': train_ds.split_fingerprint,
              'parameter_count': sum(x.numel() for x in model.parameters()),
              'torch_version': str(torch.__version__),
              'device_name': torch.cuda.get_device_name() if args.device.startswith('cuda') else 'cpu',
              'train': [{'image_id': i['image_id'], **i['provenance']} for i in train],
              'val': [{'image_id': i['image_id'], **i['provenance']} for i in val],
              'selection_skips': {'train': train_skips, 'val': val_skips},
              'training': {'batch_images': args.batch_size, 'steps_per_epoch': steps_per_epoch,
                           'total_steps': steps_per_epoch * args.epochs, 'precision': 'fp32',
                           'train_images_resolved': len(train), 'val_images_resolved': len(val),
                           'train_eval_images': len(train_eval),
                           'gradient_clip': 1., 'optimizer': 'AdamW', 'weight_decay': .01,
                           'scheduler': ('constant; fixed budget' if args.lr_schedule == 'constant' else
                                         'cosine per step from {:.2g} to {:.2g}'.format(
                                             args.lr, args.lr * args.lr_final)),
                           'loss': 'length-weighted cross entropy'},
              'selection_rule': 'seeded random eligibility under primitive/query caps; no target truncation',
              'protocol': 'per_primitive_semantic_v1; instances from geometric grouping',
              'grouping': {'proximity': args.proximity, 'connectivity': args.connectivity,
                           'provenance': ('proximity None means the fitted per-class table in '
                                          'src/data/archcad_geometric_grouping.py, adopted '
                                          '2026-09-20; a float forces one radius for every class '
                                          'and reproduces the pre-2026-09-20 behaviour')},
              'determinism': determinism}
    hashed = [Path(__file__).resolve(),
              Path('src/models/archcad_semantic_head.py'),
              Path('src/data/archcad_geometric_grouping.py'),
              Path('src/data/archcad_panoptic_dataset.py'),
              Path('src/data/archcad_panoptic.py'),
              Path('src/evaluation/archcad_panoptic.py')]
    if args.rich_features:
        hashed.append(Path('src/data/archcad_rich_features.py'))
    config['implementation_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                       for path in hashed}
    (args.output_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    start = time.monotonic()
    initial_train = evaluate_semantic(model, train_eval, train_eval_gt, args.eval_batch_size,
                                      args.proximity, args.connectivity, args.device)
    initial_val = evaluate_semantic(model, val, val_gt, args.eval_batch_size,
                                    args.proximity, args.connectivity, args.device)
    (args.output_dir / 'evaluation_000.json').write_text(
        json.dumps({'epoch': 0, 'train': initial_train, 'val': initial_val}, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'initial_train': compact(initial_train), 'initial_val': compact(initial_val),
                      'steps_per_epoch': steps_per_epoch, 'train_images': len(train),
                      'val_images': len(val), 'in_features': in_features}), flush=True)
    history = []
    evaluations = [{'epoch': 0, 'train': compact(initial_train), 'val': compact(initial_val)}]
    total_steps = steps_per_epoch * args.epochs
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        batches = training_order(train, args.batch_size, args.seed, epoch, args.batch_order == 'length-grouped')
        losses, seen = [], 0
        for batch_indices in batches:
            chunk = [train[index] for index in batch_indices]
            batch = pad_semantic_batch(chunk, args.device)
            if args.lr_schedule != 'constant':
                optimizer.param_groups[0]['lr'] = learning_rate_at(
                    step, total_steps, args.lr, args.lr_schedule, args.lr_final)
            step += 1
            optimizer.zero_grad(set_to_none=True)
            loss = semantic_classification_loss(model(batch['features'], batch['padding_mask']), batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite semantic loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()) * len(chunk))
            seen += len(chunk)
        history.append({'epoch': epoch, 'training_loss': sum(losses) / seen})
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(json.dumps(history[-1]), flush=True)
        if evaluation_due(epoch, args.epochs, args.eval_every):
            final_train = evaluate_semantic(model, train_eval, train_eval_gt, args.eval_batch_size,
                                            args.proximity, args.connectivity, args.device)
            final_val = evaluate_semantic(model, val, val_gt, args.eval_batch_size,
                                          args.proximity, args.connectivity, args.device)
            evaluations.append({'epoch': epoch, 'train': compact(final_train), 'val': compact(final_val)})
            (args.output_dir / f'evaluation_{epoch:03d}.json').write_text(
                json.dumps({'epoch': epoch, 'train': final_train, 'val': final_val}, indent=2) + '\n', encoding='utf-8')
            (args.output_dir / 'progress.json').write_text(
                json.dumps({'history': history, 'evaluations': evaluations}, indent=2) + '\n', encoding='utf-8')
            torch.save(checkpoint_payload(model, optimizer, epoch, config), args.output_dir / f'epoch_{epoch:03d}.pt')
            print(json.dumps({'evaluation': evaluations[-1]}), flush=True)
    report = {'schema_version': 'archcad-semantic-head-v1', 'config': config, 'history': history,
              'initial_train': initial_train, 'initial_val': initial_val,
              'final_train': final_train, 'final_val': final_val,
              'elapsed_seconds': time.monotonic() - start, 'evaluations': evaluations,
              'interpretation': 'per-primitive semantic head with geometric grouping; not paper-comparable'}
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    torch.save(checkpoint_payload(model, optimizer, args.epochs, config), args.output_dir / 'last.pt')
    print(json.dumps({'final_train': compact(final_train), 'final_val': compact(final_val),
                      'elapsed_seconds': report['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
