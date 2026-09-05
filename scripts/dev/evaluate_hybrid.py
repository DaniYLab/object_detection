"""Evaluate hybrid inference detector and measure fallback rate."""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import FloorPlanQueryDataset, query_collate_fn
from src.models.dual_stream_centernet import build_dual_stream_centernet
from train_hybrid_phase2 import HybridInferenceDetector


def decode_predictions(output, threshold=0.05, top_k=100):
    """Decode heatmap predictions to boxes."""
    heatmap = torch.sigmoid(output['heatmap'])
    size = output['size']
    offset = output['offset']

    B, _, H, W = heatmap.shape

    # Peak detection
    maxpool = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
    peaks = (heatmap == maxpool) & (heatmap > threshold)

    boxes = []
    scores = []

    for b in range(B):
        peak_coords = peaks[b, 0].nonzero()

        if len(peak_coords) == 0:
            continue

        for coord in peak_coords[:top_k]:
            y, x = coord
            score = heatmap[b, 0, y, x].item()

            w = size[b, 0, y, x].item()
            h = size[b, 1, y, x].item()

            dx = offset[b, 0, y, x].item()
            dy = offset[b, 1, y, x].item()

            # Convert to image coordinates
            cx = (x + dx) * 8
            cy = (y + dy) * 8

            x1 = max(0, cx - w/2)
            y1 = max(0, cy - h/2)
            x2 = min(512, cx + w/2)
            y2 = min(512, cy + h/2)

            boxes.append([x1, y1, x2, y2])
            scores.append(score)

    return boxes, scores


def evaluate(model, dataloader, device, num_classes=35):
    """Evaluate model and collect hybrid inference stats."""
    model.eval()

    all_predictions = []
    fallback_stats = {
        'total': 0,
        'fallback': 0,
        'parallel': 0,
        'by_class': {i: {'total': 0, 'fallback': 0} for i in range(num_classes)}
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            image = batch['image'].to(device)
            stroke_tokens = batch['stroke_tokens'].to(device)
            stroke_mask = batch['stroke_mask'].to(device)
            class_id = batch['class_ids'][0]

            output = model(image, stroke_tokens, stroke_mask, class_id=class_id)

            # Collect stats
            fallback_stats['total'] += 1
            fallback_stats['by_class'][class_id]['total'] += 1

            if output.get('fallback_used', False):
                fallback_stats['fallback'] += 1
                fallback_stats['by_class'][class_id]['fallback'] += 1
            else:
                fallback_stats['parallel'] += 1

            # Decode predictions
            boxes, scores = decode_predictions(output)

            all_predictions.append({
                'image_id': batch['image_ids'][0],
                'class_id': class_id,
                'boxes': boxes,
                'scores': scores,
                'fallback_used': output.get('fallback_used', False),
            })

            if batch_idx % 100 == 0:
                fallback_rate = fallback_stats['fallback'] / max(1, fallback_stats['total'])
                print(f"Batch {batch_idx}/{len(dataloader)} | Fallback rate: {fallback_rate:.2%}")

    return all_predictions, fallback_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='val_select')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--confidence_threshold', type=float, default=0.7)
    parser.add_argument('--ambiguity_threshold', type=float, default=8.0)
    parser.add_argument('--data_root', type=str, default='data/FloorPlanCAD_original')
    parser.add_argument('--output', type=str, default='outputs/hybrid_evaluation.json')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("Hybrid Inference Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    print(f"Ambiguity threshold: {args.ambiguity_threshold}")
    print("=" * 60)

    # Load model
    base_model = build_dual_stream_centernet(preset='small', num_classes=35)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Extract base model state if wrapped
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('base_model.') for k in state_dict.keys()):
        base_state = {k.replace('base_model.', ''): v for k, v in state_dict.items() if k.startswith('base_model.')}
        base_model.load_state_dict(base_state)
    else:
        base_model.load_state_dict(state_dict)

    model = HybridInferenceDetector(
        base_model=base_model,
        confidence_threshold=args.confidence_threshold,
        ambiguity_threshold=args.ambiguity_threshold,
    )
    model = model.to(device)

    print(f"\n[OK] Model loaded")

    # Dataset
    dataset = FloorPlanQueryDataset(
        root=args.data_root,
        split=args.split,
        manifest_path=f'{args.data_root}/splits_v2.json',
        vector_branch=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=query_collate_fn,
    )

    print(f"[OK] Dataset: {len(dataset)} samples\n")

    # Evaluate
    start_time = time.time()
    predictions, stats = evaluate(model, dataloader, device)
    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Total samples: {stats['total']}")
    print(f"Parallel mode: {stats['parallel']} ({stats['parallel']/stats['total']:.2%})")
    print(f"Fallback mode: {stats['fallback']} ({stats['fallback']/stats['total']:.2%})")
    print(f"Time: {elapsed:.1f}s ({stats['total']/elapsed:.1f} img/s)")

    # Per-class fallback rate
    print("\nPer-class fallback rates:")
    class_stats = []
    for class_id, data in stats['by_class'].items():
        if data['total'] > 0:
            rate = data['fallback'] / data['total']
            class_stats.append((class_id, data['total'], rate))

    class_stats.sort(key=lambda x: x[2], reverse=True)
    for class_id, total, rate in class_stats[:10]:
        print(f"  Class {class_id}: {rate:.2%} ({total} samples)")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        'checkpoint': args.checkpoint,
        'split': args.split,
        'config': {
            'confidence_threshold': args.confidence_threshold,
            'ambiguity_threshold': args.ambiguity_threshold,
        },
        'stats': stats,
        'predictions': predictions,
    }

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n[OK] Results saved to {output_path}")


if __name__ == '__main__':
    main()
