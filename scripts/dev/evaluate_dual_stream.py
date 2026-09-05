"""Evaluate DualStreamCenterNet using existing evaluation infrastructure."""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import FloorPlanQueryDataset, query_collate_fn
from src.models.dual_stream_centernet import build_dual_stream_centernet
from src.evaluation.decoder import decode_centernet
from src.evaluation.metrics import evaluate_detections


def collect_predictions_and_targets(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    score_threshold: float = 0.05,
    max_detections: int = 100,
) -> tuple[list, list]:
    """Run inference and collect predictions + targets."""
    model.eval()

    all_predictions = []
    all_targets = []

    print(f"\nRunning inference on {len(dataloader)} batches...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            # Move to device
            image = batch['image'].to(device)
            stroke_tokens = batch['stroke_tokens'].to(device)
            stroke_mask = batch['stroke_mask'].to(device)
            class_id = batch['class_ids'][0]

            # Forward pass
            output = model(image, stroke_tokens, stroke_mask, class_id=class_id)

            # Decode predictions using existing decoder
            # Rename keys to match decoder expectations
            decoder_outputs = {
                'center_heatmap': output['heatmap'],
                'size_map': output['size'],
                'offset_map': output['offset'],
            }

            detections = decode_centernet(
                outputs=decoder_outputs,
                image_ids=batch['image_ids'],
                class_ids=[class_id] * len(batch['image_ids']),
                score_threshold=score_threshold,
                topk=max_detections,
            )

            # Build predictions from detections
            for det in detections:
                all_predictions.append(det)

            # Build ground truth targets
            for i in range(len(batch['image_ids'])):
                image_id = batch['image_ids'][i]
                gt_boxes = batch['boxes'][i]
                # Filter out padding (all zeros)
                valid_mask = (gt_boxes.sum(dim=1) > 0)
                gt_boxes = gt_boxes[valid_mask]
                gt_labels = torch.full((len(gt_boxes),), class_id, dtype=torch.long)

                all_targets.append({
                    'image_id': image_id,
                    'boxes': gt_boxes,
                    'labels': gt_labels,
                })

    return all_predictions, all_targets


def main():
    parser = argparse.ArgumentParser(description='Evaluate DualStreamCenterNet')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--preset', type=str, default='small', choices=['tiny', 'small', 'base'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--split', type=str, default='val_select')
    parser.add_argument('--data_root', type=str, default='data/FloorPlanCAD_original')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--threshold', type=float, default=0.05)
    parser.add_argument('--max_detections', type=int, default=100)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("DualStreamCenterNet Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print(f"Device: {device}")
    print("=" * 60)

    # Build model
    model = build_dual_stream_centernet(preset=args.preset, num_classes=35)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"\n[OK] Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Dataset
    dataset = FloorPlanQueryDataset(
        root=args.data_root,
        split=args.split,
        manifest_path=f'{args.data_root}/splits_v2.json',
        vector_branch=True,
    )
    print(f"[OK] Dataset: {len(dataset)} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=query_collate_fn,
    )

    # Collect predictions and targets
    start_time = time.time()
    predictions, targets = collect_predictions_and_targets(
        model=model,
        dataloader=dataloader,
        device=device,
        score_threshold=args.threshold,
        max_detections=args.max_detections,
    )

    # Aggregate predictions and targets by image_id
    print("\nAggregating predictions by image...")
    from collections import defaultdict

    pred_by_image = defaultdict(lambda: {'boxes': [], 'labels': [], 'scores': []})
    target_by_image = defaultdict(lambda: {'boxes': [], 'labels': []})

    for pred in predictions:
        img_id = pred['image_id']
        pred_by_image[img_id]['boxes'].append(pred['boxes'])
        pred_by_image[img_id]['labels'].append(pred['labels'])
        pred_by_image[img_id]['scores'].append(pred['scores'])

    for target in targets:
        img_id = target['image_id']
        target_by_image[img_id]['boxes'].append(target['boxes'])
        target_by_image[img_id]['labels'].append(target['labels'])

    # Convert to final format
    final_predictions = []
    for img_id, data in pred_by_image.items():
        final_predictions.append({
            'image_id': img_id,
            'boxes': torch.cat(data['boxes']) if data['boxes'] else torch.empty((0, 4)),
            'labels': torch.cat(data['labels']) if data['labels'] else torch.empty((0,), dtype=torch.long),
            'scores': torch.cat(data['scores']) if data['scores'] else torch.empty((0,)),
        })

    final_targets = []
    for img_id, data in target_by_image.items():
        final_targets.append({
            'image_id': img_id,
            'boxes': torch.cat(data['boxes']) if data['boxes'] else torch.empty((0, 4)),
            'labels': torch.cat(data['labels']) if data['labels'] else torch.empty((0,), dtype=torch.long),
        })

    print(f"Aggregated into {len(final_predictions)} unique images")

    # Evaluate using existing metrics
    print("\nComputing metrics...")
    metrics = evaluate_detections(final_predictions, final_targets)
    eval_time = time.time() - start_time

    # Extract key metrics
    ap50 = metrics.get('AP@0.50', 0.0)
    ap50_95 = metrics.get('AP@0.50:0.95', 0.0)

    # Save results
    results = {
        'model': 'DualStreamCenterNet',
        'preset': args.preset,
        'checkpoint': args.checkpoint,
        'split': args.split,
        'threshold': args.threshold,
        'max_detections': args.max_detections,
        'eval_time_seconds': eval_time,
        'AP50': ap50,
        'AP50:95': ap50_95,
        'full_metrics': metrics,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"AP50: {ap50:.4f}")
    print(f"AP50:95: {ap50_95:.4f}")
    print(f"Time: {eval_time:.1f}s")
    print(f"\nSaved to: {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
