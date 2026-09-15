"""Evaluation script for FloorPlanNet v2.1

Evaluates model on validation/test splits with:
- Standard AP metrics (AP50, AP50:95)
- Speed benchmarking
- Stride-4 refinement analysis
- Confidence monitoring statistics
- Per-class breakdown

Usage:
    python evaluate_floorplannet_v2.py \\
        --checkpoint checkpoints/floorplannet_v2/epoch_070.pt \\
        --split val_select \\
        --output outputs/eval_floorplannet_v2.json
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import FloorPlanDataset, collate_batch
from src.data.splits import load_split_manifest
from src.models.unified_v2 import FloorPlanNetV2, create_floorplannet_v2
from src.evaluation.metrics import compute_ap_metrics


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    confidence_threshold: float = 0.3,
) -> dict:
    """Evaluate model and collect predictions."""
    model.eval()

    all_predictions = []
    all_targets = []

    confidence_stats = {
        'max_confidences': [],
        'low_confidence_count': 0,
        'stride4_used_count': 0,
    }

    total_time = 0.0
    num_images = 0

    print("Evaluating...")
    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        strokes = batch['strokes'].to(device)
        stroke_mask = batch['stroke_mask'].to(device)

        # Forward pass with timing
        start_time = time.time()
        outputs = model(images, strokes, stroke_mask, return_routing=True)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        elapsed = time.time() - start_time

        total_time += elapsed
        num_images += images.shape[0]

        # Collect confidence statistics
        if 'max_confidence' in outputs:
            confidence_stats['max_confidences'].append(outputs['max_confidence'].item())
        if outputs.get('low_confidence', False):
            confidence_stats['low_confidence_count'] += 1
        if outputs.get('stride4_used', False):
            confidence_stats['stride4_used_count'] += 1

        # Extract predictions (post-process heatmaps to boxes)
        pred_heatmap = outputs['heatmap']
        pred_size = outputs['size']
        pred_offset = outputs['offset']

        # Simple peak extraction (max pooling + NMS)
        # NOTE: This is simplified - use proper NMS in production
        batch_predictions = extract_predictions(
            pred_heatmap,
            pred_size,
            pred_offset,
            stride=outputs['stride'],
            confidence_threshold=confidence_threshold,
        )

        all_predictions.extend(batch_predictions)

        # Collect ground truth
        gt_boxes = batch['boxes']  # List of [N, 4] per image
        gt_labels = batch['labels']  # List of [N] per image

        for boxes, labels in zip(gt_boxes, gt_labels):
            all_targets.append({
                'boxes': boxes.cpu().numpy(),
                'labels': labels.cpu().numpy(),
            })

        # Progress
        if (batch_idx + 1) % 50 == 0:
            avg_time = total_time / num_images
            print(f"  Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Avg time: {avg_time*1000:.1f}ms/image")

    # Compute metrics
    metrics = compute_ap_metrics(all_predictions, all_targets, num_classes=35)

    # Add speed and confidence stats
    metrics['speed'] = {
        'total_time_sec': total_time,
        'num_images': num_images,
        'avg_time_ms': (total_time / num_images) * 1000,
        'throughput_fps': num_images / total_time,
    }

    metrics['confidence_stats'] = {
        'mean_max_confidence': sum(confidence_stats['max_confidences']) / len(confidence_stats['max_confidences']) if confidence_stats['max_confidences'] else 0.0,
        'low_confidence_rate': confidence_stats['low_confidence_count'] / num_images,
        'stride4_used_rate': confidence_stats['stride4_used_count'] / num_images,
    }

    return metrics


def extract_predictions(
    heatmap: torch.Tensor,
    size: torch.Tensor,
    offset: torch.Tensor,
    stride: int = 8,
    confidence_threshold: float = 0.3,
    max_detections_per_class: int = 100,
) -> list[dict]:
    """Extract box predictions from heatmap outputs.

    Simple version - extracts peaks via max pooling.

    Args:
        heatmap: [B, C, H, W] class heatmaps
        size: [B, 2, H, W] box sizes
        offset: [B, 2, H, W] sub-pixel offsets
        stride: Output stride
        confidence_threshold: Minimum confidence
        max_detections_per_class: Max detections per class

    Returns:
        List of predictions per image
    """
    B, C, H, W = heatmap.shape
    heatmap = torch.sigmoid(heatmap)

    predictions = []

    for b in range(B):
        boxes_list = []
        scores_list = []
        labels_list = []

        for c in range(C):
            heat = heatmap[b, c]  # [H, W]

            # Find peaks (simple: top-k)
            heat_flat = heat.flatten()
            topk_values, topk_indices = torch.topk(
                heat_flat,
                k=min(max_detections_per_class, heat_flat.numel()),
            )

            # Filter by threshold
            valid_mask = topk_values > confidence_threshold
            topk_values = topk_values[valid_mask]
            topk_indices = topk_indices[valid_mask]

            if len(topk_indices) == 0:
                continue

            # Convert indices to coordinates
            y_coords = (topk_indices // W).float()
            x_coords = (topk_indices % W).float()

            # Apply offsets
            dx = offset[b, 0, y_coords.long(), x_coords.long()]
            dy = offset[b, 1, y_coords.long(), x_coords.long()]
            x_coords = x_coords + dx
            y_coords = y_coords + dy

            # Get box sizes
            w = size[b, 0, y_coords.long(), x_coords.long()]
            h = size[b, 1, y_coords.long(), x_coords.long()]

            # Convert to image coordinates
            cx = x_coords * stride
            cy = y_coords * stride
            w = w * stride
            h = h * stride

            # Convert to [x0, y0, x1, y1]
            x0 = cx - w / 2
            y0 = cy - h / 2
            x1 = cx + w / 2
            y1 = cy + h / 2

            boxes = torch.stack([x0, y0, x1, y1], dim=1)  # [N, 4]

            boxes_list.append(boxes)
            scores_list.append(topk_values)
            labels_list.append(torch.full((len(topk_values),), c, dtype=torch.long))

        if boxes_list:
            predictions.append({
                'boxes': torch.cat(boxes_list, dim=0).cpu().numpy(),
                'scores': torch.cat(scores_list, dim=0).cpu().numpy(),
                'labels': torch.cat(labels_list, dim=0).cpu().numpy(),
            })
        else:
            predictions.append({
                'boxes': torch.zeros((0, 4)).cpu().numpy(),
                'scores': torch.zeros((0,)).cpu().numpy(),
                'labels': torch.zeros((0,), dtype=int).cpu().numpy(),
            })

    return predictions


def main():
    parser = argparse.ArgumentParser(description='Evaluate FloorPlanNet v2.1')

    # Model args
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--preset', type=str, default='default', help='Model preset')

    # Data args
    parser.add_argument('--data_root', type=str, default='data/FloorPlanCAD_original')
    parser.add_argument('--manifest', type=str, default='data/FloorPlanCAD_original/splits_v2.json')
    parser.add_argument('--split', type=str, default='val_select', help='Split to evaluate')
    parser.add_argument('--image_size', type=int, default=512)

    # Evaluation args
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--confidence_threshold', type=float, default=0.3)

    # Output
    parser.add_argument('--output', type=str, default='outputs/eval_floorplannet_v2.json')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    print("="*80)
    print("FloorPlanNet v2.1 Evaluation")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print(f"Device: {args.device}")
    print("="*80)

    device = torch.device(args.device)

    # Load model
    print("Loading model...")
    model = create_floorplannet_v2(num_classes=35, preset=args.preset).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    param_counts = model.get_num_params()
    print(f"Model parameters: {param_counts['total'] / 1e6:.2f}M")

    # Load dataset
    print("Loading dataset...")
    manifest = load_split_manifest(args.manifest)

    eval_dataset = FloorPlanDataset(
        data_root=args.data_root,
        split=args.split,
        manifest=manifest,
        image_size=args.image_size,
        augment=False,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=True,
    )

    print(f"Evaluation samples: {len(eval_dataset)}")

    # Evaluate
    metrics = evaluate_model(
        model=model,
        dataloader=eval_loader,
        device=device,
        confidence_threshold=args.confidence_threshold,
    )

    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"AP50: {metrics['AP50']:.4f}")
    print(f"AP50:95: {metrics['AP50:95']:.4f}")
    print(f"\nSpeed:")
    print(f"  Avg time: {metrics['speed']['avg_time_ms']:.1f} ms/image")
    print(f"  Throughput: {metrics['speed']['throughput_fps']:.1f} FPS")
    print(f"\nConfidence Stats:")
    print(f"  Mean max confidence: {metrics['confidence_stats']['mean_max_confidence']:.3f}")
    print(f"  Low confidence rate: {metrics['confidence_stats']['low_confidence_rate']:.2%}")
    print(f"  Stride-4 used rate: {metrics['confidence_stats']['stride4_used_rate']:.2%}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("="*80)


if __name__ == '__main__':
    main()
