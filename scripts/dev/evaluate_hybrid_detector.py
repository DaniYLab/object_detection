"""Evaluate Hybrid Parallel Detector on val_select with latency profiling.

Measures both accuracy (AP50/AP50:95) and inference speed (all-class latency).
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import FloorPlanQueryDataset, query_collate_fn
from src.models.hybrid_parallel_detector import build_hybrid_parallel_detector
from src.evaluation.metrics import evaluate_detection


def decode_parallel_predictions(
    heatmaps: torch.Tensor,
    sizes: torch.Tensor,
    offsets: torch.Tensor,
    threshold: float = 0.05,
    top_k: int = 100,
) -> list[dict]:
    """Decode parallel predictions for all classes.

    Args:
        heatmaps: [B, num_classes, H, W]
        sizes: [B, num_classes, 2, H, W]
        offsets: [B, num_classes, 2, H, W]
        threshold: confidence threshold
        top_k: max detections per class

    Returns:
        List of predictions per image
    """
    B, num_classes, H, W = heatmaps.shape
    stride = 512 // H  # Assume 512x512 input

    all_predictions = []

    for b in range(B):
        image_preds = []

        for c in range(num_classes):
            # Get heatmap and apply sigmoid
            heatmap = torch.sigmoid(heatmaps[b, c])  # [H, W]

            # Find peaks above threshold
            scores = heatmap.view(-1)
            indices = torch.where(scores > threshold)[0]

            if len(indices) == 0:
                continue

            # Get top-k
            top_scores, top_idx = torch.topk(
                scores[indices],
                min(top_k, len(indices)),
            )
            top_indices = indices[top_idx]

            # Convert to coordinates
            ys = top_indices // W
            xs = top_indices % W

            # Get sizes and offsets
            ws = sizes[b, c, 0, ys, xs]
            hs = sizes[b, c, 1, ys, xs]
            offset_xs = offsets[b, c, 0, ys, xs]
            offset_ys = offsets[b, c, 1, ys, xs]

            # Convert to image coordinates
            centers_x = (xs.float() + offset_xs) * stride
            centers_y = (ys.float() + offset_ys) * stride

            # Convert to boxes [x1, y1, x2, y2]
            x1 = centers_x - ws / 2
            y1 = centers_y - hs / 2
            x2 = centers_x + ws / 2
            y2 = centers_y + hs / 2

            boxes = torch.stack([x1, y1, x2, y2], dim=1)

            # Add to predictions
            for i in range(len(boxes)):
                image_preds.append({
                    'box': boxes[i].cpu().numpy(),
                    'score': top_scores[i].item(),
                    'class_id': c,
                })

        all_predictions.append(image_preds)

    return all_predictions


def profile_latency(
    model: torch.nn.Module,
    device: torch.device,
    num_classes: int = 35,
    num_warmup: int = 10,
    num_runs: int = 100,
) -> dict:
    """Profile all-class inference latency.

    Args:
        model: trained model
        device: torch device
        num_classes: number of classes
        num_warmup: warmup iterations
        num_runs: measurement iterations

    Returns:
        dict with latency statistics
    """
    model.eval()

    # Dummy input
    dummy_input = torch.randn(1, 3, 512, 512).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    # Synchronize before timing
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Measure
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)

    latencies = torch.tensor(latencies) * 1000  # Convert to ms

    return {
        'mean_ms': latencies.mean().item(),
        'median_ms': latencies.median().item(),
        'std_ms': latencies.std().item(),
        'min_ms': latencies.min().item(),
        'max_ms': latencies.max().item(),
        'throughput_ims_per_sec': 1000.0 / latencies.median().item(),
    }


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.05,
    top_k: int = 100,
) -> dict:
    """Evaluate model on dataset."""
    model.eval()

    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            image = batch['image'].to(device)
            boxes = batch['boxes']  # List of [N, 4] per image
            labels = batch['labels']  # List of [N] per image

            # Forward (parallel all classes)
            output = model(image)

            # Decode predictions
            batch_preds = decode_parallel_predictions(
                heatmaps=output['heatmap'],
                sizes=output['size'],
                offsets=output['offset'],
                threshold=threshold,
                top_k=top_k,
            )

            # Collect predictions and ground truths
            for b in range(len(image)):
                all_predictions.append(batch_preds[b])

                # Ground truth
                gt_boxes = boxes[b].numpy() if len(boxes[b]) > 0 else []
                gt_labels = labels[b].numpy() if len(labels[b]) > 0 else []

                gts = []
                for i in range(len(gt_boxes)):
                    gts.append({
                        'box': gt_boxes[i],
                        'class_id': gt_labels[i],
                    })
                all_ground_truths.append(gts)

    # Compute metrics
    metrics = evaluate_detection(
        predictions=all_predictions,
        ground_truths=all_ground_truths,
        num_classes=35,
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate Hybrid Parallel Detector')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--phase', type=str, choices=['phase1', 'phase2'], required=True)
    parser.add_argument('--data_root', type=str, default='data/FloorPlanCAD_original')
    parser.add_argument('--output_path', type=str, default='outputs/hybrid_evaluation.json')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--threshold', type=float, default=0.05)
    parser.add_argument('--profile_latency', action='store_true', help='Profile inference speed')
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print(f"Hybrid Parallel Detector Evaluation - {args.phase.upper()}")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print("=" * 60)

    # Build model
    use_reasoning = (args.phase == 'phase2')
    model = build_hybrid_parallel_detector(num_classes=35, use_reasoning=use_reasoning)
    model = model.to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n[OK] Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Model: {total_params:,} params")

    # Latency profiling
    if args.profile_latency:
        print("\n" + "=" * 60)
        print("Latency Profiling")
        print("=" * 60)
        latency_stats = profile_latency(model, device)
        print(f"Mean: {latency_stats['mean_ms']:.2f} ms")
        print(f"Median: {latency_stats['median_ms']:.2f} ms")
        print(f"Throughput: {latency_stats['throughput_ims_per_sec']:.1f} img/s")
    else:
        latency_stats = None

    # Dataset
    print("\n[OK] Loading val_select...")
    dataset = FloorPlanQueryDataset(
        root=args.data_root,
        split='val_select',
        manifest_path=f'{args.data_root}/splits_v2.json',
        vector_branch=False,
    )
    print(f"[OK] Dataset: {len(dataset)} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=query_collate_fn,
    )

    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluating on val_select")
    print("=" * 60)
    metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        threshold=args.threshold,
        top_k=100,
    )

    print(f"\nResults:")
    print(f"  AP50: {metrics['AP50']:.4f}")
    print(f"  AP50:95: {metrics['AP50:95']:.4f}")

    # Save results
    output = {
        'phase': args.phase,
        'checkpoint': str(args.checkpoint),
        'metrics': metrics,
        'latency': latency_stats,
        'threshold': args.threshold,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n[OK] Results saved to {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
