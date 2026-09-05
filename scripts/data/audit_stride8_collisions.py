"""Audit stride-8 collision rate and small object statistics."""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import FloorPlanQueryDataset, query_collate_fn


def compute_collision_statistics(
    dataloader: DataLoader,
    output_stride: int = 8,
    feature_size: int = 64,
) -> dict:
    """Compute collision statistics for ground truth centers."""

    total_objects = 0
    total_collisions = 0
    collisions_by_class = defaultdict(lambda: {'total': 0, 'collisions': 0})

    small_objects = 0  # Area < 32x32
    small_collisions = 0

    collision_details = []

    print(f"\nAuditing collisions at stride {output_stride}...")

    for batch in tqdm(dataloader, desc="Processing"):
        # Get ground truth boxes and class
        boxes = batch['boxes']  # [B, max_boxes, 4]
        class_id = batch['class_ids'][0]
        image_ids = batch['image_ids']

        batch_size = boxes.shape[0]

        for b in range(batch_size):
            image_id = image_ids[b]
            gt_boxes = boxes[b]  # [max_boxes, 4]

            # Filter valid boxes (non-padding)
            valid_mask = (gt_boxes.sum(dim=1) > 0)
            gt_boxes = gt_boxes[valid_mask]

            if len(gt_boxes) == 0:
                continue

            # Compute centers in feature space
            centers_x = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0  # [N]
            centers_y = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0  # [N]

            # Convert to feature map coordinates
            feat_x = (centers_x / output_stride).long()
            feat_y = (centers_y / output_stride).long()

            # Clip to feature map bounds
            feat_x = torch.clamp(feat_x, 0, feature_size - 1)
            feat_y = torch.clamp(feat_y, 0, feature_size - 1)

            # Check for collisions: multiple centers map to same pixel
            center_coords = torch.stack([feat_y, feat_x], dim=1)  # [N, 2]
            unique_coords = torch.unique(center_coords, dim=0)

            num_objects = len(gt_boxes)
            num_unique = len(unique_coords)
            num_collisions = num_objects - num_unique

            total_objects += num_objects
            total_collisions += num_collisions

            collisions_by_class[class_id]['total'] += num_objects
            collisions_by_class[class_id]['collisions'] += num_collisions

            # Small object analysis
            widths = gt_boxes[:, 2] - gt_boxes[:, 0]
            heights = gt_boxes[:, 3] - gt_boxes[:, 1]
            areas = widths * heights

            small_mask = areas < (32 * 32)
            num_small = small_mask.sum().item()
            small_objects += num_small

            # Check collisions among small objects only
            if num_small > 0:
                small_centers = center_coords[small_mask]
                unique_small = torch.unique(small_centers, dim=0)
                small_collisions += (num_small - len(unique_small))

            # Record collision events
            if num_collisions > 0:
                collision_details.append({
                    'image_id': image_id,
                    'class_id': class_id,
                    'num_objects': num_objects,
                    'num_collisions': num_collisions,
                    'collision_rate': num_collisions / num_objects,
                })

    # Compute statistics
    overall_collision_rate = total_collisions / total_objects if total_objects > 0 else 0.0
    small_collision_rate = small_collisions / small_objects if small_objects > 0 else 0.0

    per_class_stats = {}
    for class_id, stats in collisions_by_class.items():
        rate = stats['collisions'] / stats['total'] if stats['total'] > 0 else 0.0
        per_class_stats[int(class_id)] = {
            'total_objects': stats['total'],
            'collisions': stats['collisions'],
            'collision_rate': rate,
        }

    return {
        'output_stride': output_stride,
        'feature_size': feature_size,
        'total_objects': total_objects,
        'total_collisions': total_collisions,
        'overall_collision_rate': overall_collision_rate,
        'small_objects': small_objects,
        'small_collisions': small_collisions,
        'small_collision_rate': small_collision_rate,
        'per_class_stats': per_class_stats,
        'collision_events': len(collision_details),
        'worst_collisions': sorted(collision_details, key=lambda x: x['collision_rate'], reverse=True)[:20],
    }


def main():
    parser = argparse.ArgumentParser(description='Audit stride-8 collision rate')
    parser.add_argument('--split', type=str, default='val_select')
    parser.add_argument('--data_root', type=str, default='data/FloorPlanCAD_original')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output', type=str, default='outputs/stride8_collision_audit.json')
    args = parser.parse_args()

    print("=" * 60)
    print("Stride-8 Collision Audit")
    print("=" * 60)
    print(f"Split: {args.split}")
    print(f"Output stride: 8")
    print(f"Feature size: 64x64")
    print("=" * 60)

    # Load dataset
    dataset = FloorPlanQueryDataset(
        root=args.data_root,
        split=args.split,
        manifest_path=f'{args.data_root}/splits_v2.json',
        vector_branch=False,  # Don't need vectors for audit
    )
    print(f"\n[OK] Dataset: {len(dataset)} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=query_collate_fn,
    )

    # Compute statistics
    stats = compute_collision_statistics(dataloader, output_stride=8, feature_size=64)

    # Print summary
    print("\n" + "=" * 60)
    print("Collision Audit Results")
    print("=" * 60)
    print(f"Total objects: {stats['total_objects']}")
    print(f"Total collisions: {stats['total_collisions']}")
    print(f"Overall collision rate: {stats['overall_collision_rate']:.4f} ({stats['overall_collision_rate']*100:.2f}%)")
    print()
    print(f"Small objects (<32px²): {stats['small_objects']}")
    print(f"Small object collisions: {stats['small_collisions']}")
    print(f"Small collision rate: {stats['small_collision_rate']:.4f} ({stats['small_collision_rate']*100:.2f}%)")
    print()

    # Decision gate
    print("=" * 60)
    print("Decision Gate for Stride-4 Head")
    print("=" * 60)

    if stats['overall_collision_rate'] > 0.10:
        decision = "IMPLEMENT"
        reason = f"Overall collision rate {stats['overall_collision_rate']:.2%} exceeds 10% threshold"
    elif stats['small_collision_rate'] > 0.15:
        decision = "IMPLEMENT"
        reason = f"Small object collision rate {stats['small_collision_rate']:.2%} exceeds 15% threshold"
    else:
        decision = "SKIP"
        reason = f"Collision rates are acceptable (overall {stats['overall_collision_rate']:.2%}, small {stats['small_collision_rate']:.2%})"

    print(f"Decision: {decision}")
    print(f"Reason: {reason}")
    print("=" * 60)

    stats['decision'] = decision
    stats['decision_reason'] = reason

    # Top colliding classes
    print("\nTop 10 classes by collision rate:")
    sorted_classes = sorted(
        stats['per_class_stats'].items(),
        key=lambda x: x[1]['collision_rate'],
        reverse=True
    )[:10]

    for class_id, class_stats in sorted_classes:
        print(f"  Class {class_id}: {class_stats['collision_rate']:.4f} "
              f"({class_stats['collisions']}/{class_stats['total_objects']} objects)")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n[OK] Saved audit results: {output_path}")


if __name__ == '__main__':
    main()
