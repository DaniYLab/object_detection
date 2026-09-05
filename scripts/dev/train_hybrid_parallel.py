"""Train Hybrid Parallel Detector: Fast + Accurate strategy.

Phase 1: Baseline (parallel only, no reasoning)
- Train parallel decoder with all 35 classes
- Measure speed and baseline accuracy

Phase 2: Add selective reasoning
- Load teacher features
- Enable confidence-gated reasoning refinement
- Measure improvement
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from src.data.dataset import FloorPlanQueryDataset
from src.models.hybrid_parallel_detector import HybridParallelDetector
from src.models.hybrid_backbone import build_hybrid_backbone
from src.training.losses import focal_loss, l1_loss_masked


class HybridLoss(nn.Module):
    """Loss for hybrid parallel detector."""

    def __init__(
        self,
        heatmap_weight: float = 10.0,
        size_weight: float = 1.0,
        offset_weight: float = 1.0,
    ):
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.size_weight = size_weight
        self.offset_weight = offset_weight

    def forward(self, pred: dict, target: dict, class_id: int) -> dict:
        """Compute loss for single class (legacy query mode).

        Args:
            pred: Model output with heatmap, size, offset
            target: Ground truth with same keys
            class_id: Current class being trained

        Returns:
            Loss dict
        """
        # Heatmap loss
        heatmap_loss = focal_loss(
            pred=None,
            target=target['heatmap'],
            logits=pred['heatmap'],
            alpha=2.0,
            beta=4.0,
        )

        # Size/offset loss at positive locations
        positive_mask = target['heatmap'] > 0.5
        size_loss = l1_loss_masked(pred['size'], target['size'], positive_mask)
        offset_loss = l1_loss_masked(pred['offset'], target['offset'], positive_mask)

        total_loss = (
            self.heatmap_weight * heatmap_loss
            + self.size_weight * size_loss
            + self.offset_weight * offset_loss
        )

        return {
            'loss': total_loss,
            'heatmap_loss': heatmap_loss,
            'size_loss': size_loss,
            'offset_loss': offset_loss,
        }


def load_reasoning_features(data_root: str):
    """Load teacher reasoning features for train_new."""
    features_path = Path(data_root) / 'Teacher' / 'train_new_reasoning_features.npz'

    if not features_path.exists():
        print(f"[WARNING] Reasoning features not found: {features_path}")
        return None

    data = np.load(features_path, allow_pickle=True)

    # Build image_id -> embedding lookup
    embeddings = data['embeddings']  # [N, 256]
    image_ids = data['image_ids']

    lookup = {}
    for idx, img_id in enumerate(image_ids):
        lookup[str(img_id)] = torch.from_numpy(embeddings[idx]).float()

    print(f"[OK] Loaded reasoning features: {len(lookup)} images")
    return lookup


def collate_with_reasoning(batch, reasoning_lookup=None):
    """Collate batch with optional reasoning features."""
    # Stack batch
    images = torch.stack([item['image'] for item in batch])
    center_heatmaps = torch.stack([item['center_heatmap'] for item in batch])
    size_maps = torch.stack([item['size_map'] for item in batch])
    offset_maps = torch.stack([item['offset_map'] for item in batch])
    class_ids = [item['class_id'] for item in batch]
    image_ids = [item['image_id'] for item in batch]

    # Load reasoning features if available
    reasoning_embeddings = None
    if reasoning_lookup is not None:
        embeddings = []
        for img_id in image_ids:
            if img_id in reasoning_lookup:
                embeddings.append(reasoning_lookup[img_id])
            else:
                # Fallback: zero embedding
                embeddings.append(torch.zeros(256))
        reasoning_embeddings = torch.stack(embeddings)

    return {
        'image': images,
        'center_heatmap': center_heatmaps,
        'size_map': size_maps,
        'offset_map': offset_maps,
        'class_ids': class_ids,
        'image_ids': image_ids,
        'reasoning_embedding': reasoning_embeddings,
    }


def train_one_epoch(
    model: nn.Module,
    criterion: HybridLoss,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_reasoning: bool = False,
):
    """Train one epoch."""
    model.train()

    total_loss = 0.0
    total_heatmap = 0.0
    total_size = 0.0
    total_offset = 0.0
    total_ambiguous = 0.0
    num_batches = 0

    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        image = batch['image'].to(device)
        target_heatmap = batch['center_heatmap'].to(device)
        target_size = batch['size_map'].to(device)
        target_offset = batch['offset_map'].to(device)
        class_id = batch['class_ids'][0]

        reasoning_embedding = None
        if use_reasoning and batch['reasoning_embedding'] is not None:
            reasoning_embedding = batch['reasoning_embedding'].to(device)

        # Forward
        output = model(
            image=image,
            class_id=class_id,
            reasoning_embedding=reasoning_embedding,
        )

        # Compute loss
        loss_dict = criterion(
            pred=output,
            target={
                'heatmap': target_heatmap,
                'size': target_size,
                'offset': target_offset,
            },
            class_id=class_id,
        )

        # Backward
        optimizer.zero_grad()
        loss_dict['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate
        total_loss += loss_dict['loss'].item()
        total_heatmap += loss_dict['heatmap_loss'].item()
        total_size += loss_dict['size_loss'].item()
        total_offset += loss_dict['offset_loss'].item()
        total_ambiguous += output.get('ambiguous_rate', 0.0)
        num_batches += 1

        # Log
        if batch_idx % 100 == 0:
            elapsed = time.time() - start_time
            ambig_pct = output.get('ambiguous_rate', 0.0) * 100
            print(
                f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                f"Loss {loss_dict['loss'].item():.4f} | "
                f"Ambiguous {ambig_pct:.1f}% | "
                f"Time {elapsed:.1f}s"
            )

    return {
        'loss': total_loss / num_batches,
        'heatmap_loss': total_heatmap / num_batches,
        'size_loss': total_size / num_batches,
        'offset_loss': total_offset / num_batches,
        'ambiguous_rate': total_ambiguous / num_batches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data_root', type=str, default='data/FloorPlanCAD_original')
    parser.add_argument('--output_dir', type=str, default='checkpoints/hybrid_parallel')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_reasoning = (args.phase == 2)

    print("=" * 60)
    print("Hybrid Parallel Detector Training")
    print("=" * 60)
    print(f"Phase: {args.phase}")
    print(f"Device: {device}")
    print("=" * 60)

    # Build model
    backbone = build_hybrid_backbone(feature_dim=256)
    model = HybridParallelDetector(
        backbone=backbone,
        feature_dim=256,
        num_classes=35,
        grid_size=64,
        confidence_threshold=0.7,
        use_reasoning=use_reasoning,
    )
    model = model.to(device)

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    # Load reasoning
    reasoning_lookup = None
    if use_reasoning:
        reasoning_lookup = load_reasoning_features(args.data_root)

    # Dataset
    dataset = FloorPlanQueryDataset(
        root=args.data_root,
        split='train_new',
        manifest_path=f'{args.data_root}/splits_v2.json',
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: collate_with_reasoning(batch, reasoning_lookup),
    )

    criterion = HybridLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        metrics = train_one_epoch(
            model=model,
            criterion=criterion,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            use_reasoning=use_reasoning,
        )

        print(f"Epoch {epoch}: Loss {metrics['loss']:.4f}")

        checkpoint_path = output_dir / f"phase{args.phase}_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'phase': args.phase,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
        }, checkpoint_path)

        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            best_path = output_dir / f"phase{args.phase}_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': metrics,
            }, best_path)

    print("Training Complete!")


if __name__ == '__main__':
    main()
