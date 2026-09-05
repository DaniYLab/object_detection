"""Train DualStreamCenterNet with CenterNet loss and FloorPlanCAD dataset."""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import FloorPlanQueryDataset, query_collate_fn
from src.models.dual_stream_centernet import build_dual_stream_centernet
from src.training.losses import focal_loss, l1_loss_masked


class CenterNetLoss(nn.Module):
    """CenterNet loss: focal + L1 for size/offset."""

    def __init__(self, heatmap_weight: float = 10.0, size_weight: float = 1.0, offset_weight: float = 1.0):
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.size_weight = size_weight
        self.offset_weight = offset_weight

    def forward(self, pred: dict, target: dict) -> dict[str, torch.Tensor]:
        """Compute loss."""
        # Heatmap loss (use logits for numerical stability)
        heatmap_loss = focal_loss(pred=None, target=target['heatmap'], logits=pred['heatmap'], alpha=2.0, beta=4.0)

        # Size/offset loss (only at positive locations)
        positive_mask = target['heatmap'] > 0.5

        size_loss = l1_loss_masked(pred['size'], target['size'], positive_mask)
        offset_loss = l1_loss_masked(pred['offset'], target['offset'], positive_mask)

        loss = (
            self.heatmap_weight * heatmap_loss
            + self.size_weight * size_loss
            + self.offset_weight * offset_loss
        )

        return {
            'loss': loss,
            'heatmap_loss': heatmap_loss,
            'size_loss': size_loss,
            'offset_loss': offset_loss,
        }


def train_one_epoch(
    model: nn.Module,
    criterion: CenterNetLoss,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    """Train one epoch."""
    model.train()

    total_loss = 0.0
    total_heatmap = 0.0
    total_size = 0.0
    total_offset = 0.0
    num_batches = 0

    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        image = batch['image'].to(device)
        stroke_tokens = batch['stroke_tokens'].to(device)
        stroke_mask = batch['stroke_mask'].to(device)
        target_heatmap = batch['center_heatmap'].to(device)
        target_size = batch['size_map'].to(device)
        target_offset = batch['offset_map'].to(device)

        # Get class_id (batch has same class due to QueryDataset sampling)
        class_id = batch['class_ids'][0]

        # Forward
        output = model(image, stroke_tokens, stroke_mask, class_id=class_id)

        # Compute loss
        loss_dict = criterion(
            output,
            {
                'heatmap': target_heatmap,
                'size': target_size,
                'offset': target_offset,
            },
        )

        # Backward
        optimizer.zero_grad()
        loss_dict['loss'].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Accumulate
        total_loss += loss_dict['loss'].item()
        total_heatmap += loss_dict['heatmap_loss'].item()
        total_size += loss_dict['size_loss'].item()
        total_offset += loss_dict['offset_loss'].item()
        num_batches += 1

        # Logging
        if batch_idx % 100 == 0:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                f"Loss {loss_dict['loss'].item():.4f} | "
                f"Heatmap {loss_dict['heatmap_loss'].item():.4f} | "
                f"Time {elapsed:.1f}s"
            )

    return {
        'loss': total_loss / num_batches,
        'heatmap_loss': total_heatmap / num_batches,
        'size_loss': total_size / num_batches,
        'offset_loss': total_offset / num_batches,
    }


def main():
    parser = argparse.ArgumentParser(description='Train DualStreamCenterNet')
    parser.add_argument('--preset', type=str, default='small', choices=['tiny', 'small', 'base'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data_root', type=str, default='data/FloorPlanCAD_original')
    parser.add_argument('--output_dir', type=str, default='checkpoints/dual_stream_centernet')
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DualStreamCenterNet Training")
    print("=" * 60)
    print(f"Preset: {args.preset}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60)

    # Build model
    model = build_dual_stream_centernet(preset=args.preset, num_classes=35)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[OK] Model: {args.preset} ({total_params:,} params)")

    # Dataset
    print("\n[OK] Loading dataset...")
    dataset = FloorPlanQueryDataset(
        root=args.data_root,
        split='train_new',
        manifest_path=f'{args.data_root}/splits_v2.json',
        vector_branch=True,
    )
    print(f"[OK] Dataset: {len(dataset)} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=query_collate_fn,
    )

    # Loss and optimizer
    criterion = CenterNetLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"\n[OK] Training setup complete")
    print(f"  Batches per epoch: {len(dataloader)}")

    # Training loop
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        metrics = train_one_epoch(
            model=model,
            criterion=criterion,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Heatmap: {metrics['heatmap_loss']:.4f}")
        print(f"  Size: {metrics['size_loss']:.4f}")
        print(f"  Offset: {metrics['offset_loss']:.4f}")

        # Save checkpoint
        checkpoint_path = output_dir / f"epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }, checkpoint_path)
        print(f"  Saved: {checkpoint_path}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
