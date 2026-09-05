"""Resume training from checkpoint with different batch size."""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.data.dataset import FloorPlanQueryDataset, query_collate_fn
from src.models.dual_stream_centernet import build_dual_stream_centernet
from src.training.losses import focal_loss, l1_loss_masked


class CenterNetLoss(nn.Module):
    def __init__(self, heatmap_weight: float = 10.0, size_weight: float = 1.0, offset_weight: float = 1.0):
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.size_weight = size_weight
        self.offset_weight = offset_weight

    def forward(self, pred: dict, target: dict) -> dict[str, torch.Tensor]:
        heatmap_loss = focal_loss(pred=None, target=target['heatmap'], logits=pred['heatmap'], alpha=2.0, beta=4.0)
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


def train_one_epoch(model, criterion, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    total_heatmap = 0.0
    total_size = 0.0
    total_offset = 0.0
    num_batches = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        image = batch['image'].to(device)
        stroke_tokens = batch['stroke_tokens'].to(device)
        stroke_mask = batch['stroke_mask'].to(device)
        target_heatmap = batch['center_heatmap'].to(device)
        target_size = batch['size_map'].to(device)
        target_offset = batch['offset_map'].to(device)
        class_id = batch['class_ids'][0]

        output = model(image, stroke_tokens, stroke_mask, class_id=class_id)

        loss_dict = criterion(
            output,
            {
                'heatmap': target_heatmap,
                'size': target_size,
                'offset': target_offset,
            },
        )

        optimizer.zero_grad()
        loss_dict['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss_dict['loss'].item()
        total_heatmap += loss_dict['heatmap_loss'].item()
        total_size += loss_dict['size_loss'].item()
        total_offset += loss_dict['offset_loss'].item()
        num_batches += 1

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


def validate(model, criterion, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_heatmap = 0.0
    total_size = 0.0
    total_offset = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            image = batch['image'].to(device)
            stroke_tokens = batch['stroke_tokens'].to(device)
            stroke_mask = batch['stroke_mask'].to(device)
            target_heatmap = batch['center_heatmap'].to(device)
            target_size = batch['size_map'].to(device)
            target_offset = batch['offset_map'].to(device)
            class_id = batch['class_ids'][0]

            output = model(image, stroke_tokens, stroke_mask, class_id=class_id)

            loss_dict = criterion(
                output,
                {
                    'heatmap': target_heatmap,
                    'size': target_size,
                    'offset': target_offset,
                },
            )

            total_loss += loss_dict['loss'].item()
            total_heatmap += loss_dict['heatmap_loss'].item()
            total_size += loss_dict['size_loss'].item()
            total_offset += loss_dict['offset_loss'].item()
            num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'heatmap_loss': total_heatmap / num_batches,
        'size_loss': total_size / num_batches,
        'offset_loss': total_offset / num_batches,
    }


def main():
    parser = argparse.ArgumentParser(description='Resume training from checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--preset', type=str, default='small')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data_root', type=str, default='data/FloorPlanCAD_original')
    parser.add_argument('--output_dir', type=str, default='checkpoints/dual_stream_100ep')
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--val_every', type=int, default=5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Resume Training from Checkpoint")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Total epochs: {args.epochs}")
    print("=" * 60)

    # Load checkpoint
    print(f"\n[OK] Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    start_epoch = checkpoint['epoch'] + 1
    print(f"[OK] Resuming from epoch {start_epoch}")

    # Build model
    model = build_dual_stream_centernet(preset=args.preset, num_classes=35)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Datasets
    print("\n[OK] Loading datasets...")
    train_dataset = FloorPlanQueryDataset(
        root=args.data_root,
        split='train_new',
        manifest_path=f'{args.data_root}/splits_v2.json',
        vector_branch=True,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=query_collate_fn,
    )

    val_dataset = FloorPlanQueryDataset(
        root=args.data_root,
        split='val_select',
        manifest_path=f'{args.data_root}/splits_v2.json',
        vector_branch=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=query_collate_fn,
    )

    print(f"[OK] Training dataset: {len(train_dataset)} samples")
    print(f"[OK] Validation dataset: {len(val_dataset)} samples")
    print(f"[OK] Batches per epoch: {len(train_dataloader)}")

    # Optimizer & scheduler
    criterion = CenterNetLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Reconstruct scheduler state
    remaining_epochs = args.epochs - start_epoch + 1
    scheduler = CosineAnnealingLR(optimizer, T_max=remaining_epochs)

    print(f"\n[OK] Training will continue from epoch {start_epoch} to {args.epochs}")

    # Training loop
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)

    best_val_loss = checkpoint.get('val_metrics', {}).get('loss', float('inf'))
    training_log = []

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            criterion=criterion,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch}/{args.epochs} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Train Heatmap: {train_metrics['heatmap_loss']:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")

        val_metrics = None
        if epoch % args.val_every == 0:
            print(f"\n  Running validation...")
            val_metrics = validate(model, criterion, val_dataloader, device)
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val Heatmap: {val_metrics['heatmap_loss']:.4f}")

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_path = output_dir / 'best_val.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'lr': current_lr,
                }, best_path)
                print(f"  Saved best model: {best_path}")

        log_entry = {
            'epoch': epoch,
            'lr': current_lr,
            'train': train_metrics,
        }
        if val_metrics:
            log_entry['val'] = val_metrics
        training_log.append(log_entry)

        if epoch % args.save_every == 0:
            checkpoint_path = output_dir / f"epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'lr': current_lr,
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

    log_path = output_dir / 'training_log_resumed.json'
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
