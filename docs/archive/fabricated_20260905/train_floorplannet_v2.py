"""Training script for FloorPlanNet v2.1 with Knowledge Distillation

Training strategy:
- Stage 1 (20 epochs): Distillation from thin-head teacher
- Stage 2 (50 epochs): Fine-tuning with ground truth only
- Mixed precision (FP16) for 2× speedup
- Progressive learning rate schedule
- Multi-scale training augmentation

Expected timeline: 70 epochs, ~3-4 days on single GPU

Usage:
    python train_floorplannet_v2.py \\
        --teacher checkpoints/dual_pathway_v2_thin_seed1337/best.pt \\
        --epochs 70 \\
        --batch_size 12 \\
        --output_dir checkpoints/floorplannet_v2
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from src.data.dataset import FloorPlanDataset, collate_batch
from src.data.splits import load_split_manifest
from src.models.unified_v2 import FloorPlanNetV2
from src.training.losses import FocalLoss, L1Loss


class DistillationLoss(nn.Module):
    """Knowledge distillation loss for heatmap predictions.

    Combines:
    - Hard loss: Focal loss on ground truth
    - Soft loss: KL divergence from teacher predictions
    """

    def __init__(
        self,
        alpha: float = 0.5,
        temperature: float = 4.0,
        focal_alpha: float = 2.0,
        focal_beta: float = 4.0,
    ):
        super().__init__()
        self.alpha = alpha  # Balance between hard and soft loss
        self.temperature = temperature  # Temperature for softening logits

        self.focal_loss = FocalLoss(alpha=focal_alpha, beta=focal_beta)

    def forward(
        self,
        student_heatmap: torch.Tensor,
        teacher_heatmap: torch.Tensor,
        gt_heatmap: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute distillation loss.

        Args:
            student_heatmap: [B, C, H, W] student predictions
            teacher_heatmap: [B, C, H, W] teacher predictions (no grad)
            gt_heatmap: [B, C, H, W] ground truth

        Returns:
            Dict with 'total', 'hard', 'soft' losses
        """
        # Hard loss: Ground truth supervision
        hard_loss = self.focal_loss(student_heatmap, gt_heatmap)

        # Soft loss: KL divergence from teacher
        T = self.temperature

        # Soften predictions with temperature
        student_soft = torch.sigmoid(student_heatmap / T)
        teacher_soft = torch.sigmoid(teacher_heatmap / T)

        # KL divergence (binary)
        kl_pos = teacher_soft * torch.log(teacher_soft / (student_soft + 1e-7) + 1e-7)
        kl_neg = (1 - teacher_soft) * torch.log((1 - teacher_soft) / (1 - student_soft + 1e-7) + 1e-7)
        soft_loss = (kl_pos + kl_neg).mean() * (T ** 2)  # Scale by T^2

        # Combined loss
        total_loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss

        return {
            'total': total_loss,
            'hard': hard_loss,
            'soft': soft_loss,
        }


def load_teacher_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load thin-head teacher model for distillation."""
    print(f"Loading teacher from {checkpoint_path}")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Import teacher model (thin-head)
    from src.models.dual_pathway_centernet import create_dual_pathway_centernet

    teacher = create_dual_pathway_centernet(
        num_classes=35,
        preset='thin',
    ).to(device)

    teacher.load_state_dict(ckpt['model_state_dict'])
    teacher.eval()

    # Freeze teacher
    for param in teacher.parameters():
        param.requires_grad = False

    print(f"Teacher loaded: {sum(p.numel() for p in teacher.parameters()) / 1e6:.2f}M params")

    return teacher


def train_epoch(
    model: nn.Module,
    teacher: nn.Module | None,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    use_distillation: bool = True,
    distillation_alpha: float = 0.5,
) -> dict[str, float]:
    """Train one epoch with optional distillation."""
    model.train()

    # Losses
    focal_loss_fn = FocalLoss(alpha=2.0, beta=4.0)
    l1_loss_fn = L1Loss()

    if use_distillation and teacher is not None:
        distill_loss_fn = DistillationLoss(alpha=distillation_alpha, temperature=4.0)

    total_loss = 0.0
    total_heatmap_loss = 0.0
    total_size_loss = 0.0
    total_offset_loss = 0.0
    total_distill_soft = 0.0
    num_batches = 0

    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        images = batch['image'].to(device)
        strokes = batch['strokes'].to(device)
        stroke_mask = batch['stroke_mask'].to(device)

        gt_heatmap = batch['heatmap'].to(device)
        gt_size = batch['size'].to(device)
        gt_offset = batch['offset'].to(device)
        gt_heatmap_mask = batch['heatmap_mask'].to(device)

        # Forward pass with mixed precision
        optimizer.zero_grad()

        with autocast():
            # Student predictions
            outputs = model(images, strokes, stroke_mask)
            pred_heatmap = outputs['heatmap']
            pred_size = outputs['size']
            pred_offset = outputs['offset']

            # Compute losses
            if use_distillation and teacher is not None:
                # Get teacher predictions (no grad)
                with torch.no_grad():
                    teacher_outputs = teacher(images, strokes, stroke_mask)
                    teacher_heatmap = teacher_outputs['heatmap']

                # Distillation loss
                distill_losses = distill_loss_fn(pred_heatmap, teacher_heatmap, gt_heatmap)
                heatmap_loss = distill_losses['total']
                distill_soft_loss = distill_losses['soft'].item()
            else:
                # Standard focal loss
                heatmap_loss = focal_loss_fn(pred_heatmap, gt_heatmap)
                distill_soft_loss = 0.0

            # Size and offset losses (standard L1)
            size_loss = l1_loss_fn(pred_size, gt_size, gt_heatmap_mask)
            offset_loss = l1_loss_fn(pred_offset, gt_offset, gt_heatmap_mask)

            # Total loss
            loss = 10.0 * heatmap_loss + 1.0 * size_loss + 1.0 * offset_loss

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Accumulate metrics
        total_loss += loss.item()
        total_heatmap_loss += heatmap_loss.item()
        total_size_loss += size_loss.item()
        total_offset_loss += offset_loss.item()
        total_distill_soft += distill_soft_loss
        num_batches += 1

        # Log progress
        if (batch_idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch} | Batch {batch_idx+1}/{len(dataloader)} | "
                f"Loss {loss.item():.4f} | Heatmap {heatmap_loss.item():.4f} | "
                f"Time {elapsed:.1f}s"
            )

    # Average losses
    metrics = {
        'loss': total_loss / num_batches,
        'heatmap_loss': total_heatmap_loss / num_batches,
        'size_loss': total_size_loss / num_batches,
        'offset_loss': total_offset_loss / num_batches,
    }

    if use_distillation:
        metrics['distill_soft_loss'] = total_distill_soft / num_batches

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train FloorPlanNet v2.1')

    # Model args
    parser.add_argument('--num_classes', type=int, default=35)
    parser.add_argument('--preset', type=str, default='default', choices=['default', 'fast', 'small'])

    # Training args
    parser.add_argument('--epochs', type=int, default=70, help='Total epochs')
    parser.add_argument('--distill_epochs', type=int, default=20, help='Epochs with distillation')
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_epochs', type=int, default=3)

    # Distillation args
    parser.add_argument('--teacher', type=str, default=None, help='Teacher checkpoint path')
    parser.add_argument('--distillation_alpha', type=float, default=0.5, help='Distillation weight')

    # Data args
    parser.add_argument('--data_root', type=str, default='data/FloorPlanCAD_original')
    parser.add_argument('--manifest', type=str, default='data/FloorPlanCAD_original/splits_v2.json')
    parser.add_argument('--image_size', type=int, default=512)

    # Output args
    parser.add_argument('--output_dir', type=str, default='checkpoints/floorplannet_v2')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')

    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("="*80)
    print("FloorPlanNet v2.1 Training")
    print("="*80)
    print(f"Output: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs} (distillation: {args.distill_epochs})")
    print(f"Batch size: {args.batch_size}")
    print("="*80)

    # Device
    device = torch.device(args.device)

    # Load teacher (if distillation enabled)
    teacher = None
    if args.teacher and args.distill_epochs > 0:
        teacher = load_teacher_model(args.teacher, device)

    # Create model
    print("Creating FloorPlanNet v2.1...")
    from src.models.unified_v2 import create_floorplannet_v2
    model = create_floorplannet_v2(
        num_classes=args.num_classes,
        preset=args.preset,
    ).to(device)

    param_counts = model.get_num_params()
    print(f"Model parameters:")
    print(f"  Encoder: {param_counts['encoder'] / 1e6:.2f}M")
    print(f"  Decoder: {param_counts['decoder'] / 1e6:.2f}M")
    print(f"  Total: {param_counts['total'] / 1e6:.2f}M")

    # Estimate FLOPs
    flops = model.estimate_flops()
    print(f"\nEstimated FLOPs: {flops['total']:.2f} GFLOPs")

    # Load dataset
    print("\nLoading dataset...")
    manifest = load_split_manifest(args.manifest)

    train_dataset = FloorPlanDataset(
        data_root=args.data_root,
        split='train_new',
        manifest=manifest,
        image_size=args.image_size,
        augment=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=args.warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.warmup_epochs,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs],
    )

    # Mixed precision scaler
    scaler = GradScaler()

    # Training loop
    print("\nStarting training...")
    print("="*80)

    for epoch in range(1, args.epochs + 1):
        # Determine if using distillation
        use_distillation = (epoch <= args.distill_epochs) and (teacher is not None)

        stage = "Distillation" if use_distillation else "Fine-tuning"
        print(f"\n[Epoch {epoch}/{args.epochs}] Stage: {stage}")

        # Train one epoch
        metrics = train_epoch(
            model=model,
            teacher=teacher,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            use_distillation=use_distillation,
            distillation_alpha=args.distillation_alpha,
        )

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Log epoch summary
        print(f"\nEpoch {epoch}/{args.epochs} Summary:")
        print(f"  Train Loss: {metrics['loss']:.4f}")
        print(f"  Heatmap Loss: {metrics['heatmap_loss']:.4f}")
        print(f"  Size Loss: {metrics['size_loss']:.4f}")
        print(f"  Offset Loss: {metrics['offset_loss']:.4f}")
        if 'distill_soft_loss' in metrics:
            print(f"  Distill Soft Loss: {metrics['distill_soft_loss']:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")

        # Save checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            checkpoint_path = output_dir / f'epoch_{epoch:03d}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': metrics,
                'config': vars(args),
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

    print("\n" + "="*80)
    print("Training complete!")
    print(f"Final checkpoint: {output_dir / f'epoch_{args.epochs:03d}.pt'}")
    print("="*80)


if __name__ == '__main__':
    main()
