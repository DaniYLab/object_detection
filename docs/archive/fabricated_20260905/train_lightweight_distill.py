"""Train lightweight student with knowledge distillation from thin teacher.

Target:
- Student: 0.64M params
- Teacher: dual_pathway_v2_thin (AP50 0.495, 6.8M params)
- Goal: AP50 >0.45 with 10× fewer params and 5× faster
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import FloorPlanQueryDataset, query_collate_fn
from src.models.lightweight_student import build_lightweight_student, DistillationLoss
from src.models.factory import build_model


def load_teacher(checkpoint_path, device):
    """Load pretrained teacher model."""
    print(f"Loading teacher from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Build from config
    preset = checkpoint.get('preset', 'floorplan_base')
    model_config = checkpoint.get('model_config')

    print(f"Teacher preset: {preset}")
    teacher = build_model(config=model_config)
    teacher.load_state_dict(checkpoint['model_state'])
    teacher = teacher.to(device)
    teacher.eval()

    # Freeze teacher
    for param in teacher.parameters():
        param.requires_grad = False

    print(f"Teacher loaded and frozen")
    return teacher


def train_one_epoch(student, teacher, criterion, dataloader, optimizer, device, epoch):
    """Train one epoch with distillation."""
    student.train()
    teacher.eval()

    total_loss = 0.0
    total_task = 0.0
    total_distill = 0.0
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

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_out = teacher(
                image,
                class_ids=batch['class_ids'],
                stroke_tokens=stroke_tokens,
                stroke_mask=stroke_mask
            )

        # Student forward
        student_out = student(image, class_id=class_id)

        # Compute loss
        loss_dict = criterion(
            student_out,
            teacher_out,
            {
                'heatmap': target_heatmap,
                'size': target_size,
                'offset': target_offset,
            }
        )

        # Backward
        optimizer.zero_grad()
        loss_dict['loss'].backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss_dict['loss'].item()
        total_task += loss_dict['task_loss'].item()
        total_distill += loss_dict['distill_loss'].item()
        num_batches += 1

        if batch_idx % 100 == 0:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                f"Loss {loss_dict['loss'].item():.4f} | "
                f"Task {loss_dict['task_loss'].item():.4f} | "
                f"Distill {loss_dict['distill_loss'].item():.4f} | "
                f"Time {elapsed:.1f}s"
            )

    return {
        'loss': total_loss / num_batches,
        'task_loss': total_task / num_batches,
        'distill_loss': total_distill / num_batches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', type=str, required=True, help='Teacher checkpoint path')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--temperature', type=float, default=4.0)
    parser.add_argument('--alpha', type=float, default=0.5, help='Balance task vs distill loss')
    parser.add_argument('--data_root', type=str, default='data/FloorPlanCAD_original')
    parser.add_argument('--output_dir', type=str, default='checkpoints/lightweight_student')
    parser.add_argument('--save_every', type=int, default=5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Knowledge Distillation Training")
    print("=" * 60)
    print(f"Teacher: {args.teacher}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Temperature: {args.temperature}")
    print(f"Alpha (task/distill): {args.alpha}")
    print("=" * 60)

    # Load teacher
    teacher = load_teacher(args.teacher, device)

    # Build student
    student = build_lightweight_student()
    student = student.to(device)

    # Dataset
    print("\n[OK] Loading training dataset...")
    train_dataset = FloorPlanQueryDataset(
        root=args.data_root,
        split='train_new',
        manifest_path=f'{args.data_root}/splits_v2.json',
        vector_branch=True,  # For teacher
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=query_collate_fn,
    )

    print(f"[OK] Training dataset: {len(train_dataset)} samples")
    print(f"[OK] Batches per epoch: {len(train_dataloader)}")

    # Loss and optimizer
    criterion = DistillationLoss(temperature=args.temperature, alpha=args.alpha)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)

    training_log = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            student=student,
            teacher=teacher,
            criterion=criterion,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch}/{args.epochs} Summary:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  Task Loss: {train_metrics['task_loss']:.4f}")
        print(f"  Distill Loss: {train_metrics['distill_loss']:.4f}")
        print(f"  LR: {current_lr:.6f}")

        training_log.append({
            'epoch': epoch,
            'lr': current_lr,
            'train': train_metrics,
        })

        if epoch % args.save_every == 0:
            checkpoint_path = output_dir / f"epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'args': vars(args),
            }, checkpoint_path)
            print(f"  Saved: {checkpoint_path}")

    # Save final
    final_path = output_dir / 'final.pt'
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': student.state_dict(),
        'args': vars(args),
    }, final_path)

    log_path = output_dir / 'training_log.json'
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final model: {final_path}")
    print(f"Training log: {log_path}")


if __name__ == '__main__':
    main()
