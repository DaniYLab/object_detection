"""Simplified training for FloorPlanLLM - direct parallel box supervision.

Instead of converting parallel → CenterNet → parallel, we train directly on boxes.
Loss: IoU loss + confidence loss (binary cross-entropy).
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.floorplan_llm_dataset import FloorPlanLLMDataset
from src.models.floorplan_llm import build_floorplan_llm


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of boxes.

    Args:
        boxes1: [N, 4] (x, y, w, h) normalized
        boxes2: [M, 4] (x, y, w, h) normalized

    Returns:
        [N, M] IoU matrix
    """
    # Convert to corner format (x1, y1, x2, y2)
    boxes1_corners = torch.zeros_like(boxes1)
    boxes1_corners[:, 0] = boxes1[:, 0] - boxes1[:, 2] / 2  # x1
    boxes1_corners[:, 1] = boxes1[:, 1] - boxes1[:, 3] / 2  # y1
    boxes1_corners[:, 2] = boxes1[:, 0] + boxes1[:, 2] / 2  # x2
    boxes1_corners[:, 3] = boxes1[:, 1] + boxes1[:, 3] / 2  # y2

    boxes2_corners = torch.zeros_like(boxes2)
    boxes2_corners[:, 0] = boxes2[:, 0] - boxes2[:, 2] / 2
    boxes2_corners[:, 1] = boxes2[:, 1] - boxes2[:, 3] / 2
    boxes2_corners[:, 2] = boxes2[:, 0] + boxes2[:, 2] / 2
    boxes2_corners[:, 3] = boxes2[:, 1] + boxes2[:, 3] / 2

    # Compute intersection
    lt = torch.max(boxes1_corners[:, None, :2], boxes2_corners[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    # Compute union
    area1 = boxes1[:, 2] * boxes1[:, 3]  # [N]
    area2 = boxes2[:, 2] * boxes2[:, 3]  # [M]
    union = area1[:, None] + area2 - inter  # [N, M]

    # IoU
    iou = inter / (union + 1e-6)
    return iou


class ParallelBoxLoss(nn.Module):
    """Direct loss on parallel box predictions.

    Loss = λ1 * IoU_loss + λ2 * BCE_loss
    """

    def __init__(self, iou_weight: float = 1.0, bce_weight: float = 0.5):
        super().__init__()
        self.iou_weight = iou_weight
        self.bce_weight = bce_weight

    def forward(
        self,
        pred_boxes: torch.Tensor,
        pred_conf: torch.Tensor,
        gt_boxes: list[torch.Tensor],
        gt_classes: list[torch.Tensor],
        num_classes: int = 35,
    ) -> dict[str, torch.Tensor]:
        """Compute loss.

        Args:
            pred_boxes: [B, num_classes, 4]
            pred_conf: [B, num_classes]
            gt_boxes: list of [N_i, 4] for each image
            gt_classes: list of [N_i] class indices for each image

        Returns:
            {
                'loss': scalar,
                'iou_loss': scalar,
                'bce_loss': scalar,
            }
        """
        batch_size = pred_boxes.shape[0]
        device = pred_boxes.device

        total_iou_loss = 0.0
        total_bce_loss = 0.0
        num_positives = 0

        for b in range(batch_size):
            if len(gt_boxes[b]) == 0:
                # No objects in this image
                # Only penalize false positives via BCE
                target_conf = torch.zeros(num_classes, device=device)
                bce_loss = F.binary_cross_entropy_with_logits(
                    pred_conf[b],
                    target_conf,
                    reduction='mean',
                )
                total_bce_loss += bce_loss
                continue

            # Match predictions to ground truth via Hungarian matching
            # Simplified: use greedy matching based on IoU

            gt_box = gt_boxes[b]  # [N, 4]
            gt_cls = gt_classes[b]  # [N]

            # Create target confidence (0 for all, then set 1 for matched classes)
            target_conf = torch.zeros(num_classes, device=device)

            iou_loss_per_class = 0.0

            for gt_idx in range(len(gt_box)):
                cls = gt_cls[gt_idx].item()

                # Predicted box for this class
                pred_box_cls = pred_boxes[b, cls]  # [4]
                gt_box_cls = gt_box[gt_idx]  # [4]

                # IoU loss (1 - IoU)
                iou = box_iou(pred_box_cls.unsqueeze(0), gt_box_cls.unsqueeze(0))[0, 0]
                iou_loss_per_class += (1.0 - iou)

                # Mark this class as positive
                target_conf[cls] = 1.0

                num_positives += 1

            total_iou_loss += iou_loss_per_class

            # Confidence loss
            bce_loss = F.binary_cross_entropy_with_logits(
                pred_conf[b],
                target_conf,
                reduction='mean',
            )
            total_bce_loss += bce_loss

        # Aggregate
        if num_positives > 0:
            total_iou_loss = total_iou_loss / num_positives
        else:
            total_iou_loss = torch.tensor(0.0, device=device)

        total_bce_loss = total_bce_loss / batch_size

        loss = self.iou_weight * total_iou_loss + self.bce_weight * total_bce_loss

        return {
            'loss': loss,
            'iou_loss': total_iou_loss,
            'bce_loss': total_bce_loss,
        }


def collate_fn(batch):
    """Custom collate to handle variable-length boxes and strokes."""
    images = torch.stack([item['image'] for item in batch])

    # Handle variable-length stroke tokens (need padding)
    stroke_tokens_list = [item['stroke_tokens'] for item in batch]
    stroke_mask_list = [item['stroke_mask'] for item in batch]

    # Find max length in this batch
    max_strokes = max(st.shape[0] for st in stroke_tokens_list)

    # Pad stroke tokens
    stroke_tokens_padded = []
    stroke_mask_padded = []
    for st, sm in zip(stroke_tokens_list, stroke_mask_list):
        n_strokes = st.shape[0]
        if n_strokes < max_strokes:
            # Pad with zeros
            pad_size = max_strokes - n_strokes
            st_padded = torch.cat([st, torch.zeros(pad_size, 12)], dim=0)
            sm_padded = torch.cat([sm, torch.zeros(pad_size, dtype=torch.bool)], dim=0)
        else:
            st_padded = st
            sm_padded = sm
        stroke_tokens_padded.append(st_padded)
        stroke_mask_padded.append(sm_padded)

    stroke_tokens = torch.stack(stroke_tokens_padded)
    stroke_mask = torch.stack(stroke_mask_padded)

    # Ground truth boxes (variable length per image)
    gt_boxes = [item['boxes'] for item in batch]  # list of [N_i, 4]
    gt_classes = [item['classes'] for item in batch]  # list of [N_i]

    return {
        'image': images,
        'stroke_tokens': stroke_tokens,
        'stroke_mask': stroke_mask,
        'gt_boxes': gt_boxes,
        'gt_classes': gt_classes,
    }


def train_one_epoch(
    model: nn.Module,
    criterion: ParallelBoxLoss,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    """Train one epoch."""
    model.train()

    total_loss = 0.0
    total_iou_loss = 0.0
    total_bce_loss = 0.0
    num_batches = 0

    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        image = batch['image'].to(device)
        stroke_tokens = batch['stroke_tokens'].to(device)
        stroke_mask = batch['stroke_mask'].to(device)
        gt_boxes = [boxes.to(device) for boxes in batch['gt_boxes']]
        gt_classes = [classes.to(device) for classes in batch['gt_classes']]

        # Forward
        output = model(image, stroke_tokens, stroke_mask)

        # Compute loss
        loss_dict = criterion(
            output['boxes'],
            output['confidence'],
            gt_boxes,
            gt_classes,
        )

        # Backward
        optimizer.zero_grad()
        loss_dict['loss'].backward()
        optimizer.step()

        # Accumulate
        total_loss += loss_dict['loss'].item()
        total_iou_loss += loss_dict['iou_loss'].item()
        total_bce_loss += loss_dict['bce_loss'].item()
        num_batches += 1

        # Logging
        if batch_idx % 10 == 0:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                f"Loss {loss_dict['loss'].item():.4f} | "
                f"IoU {loss_dict['iou_loss'].item():.4f} | "
                f"BCE {loss_dict['bce_loss'].item():.4f} | "
                f"Time {elapsed:.1f}s"
            )

    return {
        'loss': total_loss / num_batches,
        'iou_loss': total_iou_loss / num_batches,
        'bce_loss': total_bce_loss / num_batches,
    }


def main():
    parser = argparse.ArgumentParser(description='Train FloorPlanLLM (simplified)')
    parser.add_argument('--preset', type=str, default='tiny', choices=['tiny', 'small', 'base', 'large'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data_root', type=str, default='data/FloorPlanCAD_original')
    parser.add_argument('--output_dir', type=str, default='checkpoints/floorplan_llm')
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FloorPlanLLM Training - Simplified Direct Supervision")
    print("=" * 60)
    print(f"Preset: {args.preset}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Build model
    model = build_floorplan_llm(preset=args.preset, num_classes=35)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Model parameters: {total_params:,}")

    # Dataset (use FloorPlanLLMDataset)
    print("\n✓ Loading dataset...")
    try:
        dataset = FloorPlanLLMDataset(
            root=args.data_root,
            split='train_new',
            manifest_path=f'{args.data_root}/splits_v2.json',
        )
        print(f"✓ Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Start with 0 for debugging
        collate_fn=collate_fn,
    )

    # Loss and optimizer
    criterion = ParallelBoxLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"\n✓ Training setup complete")
    print(f"  Batches per epoch: {len(dataloader)}")
    print(f"  Total iterations: {len(dataloader) * args.epochs}")

    # Training loop
    print("\n" + "=" * 60)
    print("Starting Training")
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
        print(f"  IoU Loss: {metrics['iou_loss']:.4f}")
        print(f"  BCE Loss: {metrics['bce_loss']:.4f}")

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
