"""Training script for FloorPlanLLM.

Stage 1: Pretrain on detection task using FloorPlanCAD dataset.
Convert from parallel box output to CenterNet-style heatmaps for compatibility.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.svg_dataset import FloorPlanSVGDataset
from src.models.floorplan_llm import build_floorplan_llm
from src.loss.focal_loss import FocalLoss


class ParallelToCenterNetAdapter(nn.Module):
    """Convert parallel box predictions to CenterNet-style heatmaps.

    Parallel output: [B, num_classes, 5] → (x, y, w, h, conf)
    CenterNet output: [B, num_classes, H, W, 5] → (heatmap, w, h, offset_x, offset_y)
    """

    def __init__(self, num_classes: int = 35, output_size: int = 64):
        super().__init__()
        self.num_classes = num_classes
        self.output_size = output_size

    def forward(self, parallel_output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Convert parallel predictions to CenterNet format.

        Args:
            parallel_output: {
                'boxes': [B, num_classes, 4],  # normalized (x, y, w, h)
                'confidence': [B, num_classes],
            }

        Returns:
            {
                'heatmap': [B, num_classes, H, W],
                'size': [B, num_classes, 2, H, W],  # (w, h)
                'offset': [B, num_classes, 2, H, W],  # (offset_x, offset_y)
            }
        """
        boxes = parallel_output['boxes']  # [B, num_classes, 4]
        confidence = parallel_output['confidence']  # [B, num_classes]

        B, C = confidence.shape
        H = W = self.output_size

        device = boxes.device

        # Initialize outputs
        heatmap = torch.zeros(B, C, H, W, device=device)
        size_map = torch.zeros(B, C, 2, H, W, device=device)
        offset_map = torch.zeros(B, C, 2, H, W, device=device)

        # Convert normalized boxes to heatmap coordinates
        # boxes: (cx, cy, w, h) in [0, 1]
        cx = boxes[..., 0] * (W - 1)  # [B, C]
        cy = boxes[..., 1] * (H - 1)  # [B, C]
        w = boxes[..., 2] * W  # [B, C]
        h = boxes[..., 3] * H  # [B, C]

        # Get integer center positions
        cx_int = cx.long().clamp(0, W - 1)  # [B, C]
        cy_int = cy.long().clamp(0, H - 1)  # [B, C]

        # Compute offsets
        offset_x = cx - cx_int.float()  # [B, C]
        offset_y = cy - cy_int.float()  # [B, C]

        # Create Gaussian heatmaps
        # Use confidence as peak value
        conf_prob = torch.sigmoid(confidence)  # [B, C]

        for b in range(B):
            for c in range(C):
                if conf_prob[b, c] < 0.1:  # Skip low-confidence predictions
                    continue

                cx_i = cx_int[b, c].item()
                cy_i = cy_int[b, c].item()

                # Gaussian radius based on box size
                radius = max(1, int((w[b, c] + h[b, c]) / 4))

                # Draw Gaussian
                self._draw_gaussian(
                    heatmap[b, c],
                    cx_i,
                    cy_i,
                    radius,
                    value=conf_prob[b, c].item(),
                )

                # Set size and offset at center position
                size_map[b, c, 0, cy_i, cx_i] = w[b, c]
                size_map[b, c, 1, cy_i, cx_i] = h[b, c]
                offset_map[b, c, 0, cy_i, cx_i] = offset_x[b, c]
                offset_map[b, c, 1, cy_i, cx_i] = offset_y[b, c]

        return {
            'heatmap': heatmap,
            'size': size_map,
            'offset': offset_map,
        }

    def _draw_gaussian(
        self,
        heatmap: torch.Tensor,
        cx: int,
        cy: int,
        radius: int,
        value: float = 1.0,
    ):
        """Draw 2D Gaussian on heatmap (in-place).

        Args:
            heatmap: [H, W]
            cx, cy: center position
            radius: Gaussian radius
            value: peak value
        """
        H, W = heatmap.shape
        diameter = 2 * radius + 1
        sigma = diameter / 6  # 3-sigma coverage

        # Create Gaussian kernel
        y_range = torch.arange(max(0, cy - radius), min(H, cy + radius + 1), device=heatmap.device)
        x_range = torch.arange(max(0, cx - radius), min(W, cx + radius + 1), device=heatmap.device)

        yy, xx = torch.meshgrid(y_range, x_range, indexing='ij')

        # Gaussian formula
        gaussian = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        gaussian = gaussian * value

        # Max pooling (keep existing high values)
        y_start = max(0, cy - radius)
        y_end = min(H, cy + radius + 1)
        x_start = max(0, cx - radius)
        x_end = min(W, cx + radius + 1)

        heatmap[y_start:y_end, x_start:x_end] = torch.maximum(
            heatmap[y_start:y_end, x_start:x_end],
            gaussian,
        )


class CenterNetLoss(nn.Module):
    """CenterNet-style loss for training FloorPlanLLM."""

    def __init__(self, heatmap_weight: float = 10.0, size_weight: float = 1.0, offset_weight: float = 1.0):
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.size_weight = size_weight
        self.offset_weight = offset_weight

        self.focal_loss = FocalLoss(alpha=2.0, beta=4.0)

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute loss.

        Args:
            pred: {
                'heatmap': [B, C, H, W],
                'size': [B, C, 2, H, W],
                'offset': [B, C, 2, H, W],
            }
            target: same structure as pred

        Returns:
            {
                'loss': scalar,
                'heatmap_loss': scalar,
                'size_loss': scalar,
                'offset_loss': scalar,
            }
        """
        # Heatmap loss (focal loss)
        heatmap_loss = self.focal_loss(pred['heatmap'], target['heatmap'])

        # Size loss (L1 loss on positive locations)
        positive_mask = target['heatmap'] > 0.5  # [B, C, H, W]
        positive_mask_expanded = positive_mask.unsqueeze(2)  # [B, C, 1, H, W]

        if positive_mask.sum() > 0:
            size_loss = F.l1_loss(
                pred['size'] * positive_mask_expanded.float(),
                target['size'] * positive_mask_expanded.float(),
                reduction='sum',
            ) / positive_mask.sum()

            offset_loss = F.l1_loss(
                pred['offset'] * positive_mask_expanded.float(),
                target['offset'] * positive_mask_expanded.float(),
                reduction='sum',
            ) / positive_mask.sum()
        else:
            size_loss = torch.tensor(0.0, device=pred['size'].device)
            offset_loss = torch.tensor(0.0, device=pred['offset'].device)

        # Total loss
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
    adapter: ParallelToCenterNetAdapter,
    criterion: CenterNetLoss,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    """Train one epoch."""
    model.train()
    adapter.train()

    total_loss = 0.0
    total_heatmap_loss = 0.0
    total_size_loss = 0.0
    total_offset_loss = 0.0

    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        image = batch['image'].to(device)
        stroke_tokens = batch['stroke_tokens'].to(device)
        stroke_mask = batch['stroke_mask'].to(device)
        # TODO: targets from batch

        # Forward
        parallel_output = model(image, stroke_tokens, stroke_mask)
        pred_centernet = adapter(parallel_output)

        # TODO: Compute loss against GT
        # For now, skip actual training until we have ground truth targets

        optimizer.zero_grad()
        # loss_dict['loss'].backward()
        # optimizer.step()

        # Logging
        if batch_idx % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | Time {elapsed:.1f}s")

    return {
        'loss': total_loss / len(dataloader),
        'heatmap_loss': total_heatmap_loss / len(dataloader),
        'size_loss': total_size_loss / len(dataloader),
        'offset_loss': total_offset_loss / len(dataloader),
    }


def main():
    parser = argparse.ArgumentParser(description='Train FloorPlanLLM')
    parser.add_argument('--preset', type=str, default='small', choices=['tiny', 'small', 'base', 'large'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', type=str, default='checkpoints/floorplan_llm')
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FloorPlanLLM Training - Stage 1: Detection Pretraining")
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

    adapter = ParallelToCenterNetAdapter(num_classes=35, output_size=64)
    adapter = adapter.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Model parameters: {total_params:,}")

    # Loss and optimizer
    criterion = CenterNetLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("\n✓ Training setup complete")
    print("\nNOTE: Ground truth target generation not yet implemented.")
    print("Next steps:")
    print("  1. Implement CenterNet-style ground truth generation")
    print("  2. Connect to existing FloorPlanSVGDataset")
    print("  3. Run full training loop")


if __name__ == '__main__':
    main()
