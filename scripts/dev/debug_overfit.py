"""Debug script: Test if model can overfit on single batch."""

import torch
from torch.utils.data import DataLoader

from src.data.floorplan_llm_dataset import FloorPlanLLMDataset
from src.models.floorplan_llm import build_floorplan_llm
from train_floorplan_llm_simple import ParallelBoxLoss, collate_fn


def test_overfit_single_batch():
    """Test if model can overfit on a single batch."""
    print("=" * 60)
    print("Testing Overfit on Single Batch")
    print("=" * 60)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Build tiny model
    model = build_floorplan_llm(preset='tiny', num_classes=35)
    model = model.to(device)
    print(f"Model: tiny ({sum(p.numel() for p in model.parameters()):,} params)")

    # Get single batch
    print("\nLoading single batch...")
    dataset = FloorPlanLLMDataset(
        root='data/FloorPlanCAD_original',
        split='train_new',
        manifest_path='data/FloorPlanCAD_original/splits_v2.json',
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
    )

    batch = next(iter(dataloader))
    print(f"Batch size: {batch['image'].shape[0]}")
    print(f"Objects: {[len(boxes) for boxes in batch['gt_boxes']]}")

    # Move to device
    image = batch['image'].to(device)
    stroke_tokens = batch['stroke_tokens'].to(device)
    stroke_mask = batch['stroke_mask'].to(device)
    gt_boxes = [boxes.to(device) for boxes in batch['gt_boxes']]
    gt_classes = [classes.to(device) for classes in batch['gt_classes']]

    # Loss and optimizer
    criterion = ParallelBoxLoss(iou_weight=1.0, bce_weight=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print("\n" + "=" * 60)
    print("Training on Single Batch (Target: Loss < 0.5)")
    print("=" * 60)

    model.train()
    for step in range(1000):
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

        # Check for NaN gradients
        has_nan = False
        for name, param in model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                print(f"\n✗ NaN gradient at step {step} in {name}")
                has_nan = True
                break

        if has_nan:
            print("Stopping due to NaN gradients")
            break

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Logging
        loss_val = loss_dict['loss'].item()
        iou_val = loss_dict['iou_loss'].item()
        bce_val = loss_dict['bce_loss'].item()

        if step % 50 == 0:
            print(f"Step {step:4d} | Loss {loss_val:.4f} | IoU {iou_val:.4f} | BCE {bce_val:.4f}")

        # Check for NaN loss
        if not torch.isfinite(loss_dict['loss']):
            print(f"\n✗ NaN loss at step {step}")
            print(f"  IoU loss: {iou_val}")
            print(f"  BCE loss: {bce_val}")
            break

        # Success criteria
        if loss_val < 0.5:
            print(f"\n✓ Converged! Loss < 0.5 in {step} steps")
            print(f"  Final - Loss: {loss_val:.4f}, IoU: {iou_val:.4f}, BCE: {bce_val:.4f}")
            return True

    print("\n✗ Did not converge after 1000 steps")
    return False


if __name__ == '__main__':
    success = test_overfit_single_batch()
    if success:
        print("\n" + "=" * 60)
        print("✓ Model CAN learn (overfitting successful)")
        print("  → Proceed with full training")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✗ Model CANNOT learn (debugging needed)")
        print("  → Check:")
        print("    1. Model initialization")
        print("    2. Loss function")
        print("    3. Learning rate")
        print("=" * 60)
