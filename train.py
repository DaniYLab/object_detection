"""
Training script for FloorPlanCAD Detector.

Usage:
    python train.py --config configs/default.yaml
    python train.py  # uses defaults
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from src.data.dataset import FloorPlanDataset, collate_fn, CLASS_NAMES, NUM_CLASSES
from src.models.detector import FloorPlanDetector


# ── Loss Functions ─────────────────────────────────────────────────────────────

def focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Binary Focal Loss for heatmap prediction.
    pred, target: [B, C, H, W] in [0, 1]
    """
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    pt = torch.where(target == 1, pred, 1 - pred)
    focal_weight = alpha * (1 - pt) ** gamma
    return (focal_weight * bce).mean()


def dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Dice Loss for mask overlap quality.
    pred, target: [B, C, H, W] in [0, 1]
    """
    pred_flat = pred.view(pred.shape[0], pred.shape[1], -1)
    tgt_flat  = target.view(target.shape[0], target.shape[1], -1)
    intersection = (pred_flat * tgt_flat).sum(-1)
    union = pred_flat.sum(-1) + tgt_flat.sum(-1)
    dice = (2 * intersection + smooth) / (union + smooth)
    return (1 - dice).mean()


def detection_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    focal_w: float = 1.0,
    dice_w: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Combined Focal + Dice loss."""
    # Downsample target to match pred spatial size
    if pred.shape[-2:] != target.shape[-2:]:
        target = F.interpolate(target, size=pred.shape[-2:], mode="bilinear", align_corners=False)

    fl = focal_loss(pred, target)
    dl = dice_loss(pred, target)
    total = focal_w * fl + dice_w * dl
    return {"total": total, "focal": fl, "dice": dl}


# ── Fake tokenizer (placeholder until T5 integrated) ──────────────────────────

def tokenize_texts(texts: list[str], max_len: int = 32, vocab_size: int = 32000) -> torch.Tensor:
    """Stub tokenizer — replace with real T5/CLIP tokenizer."""
    tokens = []
    for text in texts:
        ids = [hash(word) % (vocab_size - 1) + 1 for word in text.lower().split()]
        ids = ids[:max_len] + [0] * max(0, max_len - len(ids))
        tokens.append(ids)
    return torch.tensor(tokens, dtype=torch.long)


# ── Training Loop ──────────────────────────────────────────────────────────────

def train_one_epoch(
    model: FloorPlanDetector,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 20,
) -> dict[str, float]:
    model.train()
    total_loss = focal_sum = dice_sum = 0.0
    t0 = time.time()

    for step, batch in enumerate(loader):
        image   = batch["image"].to(device)           # [B, 3, H, W]
        heatmap = batch["heatmap"].to(device)         # [B, 35, H, W]
        texts_batch = batch["texts"]                  # list[list[str]]
        class_ids_batch = batch["class_ids"]          # list[list[int]]

        B = image.shape[0]

        # For each sample, pick the first class_id as primary conditioning
        # (full multi-class handled by running all present classes)
        primary_cls = torch.tensor(
            [ids[0] if ids else 0 for ids in class_ids_batch],
            dtype=torch.long, device=device,
        )

        # Tokenize first text per sample
        first_texts = [t[0] if t else "Find object in this floor plan" for t in texts_batch]
        text_ids = tokenize_texts(first_texts).to(device)   # [B, 32]

        # Forward
        optimizer.zero_grad()
        pred = model(image, text_ids, primary_cls)          # [B, 35, h, w]

        # Loss
        losses = detection_loss(pred, heatmap)
        losses["total"].backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += losses["total"].item()
        focal_sum  += losses["focal"].item()
        dice_sum   += losses["dice"].item()

        if (step + 1) % log_interval == 0:
            elapsed = time.time() - t0
            avg_loss = total_loss / (step + 1)
            print(
                f"  Epoch {epoch} | Step {step+1:4d}/{len(loader)} | "
                f"Loss {avg_loss:.4f} "
                f"(focal={focal_sum/(step+1):.4f}, dice={dice_sum/(step+1):.4f}) | "
                f"{elapsed:.1f}s"
            )

    n = len(loader)
    return {
        "loss":  total_loss / n,
        "focal": focal_sum  / n,
        "dice":  dice_sum   / n,
    }


@torch.no_grad()
def validate(
    model: FloorPlanDetector,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    iou_sum = 0.0
    n = 0

    for batch in loader:
        image   = batch["image"].to(device)
        heatmap = batch["heatmap"].to(device)
        texts_batch = batch["texts"]
        class_ids_batch = batch["class_ids"]

        primary_cls = torch.tensor(
            [ids[0] if ids else 0 for ids in class_ids_batch],
            dtype=torch.long, device=device,
        )
        first_texts = [t[0] if t else "Find object in this floor plan" for t in texts_batch]
        text_ids = tokenize_texts(first_texts).to(device)

        pred = model(image, text_ids, primary_cls)
        losses = detection_loss(pred, heatmap)
        total_loss += losses["total"].item()

        # Binary IoU @ threshold 0.5
        pred_bin = (pred > 0.5).float()
        tgt_bin  = F.interpolate(heatmap, size=pred.shape[-2:], mode="nearest")
        inter = (pred_bin * tgt_bin).sum()
        union = (pred_bin + tgt_bin).clamp(0, 1).sum()
        iou_sum += (inter / union.clamp(min=1)).item()
        n += 1

    return {"val_loss": total_loss / n, "val_iou": iou_sum / n}


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = FloorPlanDataset(
        args.data_root, split="train",
        image_size=args.image_size, crop_size=args.crop_size,
    )
    val_ds = FloorPlanDataset(
        args.data_root, split="test",
        image_size=args.image_size, crop_size=args.crop_size,
    )
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )
    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | "
          f"Steps/epoch: {len(train_dl)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = FloorPlanDetector(
        image_size=args.image_size,
        model_dim=args.model_dim,
        num_classes=NUM_CLASSES,
        num_blocks=args.num_blocks,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model params: {n_params:.1f}M")

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ── Training ──────────────────────────────────────────────────────────────
    best_iou = 0.0
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}/{args.epochs}]")
        train_metrics = train_one_epoch(
            model, train_dl, optimizer, device, epoch, args.log_interval
        )
        val_metrics = validate(model, val_dl, device)
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        print(
            f"  => Train loss: {train_metrics['loss']:.4f} | "
            f"Val loss: {val_metrics['val_loss']:.4f} | "
            f"Val IoU: {val_metrics['val_iou']:.4f} | "
            f"LR: {lr_now:.2e}"
        )

        # Save best checkpoint
        if val_metrics["val_iou"] > best_iou:
            best_iou = val_metrics["val_iou"]
            ckpt_path = ckpt_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_iou": best_iou,
                "args": vars(args),
            }, ckpt_path)
            print(f"  => Saved best checkpoint (IoU={best_iou:.4f}) → {ckpt_path}")

        # Save latest checkpoint every 5 epochs
        if epoch % 5 == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_iou": val_metrics["val_iou"],
            }, ckpt_dir / f"epoch_{epoch:03d}.pt")

    print(f"\nTraining done! Best Val IoU: {best_iou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FloorPlanCAD Detector Training")
    parser.add_argument("--data_root",    default="./data/FloorPlanCAD_dataset")
    parser.add_argument("--ckpt_dir",     default="./checkpoints")
    parser.add_argument("--image_size",   type=int,   default=512)
    parser.add_argument("--crop_size",    type=int,   default=128)
    parser.add_argument("--batch_size",   type=int,   default=4)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--model_dim",    type=int,   default=512)
    parser.add_argument("--num_blocks",   type=int,   default=4)
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--log_interval", type=int,   default=20)
    args = parser.parse_args()
    main(args)
