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

def focal_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """Penalty-reduced Focal Loss for CenterNet Gaussian heatmaps."""
    pred = torch.clamp(pred, 1e-4, 1 - 1e-4)
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    
    neg_weights = torch.pow(1 - target, beta)
    
    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds
    
    num_pos = pos_inds.sum()
    if num_pos == 0:
        return -neg_loss.sum()
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos


def l1_loss_masked(pred_size: torch.Tensor, target_size: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L1 loss for size prediction, only computed at object centers."""
    mask = mask.expand_as(pred_size)
    l1 = F.l1_loss(pred_size, target_size, reduction="none")
    return (l1 * mask).sum() / (mask.sum() + 1e-4)


def centernet_loss(
    preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    focal_w: float = 1.0,
    size_w: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Combined Focal + L1 size loss."""
    pred_hm = preds["center_heatmap"]
    pred_sz = preds["size_map"]
    
    tgt_hm = targets["center_heatmap"]
    tgt_sz = targets["size_map"]
    tgt_mask = targets["mask_map"]
    
    # Downsample targets to match pred spatial size
    if pred_hm.shape[-2:] != tgt_hm.shape[-2:]:
        tgt_hm = F.interpolate(tgt_hm, size=pred_hm.shape[-2:], mode="bilinear", align_corners=False)
        tgt_sz = F.interpolate(tgt_sz, size=pred_hm.shape[-2:], mode="nearest")
        tgt_mask = F.interpolate(tgt_mask, size=pred_hm.shape[-2:], mode="nearest")
        tgt_mask = (tgt_mask > 0).float()
    
    fl = focal_loss(pred_hm, tgt_hm)
    l1 = l1_loss_masked(pred_sz, tgt_sz, tgt_mask)
    total = focal_w * fl + size_w * l1
    return {"total": total, "focal": fl, "size_l1": l1}


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
    total_loss = focal_sum = l1_sum = 0.0
    t0 = time.time()

    for step, batch in enumerate(loader):
        image = batch["image"].to(device)
        targets = {
            "center_heatmap": batch["center_heatmap"].to(device),
            "size_map": batch["size_map"].to(device),
            "mask_map": batch["mask_map"].to(device),
        }
        texts_batch = batch["texts"]           # list[str]
        class_ids_batch = batch["class_ids"]    # list[int]

        B = image.shape[0]

        # Class conditioning per sample
        primary_cls = torch.tensor(class_ids_batch, dtype=torch.long, device=device)

        # Tokenize text prompts
        text_ids = tokenize_texts(texts_batch).to(device)   # [B, 32]

        # Forward
        optimizer.zero_grad()
        preds = model(image, text_ids, primary_cls)

        # Loss
        losses = centernet_loss(preds, targets)
        losses["total"].backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += losses["total"].item()
        focal_sum  += losses["focal"].item()
        l1_sum   += losses["size_l1"].item()

        if (step + 1) % log_interval == 0:
            elapsed = time.time() - t0
            avg_loss = total_loss / (step + 1)
            print(
                f"  Epoch {epoch} | Step {step+1:4d}/{len(loader)} | "
                f"Loss {avg_loss:.4f} "
                f"(focal={focal_sum/(step+1):.4f}, size_l1={l1_sum/(step+1):.4f}) | "
                f"{elapsed:.1f}s"
            )

    n = len(loader)
    return {
        "loss":  total_loss / n,
        "focal": focal_sum  / n,
        "size_l1": l1_sum   / n,
    }


@torch.no_grad()
def validate(
    model: FloorPlanDetector,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    n = 0

    for batch in loader:
        image = batch["image"].to(device)
        targets = {
            "center_heatmap": batch["center_heatmap"].to(device),
            "size_map": batch["size_map"].to(device),
            "mask_map": batch["mask_map"].to(device),
        }
        texts_batch = batch["texts"]
        class_ids_batch = batch["class_ids"]

        primary_cls = torch.tensor(class_ids_batch, dtype=torch.long, device=device)
        text_ids = tokenize_texts(texts_batch).to(device)

        preds = model(image, text_ids, primary_cls)
        losses = centernet_loss(preds, targets)
        total_loss += losses["total"].item()

        n += 1

    return {"val_loss": total_loss / n}


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = FloorPlanDataset(
        args.data_root, split="train",
        image_size=args.image_size,
    )
    val_ds = FloorPlanDataset(
        args.data_root, split="test",
        image_size=args.image_size,
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
        depth_per_class=args.depth_per_class,
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
    best_val_loss = float('inf')
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
            f"LR: {lr_now:.2e}"
        )

        # Save best checkpoint
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            ckpt_path = ckpt_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "args": vars(args),
            }, ckpt_path)
            print(f"  => Saved best checkpoint (val_loss={best_val_loss:.4f}) → {ckpt_path}")

        # Save latest checkpoint every 5 epochs
        if epoch % 5 == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": val_metrics["val_loss"],
            }, ckpt_dir / f"epoch_{epoch:03d}.pt")

    print(f"\nTraining done! Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FloorPlanCAD Detector Training")
    parser.add_argument("--data_root",    default="./data/FloorPlanCAD_dataset")
    parser.add_argument("--ckpt_dir",     default="./checkpoints")
    parser.add_argument("--image_size",   type=int,   default=512)
    parser.add_argument("--batch_size",   type=int,   default=4)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--model_dim",    type=int,   default=512)
    parser.add_argument("--depth_per_class", type=int, default=2)
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--log_interval", type=int,   default=20)
    args = parser.parse_args()
    main(args)
