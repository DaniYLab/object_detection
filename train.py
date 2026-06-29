"""
Training script for FloorPlanCAD Detector.

Usage:
    python train.py
    python train.py --data_root ./data/FloorPlanCAD_original
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).parent))
from src.data.dataset import FloorPlanDataset, collate_fn, CLASS_NAMES, NUM_CLASSES
from src.models.detector import FloorPlanDetector


# ── Loss Functions ─────────────────────────────────────────────────────────────

def focal_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """Penalty-reduced Focal Loss for CenterNet Gaussian heatmaps.

    `pred` is expected to be a probability map in [0, 1]. Targets must be
    generated at the same output resolution so exact peak pixels remain 1.0.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Heatmap shape mismatch: pred={pred.shape}, target={target.shape}")

    pred = torch.clamp(pred, 1e-4, 1 - 1e-4)
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()

    neg_weights = torch.pow(1 - target, beta)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds

    num_pos = pos_inds.sum()
    if num_pos == 0:
        # Should be rare because dataset indexes only classes present in an image.
        return -neg_loss.mean()
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos


def l1_loss_masked(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Smooth L1 loss only at object centers."""
    if pred.shape != target.shape:
        raise ValueError(f"Regression shape mismatch: pred={pred.shape}, target={target.shape}")
    if mask.shape[-2:] != pred.shape[-2:]:
        raise ValueError(f"Mask spatial mismatch: mask={mask.shape}, pred={pred.shape}")

    mask = mask.expand_as(pred)
    l1 = F.smooth_l1_loss(pred, target, reduction="none")
    return (l1 * mask).sum() / (mask.sum() + 1e-4)


def centernet_loss(
    preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    focal_w: float = 10.0,
    size_w: float = 1.0,
    offset_w: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Combined CenterNet loss: focal center + masked size + masked offset."""
    pred_hm = preds["center_heatmap"]
    pred_sz = preds["size_map"]
    pred_off = preds["offset_map"]

    tgt_hm = targets["center_heatmap"]
    tgt_sz = targets["size_map"]
    tgt_off = targets["offset_map"]
    tgt_mask = targets["mask_map"]

    # Targets must already be generated at output resolution.
    if pred_hm.shape[-2:] != tgt_hm.shape[-2:]:
        raise ValueError(
            "CenterNet targets must match prediction resolution. "
            f"pred={pred_hm.shape[-2:]}, target={tgt_hm.shape[-2:]}"
        )

    fl = focal_loss(pred_hm, tgt_hm)
    size_l1 = l1_loss_masked(pred_sz, tgt_sz, tgt_mask)
    offset_l1 = l1_loss_masked(pred_off, tgt_off, tgt_mask)
    total = focal_w * fl + size_w * size_l1 + offset_w * offset_l1
    num_pos = tgt_mask.sum()
    return {
        "total": total,
        "focal": fl,
        "size_l1": size_l1,
        "offset_l1": offset_l1,
        "num_pos": num_pos,
    }


# ── DataLoader helpers ─────────────────────────────────────────────────────────

def _maybe_subset(dataset: FloorPlanDataset, limit: int) -> FloorPlanDataset | Subset:
    if limit <= 0 or limit >= len(dataset):
        return dataset
    return Subset(dataset, list(range(limit)))


def _make_train_loader(
    dataset: FloorPlanDataset | Subset,
    base_dataset: FloorPlanDataset,
    args: argparse.Namespace,
    pin_memory: bool,
) -> DataLoader:
    sampler = None
    shuffle = True

    if args.sampler == "balanced":
        weights = base_dataset.get_sample_weights(balance_power=args.balance_power)
        if isinstance(dataset, Subset):
            weights = weights[dataset.indices]
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(dataset),
            replacement=True,
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )


def _print_class_stats(dataset: FloorPlanDataset) -> None:
    counts = dataset.class_counts
    if not counts:
        return
    sorted_counts = sorted(counts.items(), key=lambda kv: kv[1])
    low = ", ".join(f"{CLASS_NAMES[c]}={n}" for c, n in sorted_counts[:5])
    high = ", ".join(f"{CLASS_NAMES[c]}={n}" for c, n in sorted_counts[-5:])
    print(f"Class counts: min={sorted_counts[0][1]} | max={sorted_counts[-1][1]}")
    print(f"  rare : {low}")
    print(f"  common: {high}")


# ── Training Loop ──────────────────────────────────────────────────────────────

def train_one_epoch(
    model: FloorPlanDetector,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    focal_w: float = 10.0,
    size_w: float = 1.0,
    offset_w: float = 1.0,
    grad_clip: float = 1.0,
    log_interval: int = 20,
) -> dict[str, float]:
    model.train()
    total_loss = focal_sum = size_sum = offset_sum = pos_sum = 0.0
    t0 = time.time()

    for step, batch in enumerate(loader):
        image = batch["image"].to(device)
        targets = {
            "center_heatmap": batch["center_heatmap"].to(device),
            "size_map": batch["size_map"].to(device),
            "offset_map": batch["offset_map"].to(device),
            "mask_map": batch["mask_map"].to(device),
        }
        class_ids = torch.tensor(batch["class_ids"], dtype=torch.long, device=device)

        optimizer.zero_grad()
        preds = model(image, class_ids=class_ids)

        losses = centernet_loss(
            preds,
            targets,
            focal_w=focal_w,
            size_w=size_w,
            offset_w=offset_w,
        )
        losses["total"].backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += losses["total"].item()
        focal_sum += losses["focal"].item()
        size_sum += losses["size_l1"].item()
        offset_sum += losses["offset_l1"].item()
        pos_sum += losses["num_pos"].item()

        if (step + 1) % log_interval == 0:
            elapsed = time.time() - t0
            denom = step + 1
            print(
                f"  Epoch {epoch} | Step {step+1:4d}/{len(loader)} | "
                f"Loss {total_loss/denom:.4f} "
                f"(focal={focal_sum/denom:.4f}×{focal_w:g}, "
                f"size={size_sum/denom:.4f}×{size_w:g}, "
                f"offset={offset_sum/denom:.4f}×{offset_w:g}, "
                f"pos={pos_sum/denom:.1f}) | "
                f"LR {scheduler.get_last_lr()[0]:.2e} | "
                f"{elapsed:.1f}s"
            )

    n = len(loader)
    return {
        "loss": total_loss / n,
        "focal": focal_sum / n,
        "size_l1": size_sum / n,
        "offset_l1": offset_sum / n,
        "num_pos": pos_sum / n,
    }


@torch.no_grad()
def validate(
    model: FloorPlanDetector,
    loader: DataLoader,
    device: torch.device,
    focal_w: float = 10.0,
    size_w: float = 1.0,
    offset_w: float = 1.0,
) -> dict[str, float]:
    model.eval()
    total_loss = focal_sum = size_sum = offset_sum = pos_sum = 0.0
    n = 0

    for batch in loader:
        image = batch["image"].to(device)
        targets = {
            "center_heatmap": batch["center_heatmap"].to(device),
            "size_map": batch["size_map"].to(device),
            "offset_map": batch["offset_map"].to(device),
            "mask_map": batch["mask_map"].to(device),
        }
        class_ids = torch.tensor(batch["class_ids"], dtype=torch.long, device=device)

        preds = model(image, class_ids=class_ids)
        losses = centernet_loss(
            preds,
            targets,
            focal_w=focal_w,
            size_w=size_w,
            offset_w=offset_w,
        )
        total_loss += losses["total"].item()
        focal_sum += losses["focal"].item()
        size_sum += losses["size_l1"].item()
        offset_sum += losses["offset_l1"].item()
        pos_sum += losses["num_pos"].item()
        n += 1

    return {
        "val_loss": total_loss / n,
        "val_focal": focal_sum / n,
        "val_size_l1": size_sum / n,
        "val_offset_l1": offset_sum / n,
        "val_num_pos": pos_sum / n,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_base = FloorPlanDataset(
        args.data_root,
        split="train",
        image_size=args.image_size,
        output_stride=args.output_stride,
    )
    val_base = FloorPlanDataset(
        args.data_root,
        split="test",
        image_size=args.image_size,
        output_stride=args.output_stride,
    )
    _print_class_stats(train_base)

    train_ds = _maybe_subset(train_base, args.limit_train_samples)
    val_ds = _maybe_subset(val_base, args.limit_val_samples)

    pin_memory = device.type == "cuda"
    train_dl = _make_train_loader(train_ds, train_base, args, pin_memory=pin_memory)
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    print(
        f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | "
        f"Steps/epoch: {len(train_dl)} | Target: {train_base.output_size}×{train_base.output_size} | "
        f"Sampler: {args.sampler}"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = FloorPlanDetector(
        image_size=args.image_size,
        model_dim=args.model_dim,
        num_classes=NUM_CLASSES,
        depth_per_class=args.depth_per_class,
        fusion_mode=args.fusion_mode,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model params: {n_params:.1f}M")

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    total_steps = max(1, args.epochs * len(train_dl))
    warmup_steps = min(args.warmup_steps, max(1, total_steps - 1))
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=args.warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_steps),
        eta_min=args.lr * args.min_lr_ratio,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )
    print(
        f"Loss weights: focal={args.focal_weight:g} | size={args.size_weight:g} | "
        f"offset={args.offset_weight:g} | LR={args.lr:.2e} | warmup_steps={warmup_steps}"
    )

    # ── Training ──────────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}/{args.epochs}]")
        train_metrics = train_one_epoch(
            model,
            train_dl,
            optimizer,
            scheduler,
            device,
            epoch,
            focal_w=args.focal_weight,
            size_w=args.size_weight,
            offset_w=args.offset_weight,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
        )
        val_metrics = validate(
            model,
            val_dl,
            device,
            focal_w=args.focal_weight,
            size_w=args.size_weight,
            offset_w=args.offset_weight,
        )

        lr_now = scheduler.get_last_lr()[0]
        print(
            f"  => Train loss: {train_metrics['loss']:.4f} "
            f"(focal={train_metrics['focal']:.4f}, size={train_metrics['size_l1']:.4f}, "
            f"offset={train_metrics['offset_l1']:.4f}) | "
            f"Val loss: {val_metrics['val_loss']:.4f} "
            f"(focal={val_metrics['val_focal']:.4f}, size={val_metrics['val_size_l1']:.4f}, "
            f"offset={val_metrics['val_offset_l1']:.4f}) | "
            f"LR: {lr_now:.2e}"
        )

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            ckpt_path = ckpt_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_metrics": val_metrics,
                "args": vars(args),
            }, ckpt_path)
            print(f"  => Saved best checkpoint (val_loss={best_val_loss:.4f}) → {ckpt_path}")

        if epoch % 5 == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": val_metrics["val_loss"],
                "val_metrics": val_metrics,
                "args": vars(args),
            }, ckpt_dir / f"epoch_{epoch:03d}.pt")

    print(f"\nTraining done! Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FloorPlanCAD Detector Training")
    parser.add_argument("--data_root",    default="./data/FloorPlanCAD_original")
    parser.add_argument("--ckpt_dir",     default="./checkpoints")
    parser.add_argument("--image_size",   type=int,   default=512)
    parser.add_argument("--output_stride", type=int,  default=8)
    parser.add_argument("--batch_size",   type=int,   default=4)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--model_dim",    type=int,   default=512)
    parser.add_argument("--depth_per_class", type=int, default=2)
    parser.add_argument("--fusion_mode", choices=["current", "film", "film_cross_attn"], default="film")
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--lr",           type=float, default=1e-5)
    parser.add_argument("--focal_weight", type=float, default=10.0)
    parser.add_argument("--size_weight",  type=float, default=1.0)
    parser.add_argument("--offset_weight", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int,   default=500)
    parser.add_argument("--warmup_start_factor", type=float, default=0.1)
    parser.add_argument("--min_lr_ratio", type=float, default=0.01)
    parser.add_argument("--grad_clip",    type=float, default=1.0)
    parser.add_argument("--sampler", choices=["shuffle", "balanced"], default="balanced")
    parser.add_argument("--balance_power", type=float, default=0.5)
    parser.add_argument("--limit_train_samples", type=int, default=0)
    parser.add_argument("--limit_val_samples", type=int, default=0)
    parser.add_argument("--log_interval", type=int,   default=20)
    args = parser.parse_args()
    main(args)
