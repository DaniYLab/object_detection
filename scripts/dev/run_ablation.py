"""Sequential train + val-AP evaluation runner for the core ablation presets.

Runs each preset to completion, then evaluates the val-AP-selected checkpoint on
the val split. Writes per-preset evaluation reports under outputs/. This is a
convenience wrapper around train.py and evaluate.py for unattended runs; it does
not touch the held-out test split.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = "./data/FloorPlanCAD_original"
MANIFEST = "./data/FloorPlanCAD_original/splits.json"


def _run(cmd: list[str]) -> None:
    print("\n>>> " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def train_and_eval(
    preset: str,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    val_ap_interval: int,
    limit_val_ap_images: int,
    neg_queries_per_pos: int,
    precision: str,
) -> None:
    ckpt_dir = f"./checkpoints/{preset}_seed{seed}"
    train_cmd = [
        sys.executable, "-u", "train.py",
        "--data-root", DATA_ROOT,
        "--manifest", MANIFEST,
        "--preset", preset,
        "--seed", str(seed),
        "--ckpt-dir", ckpt_dir,
        "--batch-size", str(batch_size),
        "--num-workers", str(num_workers),
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--warmup-steps", "500",
        "--sampler", "balanced",
        "--balance-power", "0.5",
        "--neg-queries-per-pos", str(neg_queries_per_pos),
        "--precision", precision,
        "--val-ap-interval", str(val_ap_interval),
        "--limit-val-ap-images", str(limit_val_ap_images),
        "--log-interval", "50",
    ]
    _run(train_cmd)

    # Prefer the val-AP-selected checkpoint; fall back to best (val loss).
    ap_ckpt = Path(ROOT) / ckpt_dir[2:] / "best_val_ap.pt"
    checkpoint = ap_ckpt if ap_ckpt.is_file() else Path(ROOT) / ckpt_dir[2:] / "best.pt"
    report = f"./outputs/evaluation_val_{preset}_seed{seed}.json"
    eval_cmd = [
        sys.executable, "-u", "evaluate.py",
        "--data-root", DATA_ROOT,
        "--manifest", MANIFEST,
        "--split", "val",
        "--checkpoint", str(checkpoint),
        "--report", report,
        "--image-size", "512",
        "--batch-size", "1",
        "--class-chunk-size", "8",
    ]
    _run(eval_cmd)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--presets",
        nargs="+",
        default=["centernet_baseline", "shared_no_condition", "floorplan_base"],
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--val-ap-interval", type=int, default=5)
    parser.add_argument("--limit-val-ap-images", type=int, default=256)
    parser.add_argument("--neg-queries-per-pos", type=int, default=1)
    parser.add_argument(
        "--precision",
        choices=["fp32", "bf16", "fp16", "auto"],
        default="bf16",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    for preset in args.presets:
        train_and_eval(
            preset,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_workers=args.num_workers,
            val_ap_interval=args.val_ap_interval,
            limit_val_ap_images=args.limit_val_ap_images,
            neg_queries_per_pos=args.neg_queries_per_pos,
            precision=args.precision,
        )
    print("\nAll runs complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
