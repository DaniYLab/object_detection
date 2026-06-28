"""
EDA — FloorPlanCAD Dataset Explorer
Chạy sau khi download_dataset.py hoàn thành.
"""

import json
import os
import random
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


DATA_DIR = Path("./data/FloorPlanCAD")
IMAGES_DIR = DATA_DIR / "images"
META_FILE = DATA_DIR / "metadata.json"


# ─── 1. Load metadata ─────────────────────────────────────────────────────────

def load_metadata() -> list:
    with open(META_FILE, encoding="utf-8") as f:
        meta = json.load(f)
    samples = meta["samples"]
    print(f"[INFO] Loaded {len(samples)} samples")
    print(f"[INFO] Keys: {list(samples[0].keys()) if samples else 'N/A'}")
    return samples


# ─── 2. Dataset statistics ────────────────────────────────────────────────────

def print_stats(samples: list) -> None:
    print("\n" + "=" * 50)
    print("  DATASET STATISTICS")
    print("=" * 50)
    print(f"  Total samples : {len(samples)}")

    # Count label types if present
    label_keys = [k for k in samples[0].keys()
                  if k not in ("idx", "image_path")]
    print(f"  Metadata keys : {label_keys}")

    for key in label_keys:
        vals = [s[key] for s in samples if s.get(key) is not None]
        if vals and isinstance(vals[0], str):
            counts = Counter(vals)
            print(f"\n  [{key}] — {len(counts)} unique values:")
            for val, cnt in counts.most_common(10):
                bar = "█" * int(cnt / len(samples) * 30)
                print(f"    {val:30s} {bar} ({cnt})")


# ─── 3. Image size distribution ───────────────────────────────────────────────

def plot_image_sizes(samples: list, n_samples: int = 200) -> None:
    widths, heights = [], []
    subset = random.sample(samples, min(n_samples, len(samples)))

    for s in subset:
        try:
            img = Image.open(s["image_path"])
            widths.append(img.width)
            heights.append(img.height)
        except Exception:
            continue

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Image Size Distribution (sample)", fontsize=14, fontweight="bold")

    axes[0].hist(widths, bins=30, color="#4F8EF7", edgecolor="white")
    axes[0].set_xlabel("Width (px)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Width — avg: {sum(widths)//len(widths)}px")

    axes[1].hist(heights, bins=30, color="#F7774F", edgecolor="white")
    axes[1].set_xlabel("Height (px)")
    axes[1].set_title(f"Height — avg: {sum(heights)//len(heights)}px")

    plt.tight_layout()
    plt.savefig(DATA_DIR / "eda_image_sizes.png", dpi=120)
    plt.show()
    print(f"[Saved] {DATA_DIR / 'eda_image_sizes.png'}")


# ─── 4. Sample grid visualization ─────────────────────────────────────────────

def plot_sample_grid(samples: list, n: int = 16) -> None:
    cols = 4
    rows = n // cols
    subset = random.sample(samples, min(n, len(samples)))

    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
    fig.suptitle("FloorPlanCAD — Random Samples", fontsize=16, fontweight="bold")

    for ax, s in zip(axes.flatten(), subset):
        try:
            img = Image.open(s["image_path"])
            ax.imshow(img, cmap="gray" if img.mode == "L" else None)
            ax.set_title(f"idx={s['idx']}", fontsize=8)
        except Exception as e:
            ax.set_title(f"Error: {e}", fontsize=7)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(DATA_DIR / "eda_sample_grid.png", dpi=100)
    plt.show()
    print(f"[Saved] {DATA_DIR / 'eda_sample_grid.png'}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not META_FILE.exists():
        print("[ERROR] metadata.json not found. Run download_dataset.py first.")
        raise SystemExit(1)

    samples = load_metadata()
    print_stats(samples)
    plot_image_sizes(samples)
    plot_sample_grid(samples, n=16)
