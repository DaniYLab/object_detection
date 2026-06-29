"""Quick sanity test for FloorPlanDataset (per-class expanded)."""
from __future__ import annotations

import argparse
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from src.data.dataset import FloorPlanDataset, collate_fn, CLASS_NAMES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./data/FloorPlanCAD_original")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--output_stride", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    ds = FloorPlanDataset(
        args.data_root,
        split=args.split,
        image_size=args.image_size,
        output_stride=args.output_stride,
    )
    print(f"Dataset size : {len(ds):,}  (expanded: image × class)")
    print(f"Target size  : {ds.output_size}×{ds.output_size}")
    print(f"Class counts : min={min(ds.class_counts.values())}, max={max(ds.class_counts.values())}")

    sample = ds[0]
    print(f"image shape       : {sample['image'].shape}")
    print(f"center_heatmap    : {sample['center_heatmap'].shape}")
    print(f"size_map          : {sample['size_map'].shape}")
    print(f"offset_map        : {sample['offset_map'].shape}")
    print(f"mask_map          : {sample['mask_map'].shape}")
    print(f"text              : {sample['text']}")
    print(f"class_id          : {sample['class_id']} ({CLASS_NAMES[sample['class_id']]})")
    print(f"boxes             : {sample['boxes'].shape}")

    n_centers = int(sample["mask_map"].sum().item())
    heatmap_max = float(sample["center_heatmap"].max().item())
    print(f"num object centers: {n_centers}")
    print(f"heatmap max       : {heatmap_max:.4f}")

    assert sample["center_heatmap"].shape[-1] == args.image_size // args.output_stride
    assert sample["size_map"].shape == (2, ds.output_size, ds.output_size)
    assert sample["offset_map"].shape == (2, ds.output_size, ds.output_size)
    assert sample["mask_map"].shape == (1, ds.output_size, ds.output_size)
    assert n_centers > 0, "Expanded sample should have at least one object center"
    assert heatmap_max == 1.0, "Gaussian center peak should remain exactly 1.0"

    mask = sample["mask_map"].expand_as(sample["offset_map"]) > 0
    offsets = sample["offset_map"][mask]
    assert bool(((offsets >= 0) & (offsets < 1)).all().item()), "Offsets must be in [0, 1)"

    mask_sz = sample["mask_map"].expand_as(sample["size_map"]) > 0
    sizes = sample["size_map"][mask_sz]
    assert bool((sizes > 0).all().item()), "Sizes at centers must be positive"

    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=0)
    batch = next(iter(dl))
    print(f"\nBatch image       : {batch['image'].shape}")
    print(f"Batch heatmap     : {batch['center_heatmap'].shape}")
    print(f"Batch size_map    : {batch['size_map'].shape}")
    print(f"Batch offset_map  : {batch['offset_map'].shape}")
    print(f"Batch texts       : {batch['texts'][:2]}")
    print(f"Batch class_ids   : {batch['class_ids'][:4]}")
    print("Dataset OK!")


if __name__ == "__main__":
    main()
