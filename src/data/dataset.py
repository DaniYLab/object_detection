"""
FloorPlanCAD Dataset — PyTorch Dataset class.

Text-conditioned, per-class expanded dataset for CenterNet-style detection.
Each (image, class) pair is a separate sample.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ── Class mapping (id → name, ordered for index assignment) ───────────────────
CLASS_NAMES = [
    "annotation_text", "bathtub", "bed", "cabinet", "chair",
    "column", "counter", "dimension_line", "door_double", "door_revolving",
    "door_single", "door_sliding", "elevator", "escalator", "escalator_stair",
    "floor_plan_area", "oven", "parking", "plant", "ramp",
    "refrigerator", "room_label", "shower", "sink", "sofa",
    "stair", "symbol_misc", "table", "toilet", "tv",
    "wall", "washing_machine", "window", "window_bay", "window_blind",
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)  # 35

TEXT_TEMPLATE = "Find {cls} in this floor plan drawing"


def _default_transform(image_size: int = 512) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])



class FloorPlanDataset(Dataset):
    """
    FloorPlanCAD dataset — text-conditioned, per-class expanded.

    Each (image, class) pair is a separate sample in the dataset.
    If an image has N classes, it appears N times — once per class.
    Every epoch trains ALL classes of ALL images.

    Each sample returns:
      image          : Tensor [3, H, W]
      center_heatmap : Tensor [1, H, W]  — Gaussian peaks for THIS class only
      size_map       : Tensor [2, H, W]  — (w, h) at object centers
      mask_map       : Tensor [1, H, W]  — 1 at centers, 0 elsewhere
      text           : str               — "Find {class} in this floor plan drawing"
      class_id       : int               — class index
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int = 512,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.root = Path(root) / split
        self.image_size = image_size
        self.transform = transform or _default_transform(image_size)

        # Collect all sample dirs
        sample_dirs = sorted([
            d for d in self.root.iterdir()
            if d.is_dir() and (d / "original.png").exists()
        ])

        if not sample_dirs:
            raise RuntimeError(f"No samples found in {self.root}")

        # ── Build expanded index: (sample_dir, class_name) ────────────────────
        # Each class present in an image becomes a separate dataset entry.
        self.index: list[tuple[Path, str]] = []
        for sample_dir in sample_dirs:
            meta_path = sample_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                classes_in_sample = set()
                for inst in meta.get("instances", []):
                    cls_name = inst.get("class", "")
                    if cls_name in CLASS_TO_IDX:
                        classes_in_sample.add(cls_name)
                for cls_name in sorted(classes_in_sample):
                    self.index.append((sample_dir, cls_name))

        if not self.index:
            raise RuntimeError(f"No (image, class) pairs found in {self.root}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        sample_dir, target_class = self.index[idx]

        # ── Load original image ────────────────────────────────────────────────
        img = Image.open(sample_dir / "original.png").convert("RGB")
        img_w, img_h = img.size
        image_tensor = self.transform(img)

        # ── Build CenterNet targets for target_class only ──────────────────────
        center_heatmap = torch.zeros(1, self.image_size, self.image_size)
        size_map = torch.zeros(2, self.image_size, self.image_size)
        mask_map = torch.zeros(1, self.image_size, self.image_size)

        meta_path = sample_dir / "metadata.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        orig_w, orig_h = meta.get("image_size", [img_w, img_h])
        sx = self.image_size / orig_w
        sy = self.image_size / orig_h

        for inst in meta.get("instances", []):
            if inst.get("class", "") != target_class:
                continue

            x0, y0, x1, y1 = inst["bbox_px"]

            hx0 = max(0, min(self.image_size - 1, x0 * sx))
            hy0 = max(0, min(self.image_size - 1, y0 * sy))
            hx1 = max(0, min(self.image_size - 1, x1 * sx))
            hy1 = max(0, min(self.image_size - 1, y1 * sy))

            h, w = hy1 - hy0, hx1 - hx0
            if h > 0 and w > 0:
                cx = int(hx0 + w / 2)
                cy = int(hy0 + h / 2)

                if 0 <= cx < self.image_size and 0 <= cy < self.image_size:
                    radius = max(1, int(min(h, w) / 6))
                    vec = torch.arange(-radius, radius + 1, dtype=torch.float32)
                    gy, gx = torch.meshgrid(vec, vec, indexing='ij')
                    sigma = max(radius / 3.0, 0.5)
                    gaussian = torch.exp(-(gx**2 + gy**2) / (2 * sigma**2))

                    left = min(cx, radius)
                    right = min(self.image_size - cx, radius + 1)
                    top = min(cy, radius)
                    bottom = min(self.image_size - cy, radius + 1)

                    center_heatmap[0, cy - top : cy + bottom, cx - left : cx + right] = torch.maximum(
                        center_heatmap[0, cy - top : cy + bottom, cx - left : cx + right],
                        gaussian[radius - top : radius + bottom, radius - left : radius + right]
                    )

                    size_map[0, cy, cx] = w
                    size_map[1, cy, cx] = h
                    mask_map[0, cy, cx] = 1.0

        return {
            "image": image_tensor,              # [3, H, W]
            "center_heatmap": center_heatmap,   # [1, H, W]
            "size_map": size_map,               # [2, H, W]
            "mask_map": mask_map,               # [1, H, W]
            "text": TEXT_TEMPLATE.format(cls=target_class),
            "class_id": CLASS_TO_IDX[target_class],
            "sample_id": sample_dir.name,
        }


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate for text-conditioned CenterNet."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "center_heatmap": torch.stack([b["center_heatmap"] for b in batch]),
        "size_map": torch.stack([b["size_map"] for b in batch]),
        "mask_map": torch.stack([b["mask_map"] for b in batch]),
        "texts": [b["text"] for b in batch],
        "class_ids": [b["class_id"] for b in batch],
        "sample_ids": [b["sample_id"] for b in batch],
    }


if __name__ == "__main__":
    ds = FloorPlanDataset(
        root="./data/FloorPlanCAD_dataset",
        split="train",
        image_size=512,
    )
    print(f"Dataset size: {len(ds)}  (expanded: each image×class = 1 entry)")

    for i in [0, 1, 2, len(ds) // 2]:
        sample = ds[i]
        n_centers = sample['mask_map'].sum().int().item()
        print(f"  [{i}] {sample['sample_id']} | {sample['text']} | centers={n_centers}")


