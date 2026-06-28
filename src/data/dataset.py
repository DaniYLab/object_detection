"""
FloorPlanCAD Dataset — PyTorch Dataset class.

Loads:
  - original.png  → image tensor
  - {class}_{n}.png → crop tensors grouped by class
  - Text prompt: "Find {class_name} in this floor plan drawing"
"""

from __future__ import annotations

import json
import random
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


def _crop_transform(crop_size: int = 128) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


class FloorPlanDataset(Dataset):
    """
    FloorPlanCAD dataset.

    Each sample returns:
      image       : Tensor [3, H, W]  — original floor plan
      heatmap     : Tensor [C, H, W]  — binary mask per class (from crop bboxes)
      class_crops : dict[str, Tensor] — one representative crop per class present
      texts       : list[str]         — text prompts for each class present
      class_ids   : list[int]         — class indices present in this sample
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int = 512,
        crop_size: int = 128,
        max_crops_per_class: int = 4,
        transform: Optional[transforms.Compose] = None,
        crop_transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.root = Path(root) / split
        self.image_size = image_size
        self.crop_size = crop_size
        self.max_crops = max_crops_per_class
        self.transform = transform or _default_transform(image_size)
        self.crop_transform = crop_transform or _crop_transform(crop_size)

        # Collect all sample dirs
        self.samples = sorted([
            d for d in self.root.iterdir()
            if d.is_dir() and (d / "original.png").exists()
        ])

        if not self.samples:
            raise RuntimeError(f"No samples found in {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample_dir = self.samples[idx]

        # ── Load original image ────────────────────────────────────────────────
        img = Image.open(sample_dir / "original.png").convert("RGB")
        img_w, img_h = img.size
        image_tensor = self.transform(img)

        # ── Collect crops grouped by class ────────────────────────────────────
        crop_files: dict[str, list[Path]] = {}
        for f in sample_dir.glob("*.png"):
            if f.stem == "original":
                continue
            # Extract class name: e.g. "door_double_003" → find class
            cls = self._parse_class(f.stem)
            if cls is not None:
                crop_files.setdefault(cls, []).append(f)

        present_classes = sorted(crop_files.keys())
        class_ids = [CLASS_TO_IDX[c] for c in present_classes if c in CLASS_TO_IDX]

        # ── Build CenterNet Targets ───────────────────────────────────────────
        # center_heatmap [1, H, W] : Gaussian peaks at object centers
        # size_map [2, H, W]       : (w, h) of the object at the center location
        # mask_map [1, H, W]       : 1 at object centers, 0 elsewhere (for L1 loss)
        center_heatmap = torch.zeros(1, self.image_size, self.image_size)
        size_map = torch.zeros(2, self.image_size, self.image_size)
        mask_map = torch.zeros(1, self.image_size, self.image_size)

        meta_path = sample_dir / "metadata.json"

        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            orig_w, orig_h = meta.get("image_size", [img_w, img_h])
            sx = self.image_size / orig_w
            sy = self.image_size / orig_h

            for inst in meta.get("instances", []):
                x0, y0, x1, y1 = inst["bbox_px"]
                
                # Scale to output resolution
                hx0 = max(0, min(self.image_size - 1, x0 * sx))
                hy0 = max(0, min(self.image_size - 1, y0 * sy))
                hx1 = max(0, min(self.image_size - 1, x1 * sx))
                hy1 = max(0, min(self.image_size - 1, y1 * sy))
                
                h, w = hy1 - hy0, hx1 - hx0
                if h > 0 and w > 0:
                    cx = int(hx0 + w / 2)
                    cy = int(hy0 + h / 2)
                    
                    # Ensure center is within bounds
                    if 0 <= cx < self.image_size and 0 <= cy < self.image_size:
                        # Draw Gaussian bump (radius proportional to size)
                        radius = max(1, int(min(h, w) / 6))
                        vec = torch.arange(-radius, radius + 1, dtype=torch.float32)
                        y, x = torch.meshgrid(vec, vec, indexing='ij')
                        sigma = radius / 3.0
                        gaussian = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
                        
                        # Bounding box for the gaussian on the heatmap
                        left, right = min(cx, radius), min(self.image_size - cx, radius + 1)
                        top, bottom = min(cy, radius), min(self.image_size - cy, radius + 1)
                        
                        # Assign maximum value (in case of overlapping bumps)
                        center_heatmap[0, cy - top : cy + bottom, cx - left : cx + right] = torch.maximum(
                            center_heatmap[0, cy - top : cy + bottom, cx - left : cx + right],
                            gaussian[radius - top : radius + bottom, radius - left : radius + right]
                        )
                        
                        # Set size and mask at the exact center point
                        size_map[0, cy, cx] = w
                        size_map[1, cy, cx] = h
                        mask_map[0, cy, cx] = 1.0

        # We skip crops for class-agnostic phase
        class_crops = {}

        # ── Text prompts ───────────────────────────────────────────────────────
        # A single dummy prompt for class-agnostic object detection
        texts = ["Find object in this floor plan drawing"]
        class_ids = [0]

        return {
            "image": image_tensor,          # [3, H, W]
            "center_heatmap": center_heatmap, # [1, H, W]
            "size_map": size_map,           # [2, H, W]
            "mask_map": mask_map,           # [1, H, W]
            "class_crops": class_crops,     # empty dict
            "texts": texts,                 # list[str]
            "class_ids": class_ids,         # list[int]
            "sample_id": sample_dir.name,
        }

    def _parse_class(self, stem: str) -> str | None:
        """Extract class name from filename stem like 'door_double_003'."""
        # Try longest matching class name first
        for cls in sorted(CLASS_NAMES, key=len, reverse=True):
            if stem.startswith(cls + "_") or stem == cls:
                return cls
        return None


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate: handle variable-length class_ids and texts."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "center_heatmap": torch.stack([b["center_heatmap"] for b in batch]),
        "size_map": torch.stack([b["size_map"] for b in batch]),
        "mask_map": torch.stack([b["mask_map"] for b in batch]),
        "texts": [b["texts"] for b in batch],
        "class_ids": [b["class_ids"] for b in batch],
        "sample_ids": [b["sample_id"] for b in batch],
        # class_crops: list of dicts (variable per sample, don't stack)
        "class_crops": [b["class_crops"] for b in batch],
    }


if __name__ == "__main__":
    # Quick sanity check
    ds = FloorPlanDataset(
        root="./data/FloorPlanCAD_dataset",
        split="train",
        image_size=512,
        crop_size=128,
    )
    print(f"Dataset size: {len(ds)}")
    sample = ds[0]
    print(f"  image shape  : {sample['image'].shape}")
    print(f"  center_heatmap shape: {sample['center_heatmap'].shape}")
    print(f"  size_map shape : {sample['size_map'].shape}")
    print(f"  mask_map shape : {sample['mask_map'].shape}")
    print(f"  classes      : {sample['class_ids']}")
    print(f"  texts        : {sample['texts'][:3]}")

