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

        # ── Build heatmap [NUM_CLASSES, H, W] from metadata.json ─────────────
        heatmap = torch.zeros(NUM_CLASSES, self.image_size, self.image_size)
        meta_path = sample_dir / "metadata.json"

        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            orig_w, orig_h = meta.get("image_size", [img_w, img_h])
            sx = self.image_size / orig_w
            sy = self.image_size / orig_h

            for inst in meta.get("instances", []):
                cid = inst.get("class_id", -1)
                if cid < 0 or cid >= NUM_CLASSES:
                    continue
                x0, y0, x1, y1 = inst["bbox_px"]
                # Scale to heatmap resolution
                hx0 = max(0, int(x0 * sx))
                hy0 = max(0, int(y0 * sy))
                hx1 = min(self.image_size, int(x1 * sx))
                hy1 = min(self.image_size, int(y1 * sy))
                if hx1 > hx0 and hy1 > hy0:
                    heatmap[cid, hy0:hy1, hx0:hx1] = 1.0
        else:
            # Fallback: binary presence (no spatial info) until metadata is ready
            for cls in present_classes:
                if cls in CLASS_TO_IDX:
                    heatmap[CLASS_TO_IDX[cls]] = 0.1   # low signal, not misleading



        # ── Sample one representative crop per class ───────────────────────────
        class_crops: dict[str, torch.Tensor] = {}
        for cls, files in crop_files.items():
            if cls not in CLASS_TO_IDX:
                continue
            chosen = random.sample(files, min(self.max_crops, len(files)))
            crops = []
            for f in chosen:
                try:
                    c = Image.open(f).convert("RGB")
                    crops.append(self.crop_transform(c))
                except Exception:
                    continue
            if crops:
                # Stack and mean-pool → single representative tensor [3, crop_H, crop_W]
                class_crops[cls] = torch.stack(crops).mean(0)

        # ── Text prompts ───────────────────────────────────────────────────────
        texts = [TEXT_TEMPLATE.format(cls=cls) for cls in present_classes
                 if cls in CLASS_TO_IDX]

        return {
            "image": image_tensor,          # [3, H, W]
            "heatmap": heatmap,             # [35, H, W]
            "class_crops": class_crops,     # dict[str, Tensor[3, cH, cW]]
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
        "heatmap": torch.stack([b["heatmap"] for b in batch]),
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
    print(f"  heatmap shape: {sample['heatmap'].shape}")
    print(f"  classes      : {[CLASS_NAMES[i] for i in sample['class_ids']]}")
    print(f"  texts        : {sample['texts'][:3]}")
    print(f"  crops        : {list(sample['class_crops'].keys())}")
