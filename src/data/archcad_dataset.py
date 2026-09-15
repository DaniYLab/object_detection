"""PyTorch Dataset implementations for ArchCAD-400K.

Provides:
- ``ArchCADImageDataset``: Yields each full floor plan slice once (image, all-class boxes, strokes).
- ``ArchCADQueryDataset``: Query-conditioned dataset yielding (image, queried_class, targets, strokes)
  matching the exact training contract of ``FloorPlanDetector``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.archcad_constants import (
    ARCHCAD_CLASS_TO_IDX,
    ARCHCAD_IDX_TO_CLASS,
    ARCHCAD_IMAGE_SIZE,
    NUM_ARCHCAD_CLASSES,
)
from src.data.targets import generate_centernet_targets
from src.data.strokes import transform_normalized_strokes
from src.data.transforms import PairedTrainTransform, ResizeNormalize, apply_transform


class ArchCADImageDataset(Dataset):
    """Full-image dataset yielding one floor plan slice per index."""

    def __init__(
        self,
        data_root: str | Path = "data/ArchCAD",
        split: str = "train",
        manifest_path: str | Path | None = None,
        image_size: int = 512,
        vector_mode: bool = True,
        n_max_strokes: int = 1024,
        transform: Callable | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.vector_mode = vector_mode
        self.n_max_strokes = n_max_strokes

        manifest_file = (
            Path(manifest_path) if manifest_path else self.root / "splits.json"
        )
        if not manifest_file.exists():
            raise FileNotFoundError(f"Split manifest not found: {manifest_file}")

        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        if split not in manifest["splits"]:
            raise ValueError(
                f"Split {split!r} not in manifest. Choices: {list(manifest['splits'].keys())}"
            )

        self.image_ids = manifest["splits"][split]["images"]
        self.png_dir = self.root / "png"
        self.meta_dir = self.root / "metadata"

        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = PairedTrainTransform(image_size=image_size)
        else:
            self.transform = ResizeNormalize(image_size=image_size)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image_id = self.image_ids[idx]
        png_path = self.png_dir / f"{image_id}.png"
        meta_path = self.meta_dir / f"{image_id}_meta.json"

        with Image.open(png_path) as img:
            image = img.convert("RGB")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        orig_w, orig_h = meta.get("image_size", ARCHCAD_IMAGE_SIZE)

        boxes = []
        labels = []
        for inst in meta.get("instances", []):
            cat_name = inst.get("category_name")
            bbox = inst.get("bbox")
            if cat_name in ARCHCAD_CLASS_TO_IDX and bbox and len(bbox) == 4:
                boxes.append([float(v) for v in bbox])
                labels.append(ARCHCAD_CLASS_TO_IDX[cat_name])

        boxes_t = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_t = torch.tensor(labels, dtype=torch.long)

        # Apply spatial transform
        image_t, boxes_t = apply_transform(self.transform, image, boxes_t)

        # Handle vector strokes
        strokes_t, stroke_mask = self._prepare_strokes(meta.get("strokes", []))
        coordinate_matrix = getattr(self.transform, "last_coordinate_matrix", None)
        transform_input_size = getattr(self.transform, "last_input_size", None)
        transform_output_size = getattr(self.transform, "last_output_size", None)
        if (
            self.vector_mode
            and isinstance(coordinate_matrix, torch.Tensor)
            and transform_input_size
            and transform_output_size
        ):
            valid_strokes = transform_normalized_strokes(
                strokes_t[stroke_mask],
                coordinate_matrix,
                transform_input_size,
                transform_output_size,
            )
            strokes_t = strokes_t.clone()
            strokes_t[stroke_mask] = valid_strokes

        return {
            "image_id": image_id,
            "image": image_t,
            "boxes": boxes_t,
            "labels": labels_t,
            "strokes": strokes_t,
            "stroke_mask": stroke_mask,
        }

    def _prepare_strokes(
        self, raw_strokes: list[list[float]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.vector_mode or not raw_strokes:
            return (
                torch.zeros((self.n_max_strokes, 12), dtype=torch.float32),
                torch.zeros(self.n_max_strokes, dtype=torch.bool),
            )

        strokes_tensor = torch.tensor(raw_strokes, dtype=torch.float32)
        n = len(strokes_tensor)

        if n > self.n_max_strokes:
            if self.split == "train":
                indices = torch.randperm(n)[: self.n_max_strokes]
            else:
                indices = torch.linspace(0, n - 1, self.n_max_strokes).long()
            strokes_tensor = strokes_tensor[indices]
            n = self.n_max_strokes

        padded = torch.zeros((self.n_max_strokes, 12), dtype=torch.float32)
        padded[:n] = strokes_tensor
        mask = torch.zeros(self.n_max_strokes, dtype=torch.bool)
        mask[:n] = True

        return padded, mask


class ArchCADQueryDataset(Dataset):
    """Query-conditioned dataset for ArchCAD-400K training and evaluation.

    Expands each image into (image, query_class) pairs so the model learns
    conditioned detection against positive and absent negative queries.
    """

    def __init__(
        self,
        data_root: str | Path = "data/ArchCAD",
        split: str = "train",
        manifest_path: str | Path | None = None,
        image_size: int = 512,
        output_stride: int = 8,
        head_stride: int | None = None,
        vector_mode: bool = True,
        n_max_strokes: int = 1024,
        negatives_per_positive: int = 1,
        seed: int = 1337,
    ) -> None:
        super().__init__()
        self.image_dataset = ArchCADImageDataset(
            data_root=data_root,
            split=split,
            manifest_path=manifest_path,
            image_size=image_size,
            vector_mode=vector_mode,
            n_max_strokes=n_max_strokes,
        )
        self.image_size = image_size
        self.output_stride = output_stride
        self.head_stride = head_stride if head_stride is not None else output_stride
        self.split = split
        self.negatives_per_positive = negatives_per_positive
        self.num_classes = NUM_ARCHCAD_CLASSES

        # Build query index
        self.queries: list[tuple[int, int]] = []  # (image_dataset_idx, class_idx)
        self._build_query_index(seed)

    def _build_query_index(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        for img_idx in range(len(self.image_dataset)):
            image_id = self.image_dataset.image_ids[img_idx]
            meta_path = self.image_dataset.meta_dir / f"{image_id}_meta.json"
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            present_classes = set()
            for inst in meta.get("instances", []):
                cat = inst.get("category_name")
                if cat in ARCHCAD_CLASS_TO_IDX:
                    present_classes.add(ARCHCAD_CLASS_TO_IDX[cat])

            # In train: query all present classes + sampled absent negatives
            if self.split == "train":
                for c in present_classes:
                    self.queries.append((img_idx, c))

                absent = [
                    c for c in range(self.num_classes) if c not in present_classes
                ]
                if absent and self.negatives_per_positive > 0:
                    n_neg = min(
                        len(absent),
                        max(1, len(present_classes) * self.negatives_per_positive),
                    )
                    sampled_neg = rng.choice(absent, size=n_neg, replace=False)
                    for c in sampled_neg:
                        self.queries.append((img_idx, int(c)))
            else:
                # In eval: query all present classes (or all classes if evaluated per class)
                for c in present_classes:
                    self.queries.append((img_idx, c))

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_idx, class_idx = self.queries[idx]
        sample = self.image_dataset[img_idx]

        boxes = sample["boxes"]
        labels = sample["labels"]

        # Filter boxes for queried class
        mask = labels == class_idx
        class_boxes = boxes[mask]

        targets, _ = generate_centernet_targets(
            boxes=class_boxes,
            image_size=(self.image_size, self.image_size),
            output_stride=self.head_stride,
            collision_policy="largest",
        )

        return {
            "image_id": sample["image_id"],
            "image": sample["image"],
            "class_id": torch.tensor(class_idx, dtype=torch.long),
            "class_name": ARCHCAD_IDX_TO_CLASS[class_idx],
            "targets": targets,
            "strokes": sample["strokes"],
            "stroke_mask": sample["stroke_mask"],
        }
