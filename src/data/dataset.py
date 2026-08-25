"""Image-level and text-query FloorPlanCAD PyTorch datasets.

``FloorPlanImageDataset`` yields each physical image once for evaluation.
``FloorPlanQueryDataset`` expands an image only after its image-level split has
been selected, yielding one sample per class present in that image. The legacy
``FloorPlanDataset`` name remains an alias of the query dataset.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.constants import CLASS_NAMES, CLASS_TO_IDX, NUM_CLASSES, TEXT_TEMPLATE
from src.data.metadata import load_metadata
from src.data.strokes import (
    STROKE_FEATURE_DIM,
    metadata_strokes,
    normalize_strokes,
    pad_stroke_batch,
    sample_strokes,
)
from src.data.splits import (
    ImageRecord,
    image_index_fingerprint,
    load_split_manifest,
    records_for_split,
    resolve_split_manifest_path,
    validate_image_records,
)
from src.data.targets import CollisionPolicy, TargetStats, generate_centernet_targets
from src.data.transforms import (
    PairedTrainTransform,
    ResizeNormalize,
    apply_transform,
)

# Kept for code that imported the old source-directory mapping directly.
SPLIT_DIRS = {
    "train": ["train_set_1", "train_set_2"],
    "test": ["test_set"],
}


def _default_transform(image_size: int = 512, split: str = "test") -> Callable[..., object]:
    """Return paired random train or deterministic evaluation preprocessing."""

    if split == "train":
        return PairedTrainTransform(image_size=image_size)
    return ResizeNormalize(image_size=image_size)


def _record_paths(root: Path, record: ImageRecord) -> tuple[Path, Path]:
    image_path = Path(record.image_path)
    metadata_path = Path(record.metadata_path)
    if not image_path.is_absolute():
        image_path = root / image_path
    if not metadata_path.is_absolute():
        metadata_path = root / metadata_path
    return image_path, metadata_path


def _dataset_fingerprints(
    root: Path,
    manifest_path: str | Path | None,
    records: list[ImageRecord],
) -> tuple[Path | None, str | None, str]:
    resolved = resolve_split_manifest_path(root, manifest_path)
    manifest_fingerprint: str | None = None
    if resolved is not None:
        manifest_fingerprint = load_split_manifest(resolved).get("fingerprint")
    return resolved, manifest_fingerprint, image_index_fingerprint(records)


def _instances_as_tensors(
    metadata: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    boxes: list[list[float]] = []
    labels: list[int] = []
    instance_ids: list[int] = []
    class_names: list[str] = []
    for instance in metadata.get("instances", []):
        class_name = instance.get("class")
        bbox = instance.get("bbox_px")
        if class_name not in CLASS_TO_IDX or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        boxes.append([float(value) for value in bbox])
        labels.append(CLASS_TO_IDX[class_name])
        instance_ids.append(int(instance.get("instance_id", -1)))
        class_names.append(class_name)
    return (
        torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(instance_ids, dtype=torch.long),
        class_names,
    )


class FloorPlanImageDataset(Dataset):
    """Image-level dataset for leakage-free validation and test evaluation."""

    def __init__(
        self,
        root: str | Path,
        split: str = "test",
        image_size: int = 512,
        transform: Optional[Callable[..., object]] = None,
        *,
        manifest_path: str | Path | None = None,
        strict_metadata: bool = False,
        records: list[ImageRecord] | None = None,
        vector_branch: bool = False,
        vector_n_max: int = 1024,
    ) -> None:
        self.root = Path(root)
        self.split = "val" if split == "validation" else split
        self.image_size = int(image_size)
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        self.transform = transform or _default_transform(self.image_size, self.split)
        self.strict_metadata = strict_metadata
        self.vector_branch = vector_branch
        self.vector_n_max = int(vector_n_max)
        if records is not None:
            self.records = validate_image_records(
                self.root,
                records,
                strict=strict_metadata,
                manifest_path=manifest_path,
            )
        else:
            self.records = records_for_split(
                self.root,
                self.split,
                manifest_path=manifest_path,
                strict=strict_metadata,
            )
        self.manifest_path, self.split_manifest_fingerprint, self.metadata_fingerprint = (
            _dataset_fingerprints(self.root, manifest_path, self.records)
        )
        self.index = self.records
        if not self.records:
            raise RuntimeError(
                f"No images with metadata found in {self.root} for split={self.split!r}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def _load_record(
        self, record: ImageRecord
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], dict[str, Any], Path]:
        image_path, metadata_path = _record_paths(self.root, record)
        metadata = load_metadata(
            metadata_path,
            allow_legacy=True,
            strict=self.strict_metadata,
        )
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            original_size = image.size
            boxes, labels, instance_ids, class_names = _instances_as_tensors(metadata)
            image_tensor, transformed_boxes = apply_transform(
                self.transform, image, boxes
            )
        return (
            image_tensor,
            transformed_boxes,
            labels,
            instance_ids,
            class_names,
            {
                "metadata": metadata,
                "metadata_path": metadata_path,
                "original_size": original_size,
            },
            image_path,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        (
            image_tensor,
            boxes,
            labels,
            instance_ids,
            class_names,
            extra,
            image_path,
        ) = self._load_record(record)
        sample = {
            "image": image_tensor,
            "boxes": boxes,
            "labels": labels,
            "class_ids": labels,
            "instance_ids": instance_ids,
            "class_names": class_names,
            "sample_id": record.sample_id,
            "image_id": record.image_id,
            "image_path": str(image_path),
            "original_size": extra["original_size"],
            "image_size": (int(image_tensor.shape[-1]), int(image_tensor.shape[-2])),
        }
        if self.vector_branch:
            # Deterministic subsample (sorted) at eval: the same drawing always
            # yields the same primitive subset.
            tokens = normalize_strokes(
                extra["metadata"].get("strokes") or [],
                extra["metadata"].get("image_size") or extra["original_size"],
            )
            if tokens.shape[0] > self.vector_n_max:
                keep = torch.linspace(
                    0, tokens.shape[0] - 1, self.vector_n_max
                ).round().long()
                tokens = tokens.index_select(0, keep)
            sample["stroke_tokens"] = tokens
        return sample


class FloorPlanQueryDataset(Dataset):
    """Text-conditioned image/class dataset with CenterNet targets.

    Each positive query ``(image, class)`` is generated for every class present
    in the image. When ``neg_queries_per_pos > 0``, additional absent-class
    queries are sampled per positive query, producing empty heatmap targets. This
    is needed so the model learns to output empty predictions for classes that are
    not present in a given image. Use ``neg_seed`` for reproducible sampling.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int = 512,
        output_stride: int = 8,
        transform: Optional[Callable[..., object]] = None,
        *,
        manifest_path: str | Path | None = None,
        strict_metadata: bool = False,
        collision_policy: CollisionPolicy = "largest",
        min_overlap: float = 0.7,
        records: list[ImageRecord] | None = None,
        neg_queries_per_pos: int = 0,
        neg_seed: int = 0,
        cache_images: bool = False,
        vector_branch: bool = False,
        vector_n_max: int = 1024,
        vector_seed: int = 0,
    ) -> None:
        self.root = Path(root)
        self.split = "val" if split == "validation" else split
        self.image_size = int(image_size)
        self.output_stride = int(output_stride)
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.output_stride <= 0 or self.image_size % self.output_stride != 0:
            raise ValueError(
                f"image_size ({self.image_size}) must be divisible by "
                f"output_stride ({self.output_stride})"
            )
        if neg_queries_per_pos < 0:
            raise ValueError("neg_queries_per_pos cannot be negative")
        self.output_size = self.image_size // self.output_stride
        self.transform = transform or _default_transform(self.image_size, self.split)
        self.strict_metadata = strict_metadata
        self.collision_policy = collision_policy
        self.min_overlap = min_overlap
        self.neg_queries_per_pos = neg_queries_per_pos
        # Optional per-instance caches. The decoded image is resized to a square
        # image_size once and reused across all queries for that image; random
        # augmentation still runs per query on the smaller cached image. Metadata
        # tensors are cached to avoid re-reading and re-validating JSON per query.
        self.cache_images = cache_images
        # Dual-pathway vector branch: whole-drawing stroke tokens from
        # schema-v3 metadata. Per-worker RNG makes epoch-to-epoch stroke
        # subsampling stochastic while staying off the main training RNG.
        self.vector_branch = vector_branch
        self.vector_n_max = int(vector_n_max)
        if self.vector_branch and self.vector_n_max <= 0:
            raise ValueError("vector_n_max must be positive when the vector branch is on")
        self._stroke_rng = __import__("random").Random(vector_seed)
        self._stroke_cache: dict[str, torch.Tensor] = {}
        self._image_cache: dict[str, Image.Image] = {}
        self._metadata_cache: dict[
            str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]
        ] = {}
        if records is not None:
            self.records = validate_image_records(
                self.root,
                records,
                strict=strict_metadata,
                manifest_path=manifest_path,
            )
        else:
            self.records = records_for_split(
                self.root,
                self.split,
                manifest_path=manifest_path,
                strict=strict_metadata,
            )
        self.manifest_path, self.split_manifest_fingerprint, self.metadata_fingerprint = (
            _dataset_fingerprints(self.root, manifest_path, self.records)
        )

        # Build positive query index.
        self.index: list[tuple[Path, Path, str]] = []
        self._query_records: list[tuple[ImageRecord, str, bool]] = []
        self.sample_class_ids: list[int] = []
        class_counter: Counter[int] = Counter()
        for record in self.records:
            image_path, metadata_path = _record_paths(self.root, record)
            for class_name in sorted(name for name in record.classes if name in CLASS_TO_IDX):
                class_id = CLASS_TO_IDX[class_name]
                self.index.append((image_path, metadata_path, class_name))
                self._query_records.append((record, class_name, True))
                self.sample_class_ids.append(class_id)
                class_counter[class_id] += 1
        self.class_counts = dict(class_counter)

        if not self.index:
            raise RuntimeError(
                f"No (image, class) queries found in {self.root} for split={self.split!r}. "
                "Generate metadata and, for val, an image-level split manifest first."
            )

        # Build absent-class (negative) query index using a fixed RNG so that
        # the augmented dataset is reproducible regardless of DataLoader shuffle.
        self._neg_records: list[tuple[ImageRecord, str, bool]] = []
        self._neg_class_ids: list[int] = []
        if neg_queries_per_pos > 0:
            all_class_names = [name for name in CLASS_NAMES if name in CLASS_TO_IDX]
            rng = __import__("random").Random(neg_seed)
            for record in self.records:
                present = set(record.classes)
                absent = [name for name in all_class_names if name not in present]
                if not absent:
                    continue
                n_pos = sum(1 for name in record.classes if name in CLASS_TO_IDX)
                n_neg = n_pos * neg_queries_per_pos
                sampled = rng.choices(absent, k=n_neg) if len(absent) >= 1 else []
                for class_name in sampled:
                    self._neg_records.append((record, class_name, False))
                    self._neg_class_ids.append(CLASS_TO_IDX[class_name])

        # Combined view: positives first, then negatives.
        self._all_records: list[tuple[ImageRecord, str, bool]] = (
            self._query_records + self._neg_records
        )
        self._all_class_ids: list[int] = self.sample_class_ids + self._neg_class_ids

    def __len__(self) -> int:
        return len(self._all_records)

    @property
    def num_positive_queries(self) -> int:
        """Number of positive (class-present) queries."""
        return len(self._query_records)

    @property
    def num_negative_queries(self) -> int:
        """Number of absent-class (negative) queries."""
        return len(self._neg_records)

    def get_sample_weights(self, balance_power: float = 0.5) -> torch.Tensor:
        """Return inverse-frequency weights for the full query index.

        Negative queries receive the same weight as the corresponding class so
        the sampler does not systematically over- or under-sample them relative
        to positives. The caller may override by passing custom weights.
        """
        if balance_power < 0:
            raise ValueError("balance_power must be non-negative")
        return torch.tensor(
            [
                max(1, self.class_counts.get(class_id, 1)) ** (-balance_power)
                for class_id in self._all_class_ids
            ],
            dtype=torch.double,
        )

    def _load_metadata_tensors(
        self, metadata_path: Path
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], tuple[int, int]]:
        """Return cached instance tensors and original (width, height)."""

        key = str(metadata_path)
        if self.cache_images and key in self._metadata_cache:
            boxes, labels, instance_ids, class_names, original_size = self._metadata_cache[key]
            return boxes, labels, instance_ids, class_names, original_size
        metadata = load_metadata(
            metadata_path,
            allow_legacy=True,
            strict=self.strict_metadata,
        )
        boxes, labels, instance_ids, class_names = _instances_as_tensors(metadata)
        image_size = metadata.get("image_size")
        original_size = (
            (int(image_size[0]), int(image_size[1]))
            if isinstance(image_size, (list, tuple)) and len(image_size) == 2
            else (0, 0)
        )
        if self.cache_images:
            self._metadata_cache[key] = (
                boxes,
                labels,
                instance_ids,
                class_names,
                original_size,
            )
        return boxes, labels, instance_ids, class_names, original_size

    def _load_stroke_tokens(self, metadata_path: Path) -> torch.Tensor:
        """Return normalized ``[N, 12]`` whole-drawing stroke tokens (cached).

        Schema-v2 metadata has no strokes; the vector branch then degrades to
        an empty token set and the model runs image-only for that sample.
        """

        if not self.vector_branch:
            return torch.zeros((0, STROKE_FEATURE_DIM), dtype=torch.float32)
        key = str(metadata_path)
        cached = self._stroke_cache.get(key)
        if cached is not None:
            return cached
        metadata = load_metadata(metadata_path, allow_legacy=True, strict=self.strict_metadata)
        image_size = metadata.get("image_size")
        size = (
            (int(image_size[0]), int(image_size[1]))
            if isinstance(image_size, (list, tuple)) and len(image_size) == 2
            else None
        )
        strokes = metadata.get("strokes")
        if size is None or not isinstance(strokes, list):
            tokens = torch.zeros((0, STROKE_FEATURE_DIM), dtype=torch.float32)
        else:
            tokens = normalize_strokes(strokes, size)
        self._stroke_cache[key] = tokens
        return tokens


    def _load_resized_image(self, image_path: Path) -> tuple[Image.Image, tuple[int, int]]:
        """Decode an image once and cache the full-resolution RGB copy.

        The full-resolution image is cached (not pre-resized) so the transform's
        antialiased resize stays byte-identical to the uncached path; only the
        expensive PNG decode is amortized across an image's queries.
        """

        key = str(image_path)
        if self.cache_images and key in self._image_cache:
            cached = self._image_cache[key]
            return cached, cached.size
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            if self.cache_images:
                self._image_cache[key] = image
                return image, image.size
            return image.copy(), image.size

    def __getitem__(self, index: int) -> dict[str, Any]:
        record, target_class, is_positive = self._all_records[index]
        image_path, metadata_path = _record_paths(self.root, record)
        all_boxes, _all_labels, _instance_ids, all_class_names, _meta_size = (
            self._load_metadata_tensors(metadata_path)
        )

        if is_positive:
            keep = torch.tensor(
                [class_name == target_class for class_name in all_class_names],
                dtype=torch.bool,
            )
            boxes = all_boxes[keep]
        else:
            # Absent-class query: produce empty targets.
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        image, original_size = self._load_resized_image(image_path)
        image_tensor, boxes = apply_transform(self.transform, image, boxes)

        target_maps, target_stats = generate_centernet_targets(
            boxes,
            image_size=(int(image_tensor.shape[-2]), int(image_tensor.shape[-1])),
            output_stride=self.output_stride,
            min_overlap=self.min_overlap,
            collision_policy=self.collision_policy,
        )
        sample: dict[str, Any] = {
            "image": image_tensor,
            **target_maps,
            "text": TEXT_TEMPLATE.format(cls=target_class),
            "class_id": CLASS_TO_IDX[target_class],
            "class_name": target_class,
            "is_positive": is_positive,
            "sample_id": record.sample_id,
            "image_id": record.image_id,
            "image_path": str(image_path),
            "boxes": boxes,
            "target_stats": target_stats,
            "original_size": original_size,
        }
        if self.vector_branch:
            # Stochastic per-epoch subsampling: drawings above n_max see a
            # different primitive subset each epoch (free augmentation).
            sample["stroke_tokens"] = sample_strokes(
                self._load_stroke_tokens(metadata_path),
                self.vector_n_max,
                generator=self._stroke_rng,
            )
        return sample


def query_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate fixed-size query samples while preserving variable-length boxes."""

    collated = {
        "image": torch.stack([sample["image"] for sample in batch]),
        "center_heatmap": torch.stack([sample["center_heatmap"] for sample in batch]),
        "size_map": torch.stack([sample["size_map"] for sample in batch]),
        "offset_map": torch.stack([sample["offset_map"] for sample in batch]),
        "mask_map": torch.stack([sample["mask_map"] for sample in batch]),
        "texts": [sample["text"] for sample in batch],
        "class_ids": [sample["class_id"] for sample in batch],
        "class_names": [sample["class_name"] for sample in batch],
        "is_positive": [sample.get("is_positive", True) for sample in batch],
        "sample_ids": [sample["sample_id"] for sample in batch],
        "image_ids": [sample["image_id"] for sample in batch],
        "image_paths": [sample["image_path"] for sample in batch],
        "boxes": [sample["boxes"] for sample in batch],
        "target_stats": [sample["target_stats"] for sample in batch],
        "original_sizes": [sample["original_size"] for sample in batch],
    }
    if any("stroke_tokens" in sample for sample in batch):
        tokens, valid = pad_stroke_batch(
            [sample.get("stroke_tokens", torch.zeros((0, STROKE_FEATURE_DIM))) for sample in batch],
            n_max=max(sample["stroke_tokens"].shape[0] for sample in batch if "stroke_tokens" in sample) or 1,
        )
        collated["stroke_tokens"] = tokens
        collated["stroke_mask"] = valid
    return collated


def image_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate image-level evaluation samples without padding annotations."""

    collated = {
        "image": torch.stack([sample["image"] for sample in batch]),
        "boxes": [sample["boxes"] for sample in batch],
        "labels": [sample["labels"] for sample in batch],
        "class_ids": [sample["class_ids"] for sample in batch],
        "instance_ids": [sample["instance_ids"] for sample in batch],
        "class_names": [sample["class_names"] for sample in batch],
        "sample_ids": [sample["sample_id"] for sample in batch],
        "image_ids": [sample["image_id"] for sample in batch],
        "image_paths": [sample["image_path"] for sample in batch],
        "original_sizes": [sample["original_size"] for sample in batch],
        "image_sizes": [sample["image_size"] for sample in batch],
    }
    if any("stroke_tokens" in sample for sample in batch):
        tokens, valid = pad_stroke_batch(
            [sample.get("stroke_tokens", torch.zeros((0, STROKE_FEATURE_DIM))) for sample in batch],
            n_max=max(
                sample["stroke_tokens"].shape[0]
                for sample in batch
                if "stroke_tokens" in sample
            )
            or 1,
        )
        collated["stroke_tokens"] = tokens
        collated["stroke_mask"] = valid
    return collated


# Backward-compatible public API used by train.py and existing launchers.
FloorPlanDataset = FloorPlanQueryDataset
collate_fn = query_collate_fn
ImageLevelDataset = FloorPlanImageDataset
QueryLevelDataset = FloorPlanQueryDataset


__all__ = [
    "CLASS_NAMES",
    "CLASS_TO_IDX",
    "NUM_CLASSES",
    "FloorPlanImageDataset",
    "FloorPlanQueryDataset",
    "FloorPlanDataset",
    "ImageLevelDataset",
    "QueryLevelDataset",
    "image_collate_fn",
    "query_collate_fn",
    "collate_fn",
]
