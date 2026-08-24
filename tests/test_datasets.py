from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from src.data.constants import CLASS_TO_IDX
from src.data.dataset import (
    FloorPlanDataset,
    FloorPlanImageDataset,
    FloorPlanQueryDataset,
    image_collate_fn,
    query_collate_fn,
)
from src.data.splits import build_split_manifest, write_split_manifest
from src.data.transforms import ResizeNormalize


def _add_sample(directory: Path, stem: str, boxes_by_class: dict[str, list[list[int]]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 32), "white").save(directory / f"{stem}.png")
    instances = []
    instance_id = 0
    for class_name, boxes in boxes_by_class.items():
        for box in boxes:
            instances.append(
                {
                    "class": class_name,
                    "class_id": CLASS_TO_IDX[class_name],
                    "instance_id": instance_id,
                    "bbox_px": box,
                }
            )
            instance_id += 1
    metadata = {
        "image_size": [64, 32],
        "svg_viewbox": [0, 0, 64, 32],
        "num_instances": len(instances),
        "instances": instances,
    }
    (directory / f"{stem}_meta.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )


def _make_dataset(root: Path) -> Path:
    for index in range(10):
        classes = {"door_single": [[4, 4, 20, 16]]}
        if index == 0:
            classes["chair"] = [[24, 8, 40, 24]]
        _add_sample(root / "train_set_1", f"train{index:02d}", classes)
    _add_sample(
        root / "test_set",
        "test00",
        {
            "table": [[8, 4, 32, 20], [36, 8, 60, 28]],
            "window": [[2, 2, 6, 28]],
        },
    )
    manifest_path = root / "splits.json"
    write_split_manifest(build_split_manifest(root), manifest_path)
    return manifest_path


def test_image_level_dataset_yields_each_image_once(tmp_path: Path) -> None:
    manifest_path = _make_dataset(tmp_path)
    dataset = FloorPlanImageDataset(
        tmp_path,
        split="test",
        image_size=64,
        manifest_path=manifest_path,
    )
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["image"].shape == (3, 64, 64)
    assert sample["boxes"].shape == (3, 4)
    assert sample["labels"].shape == (3,)
    assert sample["image_id"] == "test_set/test00"

    batch = image_collate_fn([sample, sample])
    assert batch["image"].shape == (2, 3, 64, 64)
    assert len(batch["boxes"]) == 2


def test_query_dataset_expands_only_after_image_split(tmp_path: Path) -> None:
    manifest_path = _make_dataset(tmp_path)
    train = FloorPlanQueryDataset(
        tmp_path,
        split="train",
        image_size=64,
        output_stride=8,
        transform=ResizeNormalize(64),
        manifest_path=manifest_path,
    )
    val = FloorPlanQueryDataset(
        tmp_path,
        split="val",
        image_size=64,
        output_stride=8,
        transform=ResizeNormalize(64),
        manifest_path=manifest_path,
    )
    train_ids = {record.image_id for record in train.records}
    val_ids = {record.image_id for record in val.records}
    assert train_ids.isdisjoint(val_ids)
    assert len(train_ids) == 9
    assert len(val_ids) == 1
    assert len(train) == sum(len(record.classes) for record in train.records)
    assert len(val) == sum(len(record.classes) for record in val.records)


def test_query_targets_alias_common_constructor_and_collate(tmp_path: Path) -> None:
    manifest_path = _make_dataset(tmp_path)
    assert FloorPlanDataset is FloorPlanQueryDataset
    dataset = FloorPlanDataset(
        tmp_path,
        split="test",
        image_size=64,
        output_stride=8,
        transform=ResizeNormalize(64),
        manifest_path=manifest_path,
    )
    assert len(dataset) == 2
    table_index = next(
        index for index, (_, _, class_name) in enumerate(dataset.index) if class_name == "table"
    )
    sample = dataset[table_index]
    assert sample["boxes"].shape == (2, 4)
    assert sample["center_heatmap"].shape == (1, 8, 8)
    assert sample["size_map"].shape == (2, 8, 8)
    assert sample["offset_map"].shape == (2, 8, 8)
    assert sample["mask_map"].shape == (1, 8, 8)
    assert sample["class_id"] == CLASS_TO_IDX["table"]
    assert sample["target_stats"].total_boxes == 2

    batch = query_collate_fn([sample, sample])
    assert batch["image"].shape == (2, 3, 64, 64)
    assert batch["center_heatmap"].shape == (2, 1, 8, 8)
    assert batch["class_ids"] == [CLASS_TO_IDX["table"]] * 2
    assert all(isinstance(boxes, torch.Tensor) for boxes in batch["boxes"])
