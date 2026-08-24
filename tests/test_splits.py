from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from src.data.constants import CLASS_TO_IDX
from src.data.splits import (
    assert_no_image_leakage,
    build_split_manifest,
    index_images,
    split_image_index,
    validate_split_manifest,
)


def _add_sample(directory: Path, stem: str, classes: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), "white").save(directory / f"{stem}.png")
    instances = [
        {
            "class": class_name,
            "class_id": CLASS_TO_IDX[class_name],
            "instance_id": index,
            "bbox_px": [index + 1, 1, index + 8, 10],
        }
        for index, class_name in enumerate(classes)
    ]
    metadata = {
        "image_size": [32, 24],
        "svg_viewbox": [0, 0, 32, 24],
        "num_instances": len(instances),
        "instances": instances,
    }
    (directory / f"{stem}_meta.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )


def _make_data(root: Path) -> None:
    for index in range(10):
        classes = ["door_single"]
        if index % 2 == 0:
            classes.append("chair")
        if index in (0, 1):
            classes.append("bathtub")
        _add_sample(root / "train_set_1", f"a{index:02d}", classes)
    for index in range(10):
        _add_sample(root / "train_set_2", f"b{index:02d}", ["window"])
    for index in range(3):
        _add_sample(root / "test_set", f"t{index:02d}", ["table"])


def test_deterministic_image_level_split_and_untouched_test(tmp_path: Path) -> None:
    _make_data(tmp_path)
    records = index_images(tmp_path)
    first = split_image_index(records, seed=1337, val_fraction=0.10)
    second = split_image_index(records, seed=1337, val_fraction=0.10)

    assert [record.image_id for record in first["val"]] == [
        record.image_id for record in second["val"]
    ]
    assert len(first["train"]) == 18
    assert len(first["val"]) == 2
    assert {record.image_id for record in first["test"]} == {
        f"test_set/t{index:02d}" for index in range(3)
    }
    assert all(record.source_split == "train" for record in first["train"] + first["val"])
    assert all(record.source_split == "test" for record in first["test"])
    assert any("bathtub" in record.classes for record in first["val"])
    assert any("bathtub" in record.classes for record in first["train"])
    assert_no_image_leakage(first)


def test_manifest_has_no_query_expansion_and_reports_class_distribution(tmp_path: Path) -> None:
    _make_data(tmp_path)
    manifest = build_split_manifest(tmp_path)
    validate_split_manifest(manifest)

    all_records = sum((manifest["splits"][name] for name in ("train", "val", "test")), [])
    assert len(all_records) == 23
    assert len({record["image_id"] for record in all_records}) == 23
    assert all("classes" in record and "class_counts" in record for record in all_records)

    report = manifest["class_distribution"]
    assert report["test"]["images"] == 3
    assert report["test"]["queries"] == 3
    assert report["test"]["classes"]["table"] == {"images": 3, "instances": 3}
