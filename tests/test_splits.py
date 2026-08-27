from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from PIL import Image

from src.data.constants import CLASS_TO_IDX
from src.data.splits import (
    assert_no_image_leakage,
    build_split_manifest,
    build_split_manifest_v2,
    image_index_fingerprint,
    index_images,
    split_image_index,
    split_image_index_v2,
    split_manifest_fingerprint,
    split_role_fingerprints,
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


def test_v2_roles_are_deterministic_and_partition_images(tmp_path: Path) -> None:
    _make_data(tmp_path)
    records = index_images(tmp_path)
    first = split_image_index_v2(
        records,
        seed=1337,
        val_select_fraction=0.10,
        val_calibration_fraction=0.10,
    )
    second = split_image_index_v2(
        records,
        seed=1337,
        val_select_fraction=0.10,
        val_calibration_fraction=0.10,
    )

    roles = ("train_new", "val_select", "val_calibration", "test_set")
    assert all(
        [record.image_id for record in first[role]]
        == [record.image_id for record in second[role]]
        for role in roles
    )
    assert len(first["train_new"]) == 16
    assert len(first["val_select"]) == 2
    assert len(first["val_calibration"]) == 2
    assert len(first["test_set"]) == 3
    role_ids = [{record.image_id for record in first[role]} for role in roles]
    assert set().union(*role_ids) == {
        record.image_id for record in records
    }
    assert sum(len(ids) for ids in role_ids) == len(records)
    assert all(
        record.source_split == "train"
        for role in ("train_new", "val_select", "val_calibration")
        for record in first[role]
    )
    assert all(record.source_split == "test" for record in first["test_set"])


def test_v2_manifest_records_role_provenance_and_rejects_tampering(tmp_path: Path) -> None:
    _make_data(tmp_path)
    manifest = build_split_manifest_v2(tmp_path)
    validate_split_manifest(manifest)

    assert manifest["schema_version"] == 2
    assert set(manifest["splits"]) == {
        "train_new",
        "val_select",
        "val_calibration",
        "test_set",
    }
    assert manifest["role_fingerprints"] == split_role_fingerprints(
        manifest["splits"]
    )
    assert manifest["role_fingerprints"]["test_set"] == image_index_fingerprint(
        manifest["splits"]["test_set"]
    )

    tampered = deepcopy(manifest)
    tampered["splits"]["val_select"][0]["num_instances"] += 1
    tampered["role_fingerprints"] = split_role_fingerprints(tampered["splits"])
    tampered["fingerprint"] = split_manifest_fingerprint(tampered)
    try:
        validate_split_manifest(tampered)
    except ValueError as error:
        assert "differs from image_index" in str(error)
    else:
        raise AssertionError("tampered v2 role record was accepted")

    missing_fingerprint = deepcopy(manifest)
    del missing_fingerprint["fingerprint"]
    try:
        validate_split_manifest(missing_fingerprint)
    except ValueError as error:
        assert "requires a non-empty fingerprint" in str(error)
    else:
        raise AssertionError("v2 manifest without fingerprint was accepted")
