"""Unit test to verify zero-leakage and validity of ArchCAD splits."""

import json
from pathlib import Path
import pytest


def test_archcad_splits_integrity():
    split_path = Path("data/ArchCAD/splits.json")
    if not split_path.exists():
        pytest.skip("ArchCAD splits.json does not exist yet.")

    with open(split_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["dataset"] == "ArchCAD-400K"
    assert manifest["strategy"] == "sha256_group_partition"

    splits = manifest["splits"]
    train_imgs = set(splits["train"]["images"])
    val_imgs = set(splits["val"]["images"])
    test_imgs = set(splits["test"]["images"])

    # Disjoint image IDs
    assert train_imgs.isdisjoint(val_imgs), "Train and Val share image IDs!"
    assert train_imgs.isdisjoint(test_imgs), "Train and Test share image IDs!"
    assert val_imgs.isdisjoint(test_imgs), "Val and Test share image IDs!"

    total_images = len(train_imgs) + len(val_imgs) + len(test_imgs)
    assert total_images == manifest["num_total_images"]

    # Verify hashes from metadata
    meta_dir = Path("data/ArchCAD/metadata")
    if meta_dir.exists():
        train_hashes = set()
        for iid in list(train_imgs)[:100]:
            with open(meta_dir / f"{iid}_meta.json", "r", encoding="utf-8") as f:
                train_hashes.add(json.load(f)["png_sha256"])

        val_hashes = set()
        for iid in list(val_imgs)[:100]:
            with open(meta_dir / f"{iid}_meta.json", "r", encoding="utf-8") as f:
                val_hashes.add(json.load(f)["png_sha256"])

        assert train_hashes.isdisjoint(
            val_hashes
        ), "Zero-leakage violated: Sampled train and val share SHA-256!"
