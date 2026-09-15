"""Build deterministic, leak-free splits for ArchCAD-400K.

Groups images by SHA-256 hash so that identical drawings never cross
train / val / test boundaries (0% train-val leakage by design).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic splits for ArchCAD"
    )
    parser.add_argument("--data-root", type=str, default="data/ArchCAD")
    parser.add_argument("--output", type=str, default="data/ArchCAD/splits.json")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    args = parser.parse_args()

    root = Path(args.data_root)
    meta_dir = root / "metadata"
    meta_files = sorted(list(meta_dir.glob("*_meta.json")))
    print(f"Loading metadata from {len(meta_files)} files...")

    hash_to_images: dict[str, list[str]] = defaultdict(list)
    image_to_hash: dict[str, str] = {}
    image_class_counts: dict[str, dict[str, int]] = {}

    t0 = time.time()
    for mf in meta_files:
        with open(mf, "r", encoding="utf-8") as f:
            d = json.load(f)
        img_id = d["image_id"]
        sha = d["png_sha256"]
        hash_to_images[sha].append(img_id)
        image_to_hash[img_id] = sha
        image_class_counts[img_id] = d.get("class_counts", {})

    print(f"Loaded in {time.time() - t0:.1f}s.")
    print(f"Total image slots: {len(image_to_hash)}")
    print(f"Unique content hashes (groups): {len(hash_to_images)}")
    num_duplicates = sum(len(imgs) - 1 for imgs in hash_to_images.values())
    print(
        f"Duplicate image slots: {num_duplicates} ({num_duplicates / len(image_to_hash) * 100:.2f}%)"
    )

    # Group-level partition using reproducible PRNG
    rng = random.Random(args.seed)
    unique_hashes = sorted(list(hash_to_images.keys()))
    rng.shuffle(unique_hashes)

    total_groups = len(unique_hashes)
    num_val = int(round(total_groups * args.val_fraction))
    num_test = int(round(total_groups * args.test_fraction))

    val_hashes = set(unique_hashes[:num_val])
    test_hashes = set(unique_hashes[num_val : num_val + num_test])
    train_hashes = set(unique_hashes[num_val + num_test :])

    # Assign images based on hash membership
    train_ids = [img for sha in train_hashes for img in hash_to_images[sha]]
    val_ids = [img for sha in val_hashes for img in hash_to_images[sha]]
    test_ids = [img for sha in test_hashes for img in hash_to_images[sha]]

    # Core integrity assertions
    assert train_hashes.isdisjoint(
        val_hashes
    ), "FATAL: Train and Val share content hashes!"
    assert train_hashes.isdisjoint(
        test_hashes
    ), "FATAL: Train and Test share content hashes!"
    assert val_hashes.isdisjoint(
        test_hashes
    ), "FATAL: Val and Test share content hashes!"
    assert len(train_ids) + len(val_ids) + len(test_ids) == len(image_to_hash)

    # Class coverage stats per split
    def count_classes(id_list: list[str]) -> dict[str, int]:
        c: dict[str, int] = defaultdict(int)
        for iid in id_list:
            for cls, cnt in image_class_counts.get(iid, {}).items():
                c[cls] += cnt
        return dict(c)

    train_cls = count_classes(train_ids)
    val_cls = count_classes(val_ids)
    test_cls = count_classes(test_ids)

    # Compute manifest fingerprint
    manifest_bytes = json.dumps(
        {
            "seed": args.seed,
            "train": sorted(train_ids),
            "val": sorted(val_ids),
            "test": sorted(test_ids),
        },
        sort_keys=True,
    ).encode("utf-8")
    manifest_fingerprint = hashlib.sha256(manifest_bytes).hexdigest()

    split_manifest = {
        "dataset": "ArchCAD-400K",
        "schema_version": "v2",
        "seed": args.seed,
        "strategy": "sha256_group_partition",
        "fingerprint": manifest_fingerprint,
        "num_total_images": len(image_to_hash),
        "num_unique_hashes": len(hash_to_images),
        "splits": {
            "train": {
                "count": len(train_ids),
                "num_groups": len(train_hashes),
                "images": sorted(train_ids),
                "class_distribution": train_cls,
            },
            "val": {
                "count": len(val_ids),
                "num_groups": len(val_hashes),
                "images": sorted(val_ids),
                "class_distribution": val_cls,
            },
            "test": {
                "count": len(test_ids),
                "num_groups": len(test_hashes),
                "images": sorted(test_ids),
                "class_distribution": test_cls,
            },
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)

    print(f"\nManifest successfully created at {out_path}")
    print(f"Fingerprint: {manifest_fingerprint}")
    print(f"  Train: {len(train_ids)} images ({len(train_hashes)} groups)")
    print(f"  Val:   {len(val_ids)} images ({len(val_hashes)} groups)")
    print(f"  Test:  {len(test_ids)} images ({len(test_hashes)} groups)")
    print(
        "Train/Val content leakage: EXACTLY 0.00% (Guaranteed by group hash partition)."
    )


if __name__ == "__main__":
    main()
