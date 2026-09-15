"""Build a content-deduplicated split manifest (schema v3).

The schema-v2 manifest splits by image id, but FloorPlanCAD re-uses the same PNG
under different ids, so byte-identical content can straddle roles and inflate
validation. This script makes the content group the split unit: it groups records
by PNG SHA-256, collapses each group to one canonical record, runs the *existing*
schema-v2 splitter on those units, then expands back so the manifest still lists
every physical image.

``splits_v2.json`` is never modified. Output defaults to ``splits_v3.json``.

Usage:
    python scripts/data/dedup_splits.py --data-root data/FloorPlanCAD_original \
        --report outputs/dedup_report.json --validate-only
    python scripts/data/dedup_splits.py --data-root data/FloorPlanCAD_original \
        --output data/FloorPlanCAD_original/splits_v3.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dedup import (  # noqa: E402
    build_split_units,
    count_cross_role_duplicates,
    expand_units_to_members,
    group_by_content,
)
from src.data.splits import (  # noqa: E402
    SPLIT_MANIFEST_VERSION_V2,
    TRAIN_SOURCE_DIRS,
    V2_ROLE_PURPOSES,
    V2_SPLIT_ROLES,
    ImageRecord,
    class_distribution_report,
    index_images,
    split_image_index_v2,
    split_manifest_fingerprint,
    split_role_fingerprints,
)

V3_SCHEMA_VERSION = 3
V3_PROTOCOL = "floorplan_detection_validation_hierarchy_v3_content_dedup"


def build_dedup_manifest(
    data_root: str | Path,
    *,
    seed: int,
    val_select_fraction: float,
    val_calibration_fraction: float,
    strict: bool = False,
    verbose: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (manifest, report) for the deduplicated split hierarchy.

    Only the **training pool** is deduplicated. ``test_set`` is copied unchanged
    from ``splits_v2.json`` so that historical test results stay comparable and
    the held-out set does not silently grow. Content groups that straddle
    ``test_set`` and the training pool are therefore *reported* rather than
    fixed — moving their test members would change the held-out set.
    """

    root = Path(data_root)
    if verbose:
        print(f"[1/7] indexing images under {root} ...")
    records = index_images(root, strict=strict)
    by_id = {record.image_id: record for record in records}
    if verbose:
        print(f"      {len(records)} physical images")

    # The frozen v2 manifest defines which images are held out. Read it so the
    # test role is inherited verbatim rather than re-derived.
    v2_path = root / "splits_v2.json"
    if not v2_path.is_file():
        raise FileNotFoundError(
            f"{v2_path} is required: test_set is inherited from the frozen v2 manifest"
        )
    v2 = json.loads(v2_path.read_text(encoding="utf-8"))
    v2_test_ids = [record["image_id"] for record in v2["splits"]["test_set"]]
    if verbose:
        print(f"      inherited test_set from v2: {len(v2_test_ids)} images")

    training_pool = [
        record for record in records if record.image_id not in set(v2_test_ids)
    ]
    if verbose:
        print(f"[2/7] training pool: {len(training_pool)} images")

    if verbose:
        print("[3/7] hashing training-pool PNGs (SHA-256) ...")

    def progress(done: int) -> None:
        print(f"      hashed {done} images ...", flush=True)

    groups = group_by_content(
        training_pool, root=root, on_progress=progress if verbose else None
    )
    duplicate_members = sum(group.size for group in groups.values())
    if verbose:
        print(
            f"      {len(groups)} duplicate groups covering {duplicate_members} images "
            f"({duplicate_members - len(groups)} redundant slots)"
        )

    if verbose:
        print("[4/7] collapsing groups to split units ...")
    units, member_to_unit = build_split_units(training_pool, groups)
    if verbose:
        print(f"      {len(training_pool)} images -> {len(units)} split units")

    if verbose:
        print("[5/7] splitting units with the existing v2 strategy ...")
    unit_splits = split_image_index_v2(
        units,
        seed=seed,
        val_select_fraction=val_select_fraction,
        val_calibration_fraction=val_calibration_fraction,
    )

    if verbose:
        print("[6/7] expanding units back to physical images ...")
    expanded = expand_units_to_members(unit_splits, groups, by_id)
    # The splitter always emits a `test_set` key; since only training-pool records
    # were passed it must be empty. It is then filled from the frozen v2 manifest.
    if expanded.pop("test_set", None):
        raise AssertionError("splitter produced test records; only the pool was passed")
    expanded["test_set"] = sorted(
        (by_id[image_id] for image_id in v2_test_ids),
        key=lambda record: record.image_id,
    )

    if verbose:
        print("[7/7] verifying ...")
    _assert_no_cross_role_content(
        {role: records_ for role, records_ in expanded.items() if role != "test_set"},
        groups,
    )
    if set(expanded) != set(V2_SPLIT_ROLES):
        raise AssertionError(f"role mismatch: {sorted(expanded)} != {sorted(V2_SPLIT_ROLES)}")
    total_out = sum(len(v) for v in expanded.values())
    if total_out != len(records):
        raise AssertionError(f"image count changed: {total_out} != {len(records)}")
    out_test_ids = sorted(record.image_id for record in expanded["test_set"])
    if out_test_ids != sorted(v2_test_ids):
        raise AssertionError("test_set changed; it must be inherited verbatim from v2")

    serialized = {
        role: [record.to_dict() for record in expanded[role]] for role in V2_SPLIT_ROLES
    }
    manifest: dict[str, Any] = {
        "schema_version": V3_SCHEMA_VERSION,
        "protocol": V3_PROTOCOL,
        "seed": seed,
        "fractions": {
            "val_select": val_select_fraction,
            "val_calibration": val_calibration_fraction,
        },
        "split_strategy": "content_dedup_training_pool_then_greedy_roles_v2",
        "dedup": {
            "unit": "png_sha256",
            "scope": "training_pool_only",
            "test_set_source": "inherited_from_splits_v2_unchanged",
            "duplicate_groups": len(groups),
            "images_in_duplicate_groups": duplicate_members,
            "redundant_slots": duplicate_members - len(groups),
            "split_units": len(units),
        },
        "source_dirs": {"train": list(TRAIN_SOURCE_DIRS), "test": ["test_set"]},
        "roles": {role: {"purpose": V2_ROLE_PURPOSES[role]} for role in V2_SPLIT_ROLES},
        "image_index": [record.to_dict() for record in records],
        "splits": serialized,
        "role_fingerprints": split_role_fingerprints(expanded),
        "class_distribution": class_distribution_report(expanded),
    }
    manifest["fingerprint"] = split_manifest_fingerprint(manifest)

    residual = count_cross_role_duplicates(expanded, groups)
    inherited = _inherited_test_duplicates(root, v2, groups)
    historical_overlap = _historical_checkpoint_overlap(v2, expanded)
    report: dict[str, Any] = {
        "schema_version": V3_SCHEMA_VERSION,
        "seed": seed,
        "fractions": {
            "val_select": val_select_fraction,
            "val_calibration": val_calibration_fraction,
        },
        "totals": {
            "physical_images": len(records),
            "training_pool": len(training_pool),
            "split_units": len(units),
            "duplicate_groups": len(groups),
            "images_in_duplicate_groups": duplicate_members,
            "redundant_slots": duplicate_members - len(groups),
        },
        "role_sizes": {role: len(expanded[role]) for role in V2_SPLIT_ROLES},
        "group_size_histogram": dict(
            sorted(Counter(group.size for group in groups.values()).items())
        ),
        "residual_cross_role_duplicates": residual,
        "inherited_test_duplicates": inherited,
        "historical_checkpoint_overlap": historical_overlap,
        "manifest_fingerprint": manifest["fingerprint"],
        "role_fingerprints": manifest["role_fingerprints"],
    }
    return manifest, report


def _historical_checkpoint_overlap(
    v2: dict[str, Any],
    v3_splits: dict[str, list[ImageRecord]],
) -> dict[str, Any]:
    """Quantify why a v2-trained checkpoint cannot be a clean v3 baseline.

    Repartitioning content groups changes role membership. A checkpoint trained
    on v2 may therefore have seen physical images that v3 assigns to validation,
    even though the v3 roles themselves are mutually content-disjoint.
    """

    v2_ids = {
        role: {record["image_id"] for record in v2["splits"][role]}
        for role in ("train_new", "val_select", "val_calibration")
    }
    v3_ids = {
        role: {record.image_id for record in v3_splits[role]}
        for role in ("train_new", "val_select", "val_calibration")
    }

    seen_validation: dict[str, Any] = {}
    for role in ("val_select", "val_calibration"):
        overlap = sorted(v3_ids[role] & v2_ids["train_new"])
        seen_validation[role] = {
            "images_seen_in_v2_train": len(overlap),
            "role_images": len(v3_ids[role]),
            "fraction": len(overlap) / len(v3_ids[role]) if v3_ids[role] else 0.0,
            "examples": overlap[:10],
        }

    return {
        "v3_validation_seen_by_v2_trained_checkpoint": seen_validation,
        "conclusion": (
            "A checkpoint trained on splits_v2.json is not a clean v3 baseline. "
            "Train from scratch on splits_v3.json before interpreting validation AP."
        ),
    }


def _inherited_test_duplicates(
    root: Path,
    v2: dict[str, Any],
    groups: dict[str, Any],
) -> dict[str, Any]:
    """Report test images that share content with the training pool.

    These are NOT fixed: test_set is inherited verbatim from v2. They are
    recorded so the residual leak is explicit rather than silent.
    """

    train_hashes = set(groups)
    share: list[str] = []
    for record in v2["splits"]["test_set"]:
        image_path = Path(record["image_path"])
        path = image_path if image_path.is_absolute() else root / image_path
        from src.data.dedup import sha256_file

        if sha256_file(path) in train_hashes:
            share.append(record["image_id"])
    n_test = len(v2["splits"]["test_set"])
    return {
        "test_images_sharing_content_with_train_pool": len(share),
        "test_images_total": n_test,
        "fraction": (len(share) / n_test) if n_test else 0.0,
        "examples": sorted(share)[:10],
        "note": (
            "NOT fixed by design: test_set is inherited unchanged from v2 so historical "
            "test results stay comparable. Re-splitting test would invalidate them."
        ),
    }


def _assert_no_cross_role_content(
    role_splits: dict[str, list[ImageRecord]],
    groups: dict[str, Any],
) -> None:
    """Fail if any byte-identical group spans two roles."""

    owner: dict[str, str] = {}
    for role, records in role_splits.items():
        for record in records:
            owner[record.image_id] = role
    violations: list[str] = []
    for group in groups.values():
        roles = sorted({owner[mid] for mid in group.image_ids if mid in owner})
        if len(roles) > 1:
            violations.append(f"{group.image_ids[:3]} -> {roles}")
    if violations:
        raise AssertionError(
            f"content leakage across roles after dedup: {len(violations)} groups, "
            f"e.g. {violations[:3]}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", "--data_root", dest="data_root",
                        default="./data/FloorPlanCAD_original")
    parser.add_argument("--output", default=None,
                        help="Manifest path (default: <data-root>/splits_v3.json)")
    parser.add_argument("--report", default=None, help="Write the JSON report here")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--val-select-fraction", "--val_select_fraction",
                        dest="val_select_fraction", type=float, default=0.125)
    parser.add_argument("--val-calibration-fraction", "--val_calibration_fraction",
                        dest="val_calibration_fraction", type=float, default=0.125)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--validate-only", "--validate_only", dest="validate_only",
                        action="store_true",
                        help="Build and verify in memory without writing the manifest")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing manifest")
    args = parser.parse_args()

    manifest, report = build_dedup_manifest(
        args.data_root,
        seed=args.seed,
        val_select_fraction=args.val_select_fraction,
        val_calibration_fraction=args.val_calibration_fraction,
        strict=args.strict,
    )

    print()
    print("=" * 68)
    print("DEDUP RESULT")
    print("=" * 68)
    t = report["totals"]
    print(f"  physical images        : {t['physical_images']}")
    print(f"  split units (deduped)  : {t['split_units']}")
    print(f"  duplicate groups       : {t['duplicate_groups']}")
    print(f"  images in dup groups   : {t['images_in_duplicate_groups']}")
    print(f"  redundant slots removed: {t['redundant_slots']}")
    print()
    print("  role sizes:")
    for role in V2_SPLIT_ROLES:
        print(f"    {role:18s} {report['role_sizes'][role]}")
    print()
    resid = report["residual_cross_role_duplicates"]
    print(f"  residual cross-role dup groups: {resid['straddling_group_count']}")
    if resid["role_pair_counts"]:
        for pair, count in resid["role_pair_counts"].items():
            print(f"    {pair}: {count}")
    print()
    print(f"  manifest fingerprint: {report['manifest_fingerprint']}")
    print("=" * 68)

    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report -> {path}")

    if args.validate_only:
        print("validate-only: manifest NOT written")
        return 0

    out = Path(args.output) if args.output else Path(args.data_root) / "splits_v3.json"
    if out.exists() and not args.force:
        print(f"ERROR: {out} exists; pass --force to overwrite", file=sys.stderr)
        return 1
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
