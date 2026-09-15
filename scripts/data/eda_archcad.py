"""Comprehensive Exploratory Data Analysis (EDA) for ArchCAD-400K.

Computes:
1. Dataset Scale & Image Attributes
2. Instance & Class Frequency Distribution (Long-tail analysis)
3. Object Bounding Box Geometric Distributions (Size, Aspect Ratio, Area)
4. Stride-8 vs Stride-4 Center Collision Audit (CenterNet bottleneck analysis)
5. Vector Stroke Primitives Distribution (LINE, ARC, CIRCLE)
6. Split Consistency & Class Representation
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.archcad_constants import (
    ARCHCAD_COUNTABLE_CLASSES,
    ARCHCAD_IMAGE_SIZE,
    NUM_ARCHCAD_CLASSES,
)


def analyze_meta_file(meta_path: str) -> dict:
    """Analyze a single _meta.json file."""
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    img_w, img_h = meta.get("image_size", ARCHCAD_IMAGE_SIZE)
    instances = meta.get("instances", [])
    strokes = meta.get("strokes", [])
    class_counts = meta.get("class_counts", {})
    num_inst = len(instances)

    # Box geometries
    boxes_by_class: dict[str, list[list[float]]] = defaultdict(list)
    box_metrics: list[dict] = []

    # Stride-8 and Stride-4 center collision tracking
    # Centers in stride-8 and stride-4 grids:
    # Within-class collisions
    class_grid8: dict[str, set[tuple[int, int]]] = defaultdict(set)
    class_grid4: dict[str, set[tuple[int, int]]] = defaultdict(set)
    within_class_col8 = 0
    within_class_col4 = 0

    # Cross-class collisions
    all_grid8: set[tuple[int, int]] = set()
    all_grid4: set[tuple[int, int]] = set()
    cross_class_col8 = 0
    cross_class_col4 = 0

    for inst in instances:
        c_name = inst.get("category_name")
        bbox = inst.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        x0, y0, x1, y1 = bbox
        w = max(0.0, x1 - x0)
        h = max(0.0, y1 - y0)
        area = w * h
        min_side = min(w, h)
        max_side = max(w, h)
        aspect_ratio = w / max(1e-3, h)

        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0

        # Grid cells
        cell8 = (int(cx // 8), int(cy // 8))
        cell4 = (int(cx // 4), int(cy // 4))

        # Check within-class collision
        if cell8 in class_grid8[c_name]:
            within_class_col8 += 1
        else:
            class_grid8[c_name].add(cell8)

        if cell4 in class_grid4[c_name]:
            within_class_col4 += 1
        else:
            class_grid4[c_name].add(cell4)

        # Check cross-class collision
        if cell8 in all_grid8:
            cross_class_col8 += 1
        else:
            all_grid8.add(cell8)

        if cell4 in all_grid4:
            cross_class_col4 += 1
        else:
            all_grid4.add(cell4)

        box_info = {
            "class": c_name,
            "w": w,
            "h": h,
            "area": area,
            "min_side": min_side,
            "max_side": max_side,
            "aspect_ratio": aspect_ratio,
        }
        boxes_by_class[c_name].append([w, h, area, min_side, max_side, aspect_ratio])
        box_metrics.append(box_info)

    # Stroke primitives count by type (slot 11: 0.0=LINE, 1.0=ARC/CIRCLE)
    num_lines = 0
    num_arcs = 0
    for s in strokes:
        if len(s) >= 12 and s[11] == 1.0:
            num_arcs += 1
        else:
            num_lines += 1

    return {
        "image_id": meta["image_id"],
        "num_instances": num_inst,
        "class_counts": class_counts,
        "classes_present": list(class_counts.keys()),
        "within_class_col8": within_class_col8,
        "within_class_col4": within_class_col4,
        "cross_class_col8": cross_class_col8,
        "cross_class_col4": cross_class_col4,
        "num_strokes": len(strokes),
        "num_lines": num_lines,
        "num_arcs": num_arcs,
        "boxes_by_class": dict(boxes_by_class),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA on ArchCAD-400K dataset")
    parser.add_argument("--data-root", type=str, default="data/ArchCAD")
    parser.add_argument("--output", type=str, default="outputs/eda_archcad_report.json")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    root = Path(args.data_root)
    meta_dir = root / "metadata"
    meta_files = sorted(list(meta_dir.glob("*_meta.json")))
    if args.limit:
        meta_files = meta_files[:args.limit]

    total_images = len(meta_files)
    print(f"Starting EDA on {total_images} ArchCAD metadata files using {args.workers} workers...")
    t0 = time.time()

    file_paths = [str(f) for f in meta_files]

    total_instances = 0
    class_instance_counts: Counter[str] = Counter()
    class_image_counts: Counter[str] = Counter()
    instances_per_image: list[int] = []
    strokes_per_image: list[int] = []
    lines_per_image: list[int] = []
    arcs_per_image: list[int] = []

    total_within_col8 = 0
    total_within_col4 = 0
    total_cross_col8 = 0
    total_cross_col4 = 0

    # Aggregate box statistics per class: [w, h, area, min_side, max_side, aspect_ratio]
    all_class_box_stats: dict[str, list[list[float]]] = defaultdict(list)
    co_occurrence: dict[str, Counter[str]] = defaultdict(Counter)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for i, res in enumerate(executor.map(analyze_meta_file, file_paths, chunksize=200)):
            n_inst = res["num_instances"]
            total_instances += n_inst
            instances_per_image.append(n_inst)
            strokes_per_image.append(res["num_strokes"])
            lines_per_image.append(res["num_lines"])
            arcs_per_image.append(res["num_arcs"])

            total_within_col8 += res["within_class_col8"]
            total_within_col4 += res["within_class_col4"]
            total_cross_col8 += res["cross_class_col8"]
            total_cross_col4 += res["cross_class_col4"]

            for cls, cnt in res["class_counts"].items():
                class_instance_counts[cls] += cnt
                class_image_counts[cls] += 1

            # Co-occurrence
            present = res["classes_present"]
            for c1 in present:
                for c2 in present:
                    if c1 != c2:
                        co_occurrence[c1][c2] += 1

            # Box metrics by class
            for c_name, b_list in res["boxes_by_class"].items():
                all_class_box_stats[c_name].extend(b_list)

            if (i + 1) % 10000 == 0 or (i + 1) == total_images:
                elapsed = time.time() - t0
                print(f"  Processed {i + 1}/{total_images} ({ (i + 1) / elapsed:.1f} imgs/s)...")

    elapsed = time.time() - t0
    print(f"\nCompleted aggregation in {elapsed:.1f}s.")

    # Calculate percentiles and summary distributions
    instances_arr = np.array(instances_per_image)
    strokes_arr = np.array(strokes_per_image)
    lines_arr = np.array(lines_per_image)
    arcs_arr = np.array(arcs_per_image)

    empty_images = int(np.sum(instances_arr == 0))

    # Box dimension statistics per class
    class_geometry_summary = {}
    for c_name in ARCHCAD_COUNTABLE_CLASSES:
        b_list = all_class_box_stats.get(c_name, [])
        if not b_list:
            class_geometry_summary[c_name] = {"count": 0}
            continue
        arr = np.array(b_list)
        # cols: 0: w, 1: h, 2: area, 3: min_side, 4: max_side, 5: aspect_ratio
        class_geometry_summary[c_name] = {
            "count": len(b_list),
            "width_median": round(float(np.median(arr[:, 0])), 1),
            "width_mean": round(float(np.mean(arr[:, 0])), 1),
            "height_median": round(float(np.median(arr[:, 1])), 1),
            "height_mean": round(float(np.mean(arr[:, 1])), 1),
            "min_side_median": round(float(np.median(arr[:, 3])), 1),
            "min_side_p10": round(float(np.percentile(arr[:, 3], 10)), 1),
            "min_side_p90": round(float(np.percentile(arr[:, 3], 90)), 1),
            "aspect_ratio_median": round(float(np.median(arr[:, 5])), 2),
            "area_median": round(float(np.median(arr[:, 2])), 1),
            "sub_stride8_pct": round(float(np.mean(arr[:, 3] < 8.0) * 100), 2),
            "sub_stride4_pct": round(float(np.mean(arr[:, 3] < 4.0) * 100), 2),
        }

    # Center collision rates
    within_col8_rate = (total_within_col8 / max(1, total_instances)) * 100
    within_col4_rate = (total_within_col4 / max(1, total_instances)) * 100
    cross_col8_rate = (total_cross_col8 / max(1, total_instances)) * 100
    cross_col4_rate = (total_cross_col4 / max(1, total_instances)) * 100

    report = {
        "dataset_name": "ArchCAD-400K (40K Initial Release)",
        "total_images": total_images,
        "total_instances": total_instances,
        "empty_images_count": empty_images,
        "empty_images_pct": round(empty_images / total_images * 100, 2),
        "instances_per_image": {
            "mean": round(float(np.mean(instances_arr)), 2),
            "std": round(float(np.std(instances_arr)), 2),
            "median": int(np.median(instances_arr)),
            "p10": int(np.percentile(instances_arr, 10)),
            "p25": int(np.percentile(instances_arr, 25)),
            "p75": int(np.percentile(instances_arr, 75)),
            "p90": int(np.percentile(instances_arr, 90)),
            "max": int(np.max(instances_arr)),
        },
        "strokes_per_image": {
            "mean": round(float(np.mean(strokes_arr)), 1),
            "median": int(np.median(strokes_arr)),
            "max": int(np.max(strokes_arr)),
            "lines_pct": round(float(np.sum(lines_arr) / max(1, np.sum(strokes_arr)) * 100), 1),
            "arcs_pct": round(float(np.sum(arcs_arr) / max(1, np.sum(strokes_arr)) * 100), 1),
        },
        "collision_audit": {
            "within_class_stride8_collisions": total_within_col8,
            "within_class_stride8_rate_pct": round(within_col8_rate, 2),
            "within_class_stride4_collisions": total_within_col4,
            "within_class_stride4_rate_pct": round(within_col4_rate, 2),
            "cross_class_stride8_rate_pct": round(cross_col8_rate, 2),
            "cross_class_stride4_rate_pct": round(cross_col4_rate, 2),
            "stride4_collision_reduction_pct": round(
                (1.0 - (within_col4_rate / max(1e-5, within_col8_rate))) * 100, 2
            ),
        },
        "class_frequencies": {
            cls: {
                "instance_count": class_instance_counts[cls],
                "image_count": class_image_counts[cls],
                "instance_share_pct": round(class_instance_counts[cls] / max(1, total_instances) * 100, 2),
                "image_presence_pct": round(class_image_counts[cls] / total_images * 100, 2),
            }
            for cls in ARCHCAD_COUNTABLE_CLASSES
        },
        "class_geometries": class_geometry_summary,
        "co_occurrence_top5": {
            c1: [f"{c2} ({cnt})" for c2, cnt in co_occurrence[c1].most_common(5)]
            for c1 in list(class_instance_counts.keys())[:10]
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nEDA Report successfully saved to: {out_path}")
    print("\n--- EXECUTIVE SUMMARY ---")
    print(f"Total Slices: {total_images}")
    print(f"Total Object Instances: {total_instances} (Mean: {report['instances_per_image']['mean']}/img)")
    print(f"Empty Slices: {empty_images} ({report['empty_images_pct']}%)")
    print(f"Average Strokes per Slice: {report['strokes_per_image']['mean']} ({report['strokes_per_image']['lines_pct']}% lines, {report['strokes_per_image']['arcs_pct']}% arcs)")
    print(f"Center Collisions (Within-Class Query): Stride 8 = {within_col8_rate:.2f}% -> Stride 4 = {within_col4_rate:.2f}% ({report['collision_audit']['stride4_collision_reduction_pct']}% reduction!)")


if __name__ == "__main__":
    main()
