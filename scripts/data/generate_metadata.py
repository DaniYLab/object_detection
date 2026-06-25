"""
Generate metadata.json for each sample in FloorPlanCAD_dataset.

Reads SVG files from FloorPlanCAD_original, parses per-instance bounding boxes,
saves metadata.json alongside the images in FloorPlanCAD_dataset.

metadata.json format:
{
  "image_size": [1000, 1000],
  "svg_viewbox": [0, 0, 100, 100],
  "instances": [
    {"class": "door_single", "class_id": 10, "instance_id": 5,
     "bbox_px": [x0, y0, x1, y1]},   <- pixel coords in original image
    ...
  ]
}
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
ORIGINAL_ROOT = Path("./data/FloorPlanCAD_original")
DATASET_ROOT  = Path("./data/FloorPlanCAD_dataset")

SPLITS = {
    "train": ["train_set_1", "train_set_2"],
    "test":  ["test_set"],
}

# Semantic ID → class name (must match dataset.py CLASS_NAMES)
SEMANTIC_ID_TO_NAME = {
    1:  "wall",            2:  "door_single",     3:  "door_double",
    4:  "door_sliding",    5:  "window",           6:  "door_revolving",
    7:  "window_bay",      8:  "window_blind",     9:  "stair",
    10: "ramp",            11: "elevator",         12: "escalator",
    13: "column",          14: "toilet",           15: "sink",
    16: "bathtub",         17: "shower",           18: "washing_machine",
    19: "refrigerator",    20: "oven",             21: "bed",
    22: "sofa",            23: "table",            24: "chair",
    25: "room_label",      26: "floor_plan_area",  27: "parking",
    28: "plant",           29: "counter",          30: "cabinet",
    31: "tv",              32: "escalator_stair",  33: "dimension_line",
    34: "symbol_misc",     35: "annotation_text",
}

CLASS_NAMES = sorted(set(SEMANTIC_ID_TO_NAME.values()))
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# ── SVG path coordinate extractor ─────────────────────────────────────────────
COORD_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def path_bbox(d: str) -> tuple[float, float, float, float] | None:
    nums = [float(x) for x in COORD_RE.findall(d)]
    if len(nums) < 2:
        return None
    xs = nums[0::2]
    ys = nums[1::2]
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def merge_bboxes(bboxes: list[tuple]) -> tuple[float, float, float, float]:
    return (
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    )


# ── SVG parser ────────────────────────────────────────────────────────────────

def parse_svg(svg_path: Path) -> dict:
    """
    Parse SVG file → return metadata dict with per-instance bboxes in pixel coords.
    """
    try:
        tree = ET.parse(svg_path)
    except ET.ParseError:
        return {}

    root = tree.getroot()

    # Get viewBox and compute scale
    vb = root.get("viewBox", "0 0 100 100")
    vb_parts = [float(v) for v in vb.split()]
    svg_w = vb_parts[2] if len(vb_parts) >= 4 else 100.0
    svg_h = vb_parts[3] if len(vb_parts) >= 4 else 100.0

    # Get PNG size (assume same stem exists as .png)
    png_path = svg_path.with_suffix(".png")
    if png_path.exists():
        with Image.open(png_path) as img:
            img_w, img_h = img.size
    else:
        img_w, img_h = 1000, 1000  # default

    scale_x = img_w / svg_w
    scale_y = img_h / svg_h

    # Group path bboxes by (semantic_id, instance_id)
    # For stuff (iid=-1): group all paths of same sid together → single bbox
    # For things (iid>=0): group paths of same (sid, iid) → per-instance bbox
    stuff_bboxes:  dict[int, list]        = defaultdict(list)   # sid → [bboxes]
    thing_bboxes:  dict[tuple, list]      = defaultdict(list)   # (sid,iid) → [bboxes]

    for elem in root.iter():
        d = elem.get("d")
        if d is None:
            continue
        sid_str = elem.get("semantic-id")
        if sid_str is None:
            continue
        sid = int(sid_str)
        iid_str = elem.get("instance-id")
        iid = int(iid_str) if iid_str else -1

        bbox = path_bbox(d)
        if bbox is None:
            continue

        if iid == -1:
            stuff_bboxes[sid].append(bbox)
        else:
            thing_bboxes[(sid, iid)].append(bbox)

    instances = []

    # Stuff classes → one bbox per class (union of all paths)
    for sid, bboxes in stuff_bboxes.items():
        if not bboxes:
            continue
        cls_name = SEMANTIC_ID_TO_NAME.get(sid)
        if cls_name is None:
            continue
        x0, y0, x1, y1 = merge_bboxes(bboxes)
        # Scale to pixel coords
        px0 = max(0, int(x0 * scale_x))
        py0 = max(0, int(y0 * scale_y))
        px1 = min(img_w, int(x1 * scale_x))
        py1 = min(img_h, int(y1 * scale_y))
        if px1 > px0 and py1 > py0:
            instances.append({
                "class":      cls_name,
                "class_id":   CLASS_TO_IDX.get(cls_name, -1),
                "instance_id": -1,
                "bbox_px":    [px0, py0, px1, py1],
            })

    # Thing classes → one bbox per instance
    for (sid, iid), bboxes in thing_bboxes.items():
        if not bboxes:
            continue
        cls_name = SEMANTIC_ID_TO_NAME.get(sid)
        if cls_name is None:
            continue
        x0, y0, x1, y1 = merge_bboxes(bboxes)
        px0 = max(0, int(x0 * scale_x))
        py0 = max(0, int(y0 * scale_y))
        px1 = min(img_w, int(x1 * scale_x))
        py1 = min(img_h, int(y1 * scale_y))
        if px1 > px0 and py1 > py0:
            instances.append({
                "class":       cls_name,
                "class_id":    CLASS_TO_IDX.get(cls_name, -1),
                "instance_id": iid,
                "bbox_px":     [px0, py0, px1, py1],
            })

    return {
        "image_size":   [img_w, img_h],
        "svg_viewbox":  [0, 0, svg_w, svg_h],
        "num_instances": len(instances),
        "instances":    instances,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  FloorPlanCAD Metadata Generator")
    print("=" * 60)

    total = 0
    skipped = 0

    for split_name, src_dirs in SPLITS.items():
        out_split = DATASET_ROOT / split_name
        print(f"\n[{split_name.upper()}]")

        for src_dir_name in src_dirs:
            src_dir = ORIGINAL_ROOT / src_dir_name
            if not src_dir.exists():
                print(f"  [SKIP] {src_dir} not found")
                continue

            svg_files = sorted(src_dir.glob("*.svg"))
            print(f"  {src_dir_name}: {len(svg_files)} SVG files")

            for i, svg_path in enumerate(svg_files):
                sample_name = svg_path.stem
                out_dir = out_split / sample_name

                # Skip if sample not in dataset (shouldn't happen)
                if not out_dir.exists():
                    skipped += 1
                    continue

                # Skip if already done
                meta_path = out_dir / "metadata.json"
                if meta_path.exists():
                    total += 1
                    continue

                metadata = parse_svg(svg_path)
                if not metadata:
                    skipped += 1
                    continue

                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, separators=(",", ":"))

                total += 1

                if (i + 1) % 500 == 0:
                    print(f"    [{i+1}/{len(svg_files)}] done — {total} metadata files")

        print(f"  Split done.")

    print(f"\n{'=' * 60}")
    print(f"  Done! {total} metadata files written, {skipped} skipped")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
