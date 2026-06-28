"""
Build processed FloorPlanCAD dataset.

Input  : data/FloorPlanCAD_original/{train_set_1, train_set_2, test_set}/
Output : data/FloorPlanCAD_dataset/{train, test}/
         Each sample -> one subfolder containing:
           - original.png           : full original image
           - {class_name}_{n}.png  : cropped patch for each annotated instance

Annotation format: SVG paths with semantic-id + instance-id attributes.
SVG viewBox: 0 0 100 100  |  PNG size: 1000x1000  => scale = 10x
"""

import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from PIL import Image

# ─── Class ID → Name mapping (FloorPlanCAD 30-class panoptic) ─────────────────
# Based on the official paper: "FloorPlanCAD: A Large-Scale CAD Floor Plan Dataset"
# Stuff classes (semantic regions): wall, parking
# Thing classes (countable instances): doors, windows, furniture, etc.
SEMANTIC_ID_TO_NAME = {
    1:  "wall",
    2:  "door_single",
    3:  "door_double",
    4:  "door_sliding",
    5:  "window",
    6:  "door_revolving",
    7:  "window_bay",
    8:  "window_blind",
    9:  "stair",
    10: "ramp",
    11: "elevator",
    12: "escalator",
    13: "column",
    14: "toilet",
    15: "sink",
    16: "bathtub",
    17: "shower",
    18: "washing_machine",
    19: "refrigerator",
    20: "oven",
    21: "bed",
    22: "sofa",
    23: "table",
    24: "chair",
    25: "room_label",
    26: "floor_plan_area",
    27: "parking",
    28: "plant",
    29: "counter",
    30: "cabinet",
    31: "tv",
    32: "escalator_stair",
    33: "dimension_line",
    34: "symbol_misc",
    35: "annotation_text",
}


def get_class_name(semantic_id: int) -> str:
    return SEMANTIC_ID_TO_NAME.get(semantic_id, f"class_{semantic_id:02d}")


# ─── SVG coordinate parser ─────────────────────────────────────────────────────

from svgpathtools import parse_path

def parse_path_bbox(d: str) -> tuple[float, float, float, float] | None:
    try:
        path = parse_path(d)
        if not path:
            return None
        xmin, xmax, ymin, ymax = path.bbox()
        return xmin, ymin, xmax, ymax
    except Exception:
        return None


def svg_to_pixel(bbox: tuple, scale_x: float, scale_y: float) -> tuple[int, int, int, int]:
    """Convert SVG coordinates to pixel coordinates."""
    x_min, y_min, x_max, y_max = bbox
    return (
        int(x_min * scale_x),
        int(y_min * scale_y),
        int(x_max * scale_x),
        int(y_max * scale_y),
    )


# ─── Core processing ───────────────────────────────────────────────────────────

def process_sample(
    png_path: Path,
    svg_path: Path,
    out_dir: Path,
    padding: int = 5,
    min_size: int = 8,
) -> int:
    """
    Process one (PNG, SVG) pair:
    - Copy original.png
    - Crop each annotated instance and save as {class_name}_{n}.png

    Returns number of crops saved.
    """
    # Load image
    img = Image.open(png_path).convert("RGB")
    img_w, img_h = img.size

    # Parse SVG
    try:
        tree = ET.parse(svg_path)
    except ET.ParseError:
        return 0
    root = tree.getroot()

    # Get SVG viewBox dimensions for scaling
    vb = root.get("viewBox", "0 0 100 100")
    vb_parts = [float(v) for v in vb.split()]
    svg_w = vb_parts[2] if len(vb_parts) >= 4 else 100.0
    svg_h = vb_parts[3] if len(vb_parts) >= 4 else 100.0
    scale_x = img_w / svg_w
    scale_y = img_h / svg_h

    # Group paths by (semantic_id, instance_id) → aggregate bounding boxes
    # instance-id == -1 means "stuff" (wall, etc.) — treat each path as own instance
    instance_bboxes: dict[tuple, list] = defaultdict(list)

    ns = {"svg": "http://www.w3.org/2000/svg"}
    for elem in root.iter():
        d = elem.get("d")
        if d is None:
            continue
        sid_str = elem.get("semantic-id")
        iid_str = elem.get("instance-id")
        if sid_str is None:
            continue

        sid = int(sid_str)
        iid = int(iid_str) if iid_str else -1
        bbox = parse_path_bbox(d)
        if bbox is None:
            continue

        # Key: use (sid, iid) — for stuff (iid==-1) every path is unique
        key = (sid, iid) if iid != -1 else (sid, id(elem))
        instance_bboxes[key].append(bbox)

    # No annotations found
    if not instance_bboxes:
        return 0

    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save original image
    img.save(out_dir / "original.png")

    # Crop each instance + collect metadata
    crops_saved = 0
    class_counters: dict[int, int] = defaultdict(int)
    instances_meta = []

    # Class name → class index (alphabetical, matches dataset.py CLASS_NAMES)
    all_class_names = sorted(set(SEMANTIC_ID_TO_NAME.values()))
    cls_to_idx = {name: i for i, name in enumerate(all_class_names)}

    for (sid, iid), bboxes in instance_bboxes.items():
        # Merge all path bboxes for this instance
        x_min = min(b[0] for b in bboxes)
        y_min = min(b[1] for b in bboxes)
        x_max = max(b[2] for b in bboxes)
        y_max = max(b[3] for b in bboxes)

        # Convert to pixels with padding
        px0, py0, px1, py1 = svg_to_pixel((x_min, y_min, x_max, y_max), scale_x, scale_y)
        px0 = max(0, px0 - padding)
        py0 = max(0, py0 - padding)
        px1 = min(img_w, px1 + padding)
        py1 = min(img_h, py1 + padding)

        # Skip tiny crops
        if (px1 - px0) < min_size or (py1 - py0) < min_size:
            continue

        crop = img.crop((px0, py0, px1, py1))
        class_name = get_class_name(sid)
        class_counters[sid] += 1
        n = class_counters[sid]
        crop.save(out_dir / f"{class_name}_{n:03d}.png")
        crops_saved += 1

        # Record bbox metadata — iid as original instance id (-1 for stuff)
        instances_meta.append({
            "class":       class_name,
            "class_id":    cls_to_idx.get(class_name, -1),
            "instance_id": iid if isinstance(iid, int) else -1,
            "bbox_px":     [px0, py0, px1, py1],
        })

    # Write metadata.json with all bbox coordinates
    metadata = {
        "image_size":    [img_w, img_h],
        "svg_viewbox":   [0, 0, svg_w, svg_h],
        "num_instances": len(instances_meta),
        "instances":     instances_meta,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, separators=(",", ":"))

    return crops_saved



# ─── Dataset builder ───────────────────────────────────────────────────────────

SPLITS = {
    "train": ["train_set_1", "train_set_2"],
    "test":  ["test_set"],
}

# Cho phép override qua env var (dùng khi build trên Colab để tránh ghi chậm lên Drive)
# Ví dụ: OUTPUT_ROOT=/content/FloorPlanCAD_dataset python build_dataset.py
import os as _os
ORIGINAL_ROOT = Path(_os.environ.get("ORIGINAL_ROOT", "./data/FloorPlanCAD_original"))
OUTPUT_ROOT   = Path(_os.environ.get("OUTPUT_ROOT",   "./data/FloorPlanCAD_dataset"))


def build_dataset() -> None:
    print("=" * 60)
    print("  FloorPlanCAD Dataset Builder")
    print(f"  Input  : {ORIGINAL_ROOT.resolve()}")
    print(f"  Output : {OUTPUT_ROOT.resolve()}")
    print("=" * 60)

    total_samples = 0
    total_crops = 0

    for split_name, source_dirs in SPLITS.items():
        split_out = OUTPUT_ROOT / split_name
        split_out.mkdir(parents=True, exist_ok=True)
        print(f"\n[{split_name.upper()}] Processing {source_dirs}...")

        split_samples = 0
        split_crops = 0

        for src_dir_name in source_dirs:
            src_dir = ORIGINAL_ROOT / src_dir_name
            if not src_dir.exists():
                print(f"  [SKIP] {src_dir} not found")
                continue

            # Find all PNG files (skip coco_vis subfolder)
            png_files = sorted([
                p for p in src_dir.glob("*.png")
                if p.stem and not p.parent.name == "coco_vis"
            ])

            print(f"  {src_dir_name}: {len(png_files)} samples")

            for i, png_path in enumerate(png_files):
                svg_path = png_path.with_suffix(".svg")
                if not svg_path.exists():
                    continue

                sample_name = png_path.stem
                out_dir = split_out / sample_name

                crops = process_sample(png_path, svg_path, out_dir)
                split_samples += 1
                split_crops += crops

                if (i + 1) % 200 == 0:
                    print(f"    [{i+1}/{len(png_files)}] processed — "
                          f"{split_crops} crops so far")

        print(f"  => {split_samples} samples, {split_crops} crops")
        total_samples += split_samples
        total_crops += split_crops

    print(f"\n{'=' * 60}")
    print(f"  DONE!")
    print(f"  Total samples : {total_samples}")
    print(f"  Total crops   : {total_crops}")
    print(f"  Output        : {OUTPUT_ROOT.resolve()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    build_dataset()
