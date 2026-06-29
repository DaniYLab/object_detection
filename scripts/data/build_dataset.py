"""
Build FloorPlanCAD metadata — parse SVG annotations, write {stem}_meta.json
alongside each original PNG. No image copying.

Input : FloorPlanCAD_original/{train_set_1, train_set_2, test_set}/
          Each folder: xxx.png + xxx.svg
Output: Same folders, new file xxx_meta.json per sample

Usage:
    python scripts/data/build_dataset.py
    python scripts/data/build_dataset.py --data_root /content/FloorPlanCAD_orig
    DATA_ROOT=/content/FloorPlanCAD_orig python scripts/data/build_dataset.py

Annotation format: SVG paths with semantic-id + instance-id attributes.
"""

import json
import xml.etree.ElementTree as ET
import os
from collections import defaultdict
from pathlib import Path

from PIL import Image

# ─── Shared class mappings ─────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.constants import SEMANTIC_ID_TO_NAME, CLASS_TO_IDX


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
    min_size: int = 8,
) -> int:
    """
    Parse one SVG file → write {stem}_meta.json alongside the PNG.
    No image copying — PNG stays where it is.

    Returns number of instances found.
    """
    meta_path = png_path.with_name(png_path.stem + "_meta.json")

    # Skip if already done
    if meta_path.exists():
        return -1  # -1 = skipped

    # Get image dimensions (open minimally)
    try:
        with Image.open(png_path) as img:
            img_w, img_h = img.size
    except Exception:
        return 0

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
        bbox = parse_path_bbox(d)
        if bbox is None:
            continue

        key = (sid, iid) if iid != -1 else (sid, id(elem))
        instance_bboxes[key].append(bbox)

    if not instance_bboxes:
        return 0

    # Build instances list
    instances_meta = []
    for (sid, iid), bboxes in instance_bboxes.items():
        x_min = min(b[0] for b in bboxes)
        y_min = min(b[1] for b in bboxes)
        x_max = max(b[2] for b in bboxes)
        y_max = max(b[3] for b in bboxes)

        px0, py0, px1, py1 = svg_to_pixel((x_min, y_min, x_max, y_max), scale_x, scale_y)
        px0 = max(0, px0)
        py0 = max(0, py0)
        px1 = min(img_w, px1)
        py1 = min(img_h, py1)

        if (px1 - px0) < min_size or (py1 - py0) < min_size:
            continue

        class_name = get_class_name(sid)
        instances_meta.append({
            "class":       class_name,
            "class_id":    CLASS_TO_IDX.get(class_name, -1),
            "instance_id": iid if isinstance(iid, int) else -1,
            "bbox_px":     [px0, py0, px1, py1],
        })

    # Write metadata alongside the PNG
    metadata = {
        "image_size":    [img_w, img_h],
        "svg_viewbox":   [0, 0, svg_w, svg_h],
        "num_instances": len(instances_meta),
        "instances":     instances_meta,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, separators=(",", ":"))

    return len(instances_meta)


# ─── Dataset builder ───────────────────────────────────────────────────────────

SPLITS = {
    "train": ["train_set_1", "train_set_2"],
    "test":  ["test_set"],
}


def build_dataset(
    data_root: Path,
    min_size: int = 8,
) -> None:
    print("=" * 60)
    print("  FloorPlanCAD Metadata Builder")
    print(f"  Data root : {data_root.resolve()}")
    print(f"  min_size  : {min_size}px")
    print("  (No image copying — writes *_meta.json alongside PNGs)")
    print("=" * 60)

    total_samples = 0
    total_skipped = 0
    total_instances = 0

    for split_name, source_dirs in SPLITS.items():
        print(f"\n[{split_name.upper()}]")

        for src_dir_name in source_dirs:
            src_dir = data_root / src_dir_name
            if not src_dir.exists():
                print(f"  [SKIP] {src_dir} not found")
                continue

            png_files = sorted([
                p for p in src_dir.glob("*.png")
                if not p.parent.name == "coco_vis"
            ])
            print(f"  {src_dir_name}: {len(png_files)} samples")

            split_instances = 0
            split_skipped = 0

            for i, png_path in enumerate(png_files):
                svg_path = png_path.with_suffix(".svg")
                if not svg_path.exists():
                    continue

                n = process_sample(png_path, svg_path, min_size=min_size)
                if n == -1:
                    split_skipped += 1
                else:
                    split_instances += n
                    total_samples += 1

                if (i + 1) % 500 == 0:
                    print(f"    [{i+1}/{len(png_files)}] — {split_instances} instances so far")

            total_instances += split_instances
            total_skipped += split_skipped
            print(f"  => {total_samples} processed, {split_skipped} skipped (already done)")

    print(f"\n{'=' * 60}")
    print(f"  DONE!")
    print(f"  Total processed : {total_samples}")
    print(f"  Total skipped   : {total_skipped} (already had _meta.json)")
    print(f"  Total instances : {total_instances}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate *_meta.json metadata for FloorPlanCAD (no image copying)"
    )
    parser.add_argument(
        "--data_root",
        default=os.environ.get("DATA_ROOT", "./data/FloorPlanCAD_original"),
        help="Path to FloorPlanCAD_original/ (contains train_set_1/, etc.)",
    )
    parser.add_argument(
        "--min_size", type=int, default=8,
        help="Skip bboxes smaller than this (pixels)",
    )
    args = parser.parse_args()

    build_dataset(
        data_root=Path(args.data_root),
        min_size=args.min_size,
    )
