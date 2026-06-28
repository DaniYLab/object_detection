"""
Build processed FloorPlanCAD dataset.

Input  : data/FloorPlanCAD_original/{train_set_1, train_set_2, test_set}/
Output : data/FloorPlanCAD_dataset/{train, test}/
         Each sample -> one subfolder containing:
           - original.png    : full original image
           - metadata.json   : per-instance bounding boxes + class info

Usage:
    python scripts/data/build_dataset.py
    python scripts/data/build_dataset.py --original_root /path/to/raw --output_root /path/to/out
    ORIGINAL_ROOT=/content/raw OUTPUT_ROOT=/content/out python scripts/data/build_dataset.py

Annotation format: SVG paths with semantic-id + instance-id attributes.
"""

import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from PIL import Image

# ─── Shared class mappings (single source of truth) ───────────────────────────
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
    - Save original.png
    - Parse SVG annotations → metadata.json with per-instance bboxes

    Returns number of instances found.
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

    # Collect metadata (no crops — dataset.py only uses original.png + metadata)
    instances_meta = []

    for (sid, iid), bboxes in instance_bboxes.items():
        # Merge all path bboxes for this instance
        x_min = min(b[0] for b in bboxes)
        y_min = min(b[1] for b in bboxes)
        x_max = max(b[2] for b in bboxes)
        y_max = max(b[3] for b in bboxes)

        # Convert to pixels
        px0, py0, px1, py1 = svg_to_pixel((x_min, y_min, x_max, y_max), scale_x, scale_y)
        px0 = max(0, px0)
        py0 = max(0, py0)
        px1 = min(img_w, px1)
        py1 = min(img_h, py1)

        # Skip tiny instances
        if (px1 - px0) < min_size or (py1 - py0) < min_size:
            continue

        class_name = get_class_name(sid)

        instances_meta.append({
            "class":       class_name,
            "class_id":    CLASS_TO_IDX.get(class_name, -1),
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

    return len(instances_meta)



# ─── Dataset builder ───────────────────────────────────────────────────────────

SPLITS = {
    "train": ["train_set_1", "train_set_2"],
    "test":  ["test_set"],
}


def build_dataset(
    original_root: Path,
    output_root: Path,
    min_size: int = 8,
) -> None:
    print("=" * 60)
    print("  FloorPlanCAD Dataset Builder")
    print(f"  Input    : {original_root.resolve()}")
    print(f"  Output   : {output_root.resolve()}")
    print(f"  min_size : {min_size}px")
    print("=" * 60)

    total_samples = 0
    total_instances = 0

    for split_name, source_dirs in SPLITS.items():
        split_out = output_root / split_name
        split_out.mkdir(parents=True, exist_ok=True)
        print(f"\n[{split_name.upper()}] Processing {source_dirs}...")

        split_samples = 0
        split_instances = 0

        for src_dir_name in source_dirs:
            src_dir = original_root / src_dir_name
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

                n_inst = process_sample(png_path, svg_path, out_dir, min_size=min_size)
                split_samples += 1
                split_instances += n_inst

                if (i + 1) % 200 == 0:
                    print(f"    [{i+1}/{len(png_files)}] processed — "
                          f"{split_instances} instances so far")

        print(f"  => {split_samples} samples, {split_instances} instances")
        total_samples += split_samples
        total_instances += split_instances

    print(f"\n{'=' * 60}")
    print(f"  DONE!")
    print(f"  Total samples   : {total_samples}")
    print(f"  Total instances : {total_instances}")
    print(f"  Output          : {output_root.resolve()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Build FloorPlanCAD processed dataset")
    parser.add_argument(
        "--original_root",
        default=os.environ.get("ORIGINAL_ROOT", "./data/FloorPlanCAD_original"),
        help="Path to raw FloorPlanCAD (contains train_set_1/, etc.)",
    )
    parser.add_argument(
        "--output_root",
        default=os.environ.get("OUTPUT_ROOT", "./data/FloorPlanCAD_dataset"),
        help="Output path for processed dataset",
    )
    parser.add_argument(
        "--min_size", type=int, default=8,
        help="Skip bboxes smaller than this (pixels)",
    )
    args = parser.parse_args()

    build_dataset(
        original_root=Path(args.original_root),
        output_root=Path(args.output_root),
        min_size=args.min_size,
    )

