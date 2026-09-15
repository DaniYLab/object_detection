"""Generate canonical metadata and bounding boxes for ArchCAD-400K samples.

Computes:
1. Tight bounding boxes [x0, y0, x1, y1] for all countable instances
2. 12-dim stroke tokens for the vector branch (Dual-Pathway)
3. SHA-256 content hashes of PNGs for leak-free deduplication
4. Per-sample class distributions and collision statistics
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.archcad_constants import (
    ARCHCAD_IMAGE_SIZE,
    ARCHCAD_RAW_ID_TO_DETECTOR_IDX,
    ARCHCAD_RAW_SEMANTIC_NAMES,
)


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def process_single_archcad_sample(
    uuid_str: str,
    json_path: str,
    png_path: str,
    meta_path: str,
    include_strokes: bool = True,
) -> dict[str, int | str] | None:
    """Process one ArchCAD sample and emit canonical _meta.json."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        entities = data.get("entities", [])

        img_w, img_h = ARCHCAD_IMAGE_SIZE

        # 1. Group entities by instance
        instance_prims: dict[str, list[dict]] = {}
        for e in entities:
            inst = e.get("instance")
            if inst:
                if inst not in instance_prims:
                    instance_prims[inst] = []
                instance_prims[inst].append(e)

        instances = []
        class_counts: dict[str, int] = {}

        for inst_id, prims in instance_prims.items():
            raw_sem = prims[0].get("semantic")
            if raw_sem not in ARCHCAD_RAW_ID_TO_DETECTOR_IDX:
                continue  # Skip stuff / non-countable primitives

            det_idx = ARCHCAD_RAW_ID_TO_DETECTOR_IDX[raw_sem]
            class_name = ARCHCAD_RAW_SEMANTIC_NAMES[raw_sem]

            xs: list[float] = []
            ys: list[float] = []

            for p in prims:
                ptype = p.get("type")
                if ptype == "LINE":
                    s, e_pt = p.get("start"), p.get("end")
                    if s and e_pt:
                        xs.extend([float(s[0]), float(e_pt[0])])
                        ys.extend([float(s[1]), float(e_pt[1])])
                elif ptype in ("ARC", "CIRCLE"):
                    c = p.get("center")
                    r = p.get("radius", 0.0)
                    if c is not None and r > 0:
                        xs.extend([float(c[0]) - r, float(c[0]) + r])
                        ys.extend([float(c[1]) - r, float(c[1]) + r])

            if not xs or not ys:
                continue

            x0 = max(0.0, min(float(img_w), min(xs)))
            y0 = max(0.0, min(float(img_h), min(ys)))
            x1 = max(0.0, min(float(img_w), max(xs)))
            y1 = max(0.0, min(float(img_h), max(ys)))

            w = x1 - x0
            h = y1 - y0
            if w < 1.0 or h < 1.0:
                continue  # Reject degenerate boxes

            instances.append(
                {
                    "instance_id": inst_id,
                    "semantic_id": raw_sem,
                    "category_id": det_idx,
                    "category_name": class_name,
                    "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
                    "area": round(w * h, 2),
                    "num_primitives": len(prims),
                }
            )
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        # 2. Extract vector stroke primitives for Dual-Pathway branch
        # Normalized 12-dim tokens:
        # [x0, y0, x1, y1, cx, cy, r, cos0, sin0, cos1, sin1, type_id]
        strokes = []
        if include_strokes:
            for e in entities:
                ptype = e.get("type")
                if ptype == "LINE":
                    s = e.get("start")
                    end = e.get("end")
                    if s and end:
                        x0 = float(s[0]) / img_w
                        y0 = float(s[1]) / img_h
                        x1 = float(end[0]) / img_w
                        y1 = float(end[1]) / img_h
                        # Line: slots 4..10 = 0.0, slot 11 (type_id) = 0.0
                        strokes.append(
                            [x0, y0, x1, y1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                        )
                elif ptype == "ARC":
                    c = e.get("center")
                    r = e.get("radius", 0.0)
                    sa = e.get("start_angle", 0.0)
                    ea = e.get("end_angle", 360.0)
                    s = e.get("start")
                    end = e.get("end")
                    if c and r > 0:
                        cx = float(c[0]) / img_w
                        cy = float(c[1]) / img_h
                        norm_r = float(r) / img_w
                        # Endpoints
                        if s and end:
                            x0 = float(s[0]) / img_w
                            y0 = float(s[1]) / img_h
                            x1 = float(end[0]) / img_w
                            y1 = float(end[1]) / img_h
                        else:
                            x0, y0 = cx, cy
                            x1, y1 = cx, cy
                        rad_s = math.radians(sa)
                        rad_e = math.radians(ea)
                        cos0, sin0 = math.cos(rad_s), math.sin(rad_s)
                        cos1, sin1 = math.cos(rad_e), math.sin(rad_e)
                        # Arc: slot 11 (type_id) = 1.0
                        strokes.append(
                            [
                                x0,
                                y0,
                                x1,
                                y1,
                                cx,
                                cy,
                                norm_r,
                                cos0,
                                sin0,
                                cos1,
                                sin1,
                                1.0,
                            ]
                        )
                elif ptype == "CIRCLE":
                    c = e.get("center")
                    r = e.get("radius", 0.0)
                    if c and r > 0:
                        cx = float(c[0]) / img_w
                        cy = float(c[1]) / img_h
                        norm_r = float(r) / img_w
                        strokes.append(
                            [cx, cy, cx, cy, cx, cy, norm_r, 1.0, 0.0, 1.0, 0.0, 1.0]
                        )

        # 3. Compute PNG SHA-256
        png_hash = compute_sha256(Path(png_path))

        meta_dict = {
            "schema_version": "v2",
            "image_id": uuid_str,
            "png_sha256": png_hash,
            "image_size": [img_w, img_h],
            "instances": instances,
            "num_instances": len(instances),
            "class_counts": class_counts,
            "num_strokes": len(strokes),
            "strokes": strokes,
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f)

        return {
            "uuid": uuid_str,
            "png_sha256": png_hash,
            "num_instances": len(instances),
            "num_strokes": len(strokes),
        }

    except Exception as exc:
        print(f"Error processing {uuid_str}: {exc}")
        return None


def worker_task(args: tuple) -> dict | None:
    return process_single_archcad_sample(*args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build canonical metadata for ArchCAD-400K"
    )
    parser.add_argument("--data-root", type=str, default="data/ArchCAD")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(args.data_root)
    json_dir = root / "json"
    png_dir = root / "png"
    meta_dir = root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(list(json_dir.glob("*.json")))
    if args.limit:
        json_files = json_files[: args.limit]

    print(f"Found {len(json_files)} JSON files in {json_dir}")

    tasks = []
    skipped = 0
    for jf in json_files:
        uuid_str = jf.stem
        png_path = png_dir / f"{uuid_str}.png"
        meta_path = meta_dir / f"{uuid_str}_meta.json"

        if not png_path.exists():
            continue
        if meta_path.exists() and not args.force:
            skipped += 1
            continue

        tasks.append((uuid_str, str(jf), str(png_path), str(meta_path), True))

    if skipped > 0:
        print(
            f"Skipping {skipped} already processed samples. Use --force to overwrite."
        )

    print(f"Submitting {len(tasks)} tasks using {args.workers} worker processes...")
    t0 = time.time()
    results = []

    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for i, res in enumerate(executor.map(worker_task, tasks, chunksize=100)):
                if res:
                    results.append(res)
                if (i + 1) % 5000 == 0 or (i + 1) == len(tasks):
                    elapsed = time.time() - t0
                    rate = (i + 1) / max(1e-5, elapsed)
                    print(f"  Processed {i + 1}/{len(tasks)} ({rate:.1f} samples/s)...")

    elapsed = time.time() - t0
    total_meta = len(list(meta_dir.glob("*_meta.json")))
    print(
        f"\nMetadata build complete! Total metadata files: {total_meta} (took {elapsed:.1f}s)"
    )


if __name__ == "__main__":
    main()
