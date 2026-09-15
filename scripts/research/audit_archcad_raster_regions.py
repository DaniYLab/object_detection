"""Audit raster-assisted free-space candidates on an ArchCAD task subset."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.drawing_graph import (
    add_raster_region_candidates,
    build_wall_barrier_mask,
    compile_archcad_graph,
    infer_raster_free_space_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/ArchCAD"))
    parser.add_argument(
        "--task-manifest",
        type=Path,
        default=Path("outputs/drawing_graph/archcad_task_manifest.json"),
    )
    parser.add_argument("--subset", default="architectural_reading_candidate")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--wall-width", type=int, default=3)
    parser.add_argument("--door-closure-width", type=int, default=15)
    parser.add_argument("--minimum-area", type=int, default=1000)
    parser.add_argument("--minimum-hydraulic-radius", type=float, default=12.0)
    parser.add_argument("--minimum-bbox-fill-ratio", type=float, default=0.35)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/drawing_graph/raster_regions_architectural_100_report.json"),
    )
    parser.add_argument("--sample-overlays", type=int, default=5)
    return parser.parse_args()


def _render_overlay(
    image_path: Path,
    annotation: dict,
    candidates,
    output_path: Path,
    *,
    wall_width: int,
    door_closure_width: int,
) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGBA")
    barrier, _ = build_wall_barrier_mask(
        annotation,
        image_size=image.size,
        wall_width=wall_width,
        door_closure_width=door_closure_width,
    )
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    rows, columns = barrier.nonzero()
    for row, column in zip(rows, columns):
        draw.point((int(column), int(row)), fill=(0, 145, 255, 180))
    colors = [(255, 70, 30, 70), (255, 180, 20, 70), (150, 50, 230, 70)]
    for index, candidate in enumerate(candidates):
        color = colors[index % len(colors)]
        for row, start, end in candidate.spans:
            draw.line([(start, row), (end - 1, row)], fill=color, width=1)
        draw.rectangle(candidate.bbox, outline=(*color[:3], 240), width=3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(image, overlay).convert("RGB").save(output_path)


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    with args.task_manifest.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("purpose") != "task_selection_only_not_an_evaluation_split":
        raise ValueError("unexpected task manifest purpose")
    if args.subset not in manifest.get("subsets", {}):
        raise ValueError(f"unknown subset: {args.subset!r}")

    population = list(manifest["subsets"][args.subset])
    selected = random.Random(args.seed).sample(population, min(args.limit, len(population)))
    selected.sort()
    totals: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    sample_records: list[dict] = []
    overlays_written = 0
    for image_id in selected:
        annotation_path = args.data_root / "json" / f"{image_id}.json"
        with annotation_path.open("r", encoding="utf-8") as handle:
            annotation = json.load(handle)
        without_closure, _ = infer_raster_free_space_candidates(
            annotation,
            wall_width=args.wall_width,
            close_doors=False,
            minimum_area=args.minimum_area,
            minimum_hydraulic_radius=args.minimum_hydraulic_radius,
            minimum_bbox_fill_ratio=args.minimum_bbox_fill_ratio,
        )
        candidates, diagnostics = infer_raster_free_space_candidates(
            annotation,
            wall_width=args.wall_width,
            close_doors=True,
            door_closure_width=args.door_closure_width,
            minimum_area=args.minimum_area,
            minimum_hydraulic_radius=args.minimum_hydraulic_radius,
            minimum_bbox_fill_ratio=args.minimum_bbox_fill_ratio,
        )
        totals["candidates_without_closure"] += len(without_closure)
        totals["candidates_with_closure"] += len(candidates)
        totals["samples_with_candidates_without_closure"] += int(bool(without_closure))
        totals["samples_with_candidates_with_closure"] += int(bool(candidates))
        totals["door_closures"] += int(diagnostics["door_closure_count"])
        rejected.update(diagnostics["rejected"])
        graph = compile_archcad_graph(
            annotation,
            drawing_id=image_id,
            infer_topology_candidates=False,
        )
        typed = add_raster_region_candidates(
            graph,
            annotation,
            candidates=candidates,
            diagnostics=diagnostics,
        )
        categories.update(typed["category_counts"])
        sample_records.append(
            {
                "image_id": image_id,
                "without_closure": len(without_closure),
                "with_closure": len(candidates),
                "door_closures": diagnostics["door_closure_count"],
            }
        )
        if candidates and overlays_written < args.sample_overlays:
            _render_overlay(
                args.data_root / "png" / f"{image_id}.png",
                annotation,
                candidates,
                args.output.parent / "raster_overlays" / f"{image_id}.png",
                wall_width=args.wall_width,
                door_closure_width=args.door_closure_width,
            )
            overlays_written += 1

    count = len(selected)
    report = {
        "schema_version": "archcad-raster-region-audit-v0",
        "selection": args.subset,
        "task_manifest_fingerprint": manifest["fingerprint"],
        "seed": args.seed,
        "sample_count": count,
        "parameters": {
            "wall_width": args.wall_width,
            "door_closure_width": args.door_closure_width,
            "minimum_area": args.minimum_area,
            "minimum_hydraulic_radius": args.minimum_hydraulic_radius,
            "minimum_bbox_fill_ratio": args.minimum_bbox_fill_ratio,
        },
        "coverage": {
            "without_closure_count": totals["samples_with_candidates_without_closure"],
            "without_closure_pct": round(100 * totals["samples_with_candidates_without_closure"] / count, 2),
            "with_closure_count": totals["samples_with_candidates_with_closure"],
            "with_closure_pct": round(100 * totals["samples_with_candidates_with_closure"] / count, 2),
        },
        "averages": {
            "candidates_without_closure": round(totals["candidates_without_closure"] / count, 2),
            "candidates_with_closure": round(totals["candidates_with_closure"] / count, 2),
            "door_closures": round(totals["door_closures"] / count, 2),
        },
        "rejected_components": dict(sorted(rejected.items())),
        "candidate_categories": dict(sorted(categories.items())),
        "semantic_precision_audited": False,
        "room_ground_truth_ready": False,
        "sample_records": sample_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: report[key] for key in ("coverage", "averages", "rejected_components")}, indent=2))


if __name__ == "__main__":
    main()
