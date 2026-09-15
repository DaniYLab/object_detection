"""Create a deterministic room/connectivity annotation pack with candidate hints."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.drawing_graph import (
    classify_archcad_task,
    infer_raster_free_space_candidates,
    validate_room_gold_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/ArchCAD"))
    parser.add_argument(
        "--task-manifest",
        type=Path,
        default=Path("outputs/drawing_graph/archcad_task_manifest.json"),
    )
    parser.add_argument("--subset", default="floor_plan_reading_candidate")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/drawing_graph/room_gold_pack_v0.jsonl"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    with args.task_manifest.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    population = manifest.get("subsets", {}).get(args.subset)
    if not population:
        raise ValueError(f"subset is empty or missing: {args.subset!r}")
    selected = random.Random(args.seed).sample(population, min(args.count, len(population)))
    selected.sort()

    records = []
    for image_id in selected:
        annotation_path = args.data_root / "json" / f"{image_id}.json"
        with annotation_path.open("r", encoding="utf-8") as handle:
            source = json.load(handle)
        tags = classify_archcad_task(source)
        candidates, _ = infer_raster_free_space_candidates(
            source,
            minimum_hydraulic_radius=20,
            minimum_bbox_fill_ratio=0.35,
        )
        record = {
            "schema_version": "room-gold-v0",
            "image_id": image_id,
            "image_path": str(args.data_root / "png" / f"{image_id}.png"),
            "annotation_path": str(annotation_path),
            "selection": {
                "subset": args.subset,
                "task_manifest_fingerprint": manifest["fingerprint"],
                "seed": args.seed,
                "tags": tags,
            },
            "preannotations": {
                "method": "raster_free_space_high_precision_v0",
                "not_ground_truth": True,
                "regions": [
                    {
                        "candidate_id": f"raster:{index:04d}",
                        "bbox": list(candidate.bbox),
                        "area": candidate.area,
                        "hydraulic_radius": round(candidate.hydraulic_radius, 4),
                        "bbox_fill_ratio": round(candidate.bbox_fill_ratio, 4),
                    }
                    for index, candidate in enumerate(candidates)
                ],
            },
            "review_status": "unreviewed",
            "reviewer": None,
            "annotation": {
                "view_type": None,
                "regions": [],
                "relations": [],
                "notes": "",
            },
        }
        errors = validate_room_gold_record(record)
        if errors:
            raise ValueError(f"invalid generated record {image_id}: {errors}")
        records.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "schema_version": "room-gold-pack-summary-v0",
        "record_count": len(records),
        "seed": args.seed,
        "subset": args.subset,
        "task_manifest_fingerprint": manifest["fingerprint"],
        "records_with_preannotations": sum(bool(record["preannotations"]["regions"]) for record in records),
        "annotation_file": str(args.output),
        "not_an_evaluation_split": True,
    }
    summary_path = args.output.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
