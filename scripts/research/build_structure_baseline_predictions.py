"""Build deterministic CDG predictions for the structure bake-off."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.drawing_graph import (  # noqa: E402
    add_raster_region_candidates,
    compile_archcad_graph,
    validate_graph,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pack",
        type=Path,
        default=Path("outputs/drawing_graph/room_gold_pack_v0.jsonl"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/ArchCAD"))
    parser.add_argument(
        "--method",
        choices=("endpoint_cycles", "raster_free_space"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or Path(
        f"outputs/drawing_graph/predictions_{args.method}_v0"
    )
    records = [
        json.loads(line)
        for line in args.pack.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    reviewed = [record for record in records if record.get("review_status") == "reviewed"]
    if not reviewed:
        raise ValueError("no reviewed records; deterministic predictions remain gated")
    output_dir.mkdir(parents=True, exist_ok=True)
    emitted = 0
    for record in reviewed:
        image_id = str(record["image_id"])
        annotation_path = args.data_root / "json" / f"{image_id}.json"
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        graph = compile_archcad_graph(
            annotation,
            drawing_id=image_id,
            source_path=str(annotation_path),
            infer_topology_candidates=args.method == "endpoint_cycles",
        )
        if args.method == "raster_free_space":
            add_raster_region_candidates(
                graph,
                annotation,
                wall_width=3,
                door_closure_width=15,
                minimum_area=1_000,
                minimum_hydraulic_radius=20.0,
                minimum_bbox_fill_ratio=0.35,
            )
        validation = validate_graph(graph)
        if not validation.is_valid:
            raise ValueError(
                f"baseline produced invalid graph for {image_id}: {validation.to_dict()}"
            )
        graph.metadata["prediction_method"] = args.method
        graph.write_json(output_dir / f"{image_id}.cdg.json")
        emitted += 1
    summary = {
        "schema_version": "structure-baseline-predictions-v0",
        "method": args.method,
        "reviewed_record_count": len(reviewed),
        "prediction_count": emitted,
        "output_dir": str(output_dir),
    }
    summary_path = output_dir.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
