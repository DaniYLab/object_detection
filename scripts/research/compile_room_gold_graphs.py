"""Compile reviewed room-gold records into evidence-ready CDG JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.drawing_graph import (  # noqa: E402
    add_room_gold_annotations,
    load_archcad_graph,
    validate_room_gold_record,
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
        "--output-dir",
        type=Path,
        default=Path("outputs/drawing_graph/room_gold_graphs_v0"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = [
        json.loads(line)
        for line in args.pack.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    status_counts: Counter[str] = Counter()
    compiled_count = 0
    compiled_floor_plan_count = 0
    region_count = 0
    relation_count = 0
    errors: list[dict[str, object]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        status_counts[str(record.get("review_status"))] += 1
        if record.get("review_status") != "reviewed":
            continue
        validation_errors = validate_room_gold_record(record)
        if validation_errors:
            errors.append(
                {"image_id": record.get("image_id"), "errors": validation_errors}
            )
            continue
        image_id = str(record["image_id"])
        try:
            graph = load_archcad_graph(args.data_root / "json" / f"{image_id}.json")
            counts = add_room_gold_annotations(graph, record)
            graph.write_json(args.output_dir / f"{image_id}.cdg.json")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({"image_id": image_id, "errors": [str(exc)]})
            continue
        compiled_count += 1
        compiled_floor_plan_count += int(
            record["annotation"].get("view_type") == "floor_plan"
        )
        region_count += counts["region_count"]
        relation_count += counts["relation_count"]

    summary = {
        "schema_version": "room-gold-graph-compile-summary-v0",
        "pack": str(args.pack),
        "record_count": len(records),
        "review_status_counts": dict(sorted(status_counts.items())),
        "compiled_count": compiled_count,
        "compiled_floor_plan_count": compiled_floor_plan_count,
        "region_count": region_count,
        "relation_count": relation_count,
        "error_count": len(errors),
        "errors": errors,
        "output_dir": str(args.output_dir),
    }
    summary_path = args.output_dir.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
