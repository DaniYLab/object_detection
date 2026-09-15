"""Validate a room-gold JSONL pack and report annotation progress."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.drawing_graph.gold_annotations import (  # noqa: E402
    summarize_room_gold_records,
    validate_room_gold_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pack",
        nargs="?",
        type=Path,
        default=Path("outputs/drawing_graph/room_gold_pack_v0.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON report path (default: <pack>.validation.json)",
    )
    return parser.parse_args()


def validate_pack(path: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seen_image_ids: dict[str, int] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append({"line": line_number, "errors": [f"invalid JSON: {exc.msg}"]})
                continue
            if not isinstance(record, dict):
                errors.append({"line": line_number, "errors": ["record must be a mapping"]})
                continue
            record_errors = validate_room_gold_record(record)
            image_id = record.get("image_id")
            if isinstance(image_id, str) and image_id:
                previous_line = seen_image_ids.get(image_id)
                if previous_line is not None:
                    record_errors.append(
                        f"duplicate image_id also present on line {previous_line}: {image_id}"
                    )
                else:
                    seen_image_ids[image_id] = line_number
            if record_errors:
                errors.append(
                    {"line": line_number, "image_id": image_id, "errors": record_errors}
                )
            records.append(record)

    summary = summarize_room_gold_records(records)
    floor_plan_gate = summary["reviewed_floor_plan_count"]
    return {
        "schema_version": "room-gold-validation-v0",
        "pack": str(path),
        **summary,
        "valid": not errors,
        "error_record_count": len(errors),
        "errors": errors,
        "decision_gates": {
            "minimum_records": {
                "target": 200,
                "actual": len(records),
                "passed": len(records) >= 200,
            },
            "minimum_reviewed_floor_plans": {
                "target": 100,
                "actual": floor_plan_gate,
                "passed": floor_plan_gate >= 100,
            },
        },
    }


def main() -> None:
    args = parse_args()
    report = validate_pack(args.pack)
    output = args.output or args.pack.with_suffix(".validation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
