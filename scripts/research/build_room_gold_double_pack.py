"""Build a blinded, deterministic second-review subset from a room-gold pack."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.drawing_graph import validate_room_gold_record  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-pack",
        type=Path,
        default=Path("outputs/drawing_graph/room_gold_pack_v0.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/drawing_graph/room_gold_pack_v0_double.jsonl"),
    )
    parser.add_argument("--fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for index, record in enumerate(records):
        errors = validate_room_gold_record(record)
        if errors:
            raise ValueError(f"invalid source record {index}: {errors}")
    return records


def build_double_review_records(
    records: list[dict[str, Any]], *, fraction: float, seed: int
) -> list[dict[str, Any]]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    target = max(1, round(len(records) * fraction)) if records else 0
    selected_indices = sorted(
        random.Random(seed).sample(range(len(records)), target)
    )
    blinded: list[dict[str, Any]] = []
    for index in selected_indices:
        source = records[index]
        record = {
            key: copy.deepcopy(source[key])
            for key in (
                "schema_version",
                "image_id",
                "image_path",
                "annotation_path",
                "selection",
                "preannotations",
            )
            if key in source
        }
        record["double_annotation"] = {
            "role": "independent_secondary_review",
            "seed": seed,
            "fraction": fraction,
            "source_record_index": index,
            "primary_annotation_blinded": True,
        }
        record["review_status"] = "unreviewed"
        record["reviewer"] = None
        record["annotation"] = {
            "view_type": None,
            "regions": [],
            "relations": [],
            "notes": "",
        }
        errors = validate_room_gold_record(record)
        if errors:
            raise ValueError(f"generated invalid double-review record: {errors}")
        blinded.append(record)
    return blinded


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {args.output}; pass --force")
    records = load_records(args.primary_pack)
    blinded = build_double_review_records(
        records, fraction=args.fraction, seed=args.seed
    )
    serialized = [json.dumps(record, ensure_ascii=False) for record in blinded]
    fingerprint = hashlib.sha256("\n".join(serialized).encode("utf-8")).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(serialized) + "\n", encoding="utf-8")
    summary = {
        "schema_version": "room-gold-double-pack-summary-v0",
        "primary_pack": str(args.primary_pack),
        "secondary_pack": str(args.output),
        "primary_record_count": len(records),
        "secondary_record_count": len(blinded),
        "fraction": args.fraction,
        "seed": args.seed,
        "fingerprint": fingerprint,
        "primary_annotation_blinded": True,
        "selected_image_ids": [record["image_id"] for record in blinded],
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
