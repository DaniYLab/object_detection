"""Measure independent room/connectivity review agreement between two packs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.drawing_graph import summarize_room_gold_agreement  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-pack",
        type=Path,
        default=Path("outputs/drawing_graph/room_gold_pack_v0.jsonl"),
    )
    parser.add_argument(
        "--secondary-pack",
        type=Path,
        default=Path("outputs/drawing_graph/room_gold_pack_v0_double.jsonl"),
    )
    parser.add_argument("--polygon-iou-threshold", type=float, default=0.90)
    parser.add_argument("--relation-f1-threshold", type=float, default=0.95)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/drawing_graph/room_gold_agreement_v0.json"),
    )
    return parser.parse_args()


def _load(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    args = parse_args()
    report = summarize_room_gold_agreement(
        _load(args.primary_pack),
        _load(args.secondary_pack),
        polygon_iou_threshold=args.polygon_iou_threshold,
        relation_f1_threshold=args.relation_f1_threshold,
    )
    report.update(
        {
            "primary_pack": str(args.primary_pack),
            "secondary_pack": str(args.secondary_pack),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
