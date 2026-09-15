"""Evaluate one or more CDG prediction directories against human-gold CDGs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.drawing_graph import (  # noqa: E402
    CanonicalDrawingGraph,
    evaluate_structure_prediction,
    summarize_structure_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold-dir",
        type=Path,
        default=Path("outputs/drawing_graph/room_gold_graphs_v0"),
    )
    parser.add_argument(
        "--prediction",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="Repeat for deterministic, HEAT, and RoomFormer prediction directories.",
    )
    parser.add_argument("--region-iou-threshold", type=float, default=0.50)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/drawing_graph/structure_bakeoff_v0.json"),
    )
    return parser.parse_args()


def _prediction_specs(values: list[str]) -> dict[str, Path]:
    specs: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"prediction must be NAME=DIR, got {value!r}")
        name, directory = value.split("=", 1)
        if not name or name in specs:
            raise ValueError(f"prediction name is empty or duplicated: {name!r}")
        specs[name] = Path(directory)
    if not specs:
        raise ValueError("at least one --prediction NAME=DIR is required")
    return specs


def main() -> None:
    args = parse_args()
    specs = _prediction_specs(args.prediction)
    gold_paths = sorted(args.gold_dir.glob("*.cdg.json"))
    if not gold_paths:
        raise ValueError(
            f"no human-gold CDGs in {args.gold_dir}; complete reviewed annotations first"
        )
    runs = {}
    for name, prediction_dir in specs.items():
        records = []
        missing = []
        for gold_path in gold_paths:
            prediction_path = prediction_dir / gold_path.name
            if not prediction_path.exists():
                missing.append(gold_path.stem.removesuffix(".cdg"))
                continue
            gold = CanonicalDrawingGraph.read_json(gold_path)
            prediction = CanonicalDrawingGraph.read_json(prediction_path)
            records.append(
                evaluate_structure_prediction(
                    prediction,
                    gold,
                    region_iou_threshold=args.region_iou_threshold,
                )
            )
        runs[name] = {
            "prediction_dir": str(prediction_dir),
            "missing_prediction_count": len(missing),
            "missing_prediction_ids": missing,
            "summary": summarize_structure_metrics(records),
            "records": records,
        }
    report = {
        "schema_version": "structure-bakeoff-v0",
        "gold_dir": str(args.gold_dir),
        "gold_graph_count": len(gold_paths),
        "region_iou_threshold": args.region_iou_threshold,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
