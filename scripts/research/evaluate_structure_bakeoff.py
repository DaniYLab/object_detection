"""Evaluate CDG predictions directly against ArchCAD primitive/instance ground truth."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.drawing_graph import (  # noqa: E402
    CanonicalDrawingGraph,
    evaluate_archcad_prediction,
    load_archcad_graph,
    summarize_archcad_metrics,
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
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--prediction",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="Repeat for each model prediction directory.",
    )
    parser.add_argument("--primitive-tolerance", type=float, default=2.0)
    parser.add_argument("--junction-tolerance", type=float, default=2.0)
    parser.add_argument("--instance-iou-threshold", type=float, default=0.50)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/drawing_graph/structure_bakeoff_v1.json"),
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


def _selected_ids(manifest: dict, subset: str, sample_count: int, seed: int) -> list[str]:
    if manifest.get("purpose") != "task_selection_only_not_an_evaluation_split":
        raise ValueError("task manifest has an unexpected purpose")
    population = manifest.get("subsets", {}).get(subset)
    if not population:
        raise ValueError(f"subset is empty or missing: {subset!r}")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    selected = random.Random(seed).sample(
        list(population), min(sample_count, len(population))
    )
    return sorted(selected)


def main() -> None:
    args = parse_args()
    specs = _prediction_specs(args.prediction)
    manifest = json.loads(args.task_manifest.read_text(encoding="utf-8"))
    image_ids = _selected_ids(manifest, args.subset, args.sample_count, args.seed)
    ground_truth = {
        image_id: load_archcad_graph(args.data_root / "json" / f"{image_id}.json")
        for image_id in image_ids
    }
    # Candidate topology may exist in the compiled graph, but direct metrics
    # consume only raw primitives, instances, junctions, and incident_to edges.
    runs = {}
    for name, prediction_dir in specs.items():
        records = []
        missing = []
        for image_id in image_ids:
            prediction_path = prediction_dir / f"{image_id}.cdg.json"
            if not prediction_path.exists():
                missing.append(image_id)
                continue
            prediction = CanonicalDrawingGraph.read_json(prediction_path)
            records.append(
                evaluate_archcad_prediction(
                    prediction,
                    ground_truth[image_id],
                    primitive_tolerance=args.primitive_tolerance,
                    junction_tolerance=args.junction_tolerance,
                    instance_iou_threshold=args.instance_iou_threshold,
                )
            )
        runs[name] = {
            "prediction_dir": str(prediction_dir),
            "missing_prediction_count": len(missing),
            "missing_prediction_ids": missing,
            "summary": summarize_archcad_metrics(records),
            "records": records,
        }
    report = {
        "schema_version": "structure-bakeoff-v1-archcad-direct-gt",
        "ground_truth": "ArchCAD raw primitive semantic/instance annotations",
        "task_manifest_fingerprint": manifest.get("fingerprint"),
        "task_manifest_is_evaluation_split": False,
        "publication_ready": False,
        "publication_blocker": "source_drawing_id is unavailable",
        "subset": args.subset,
        "sample_count": len(image_ids),
        "seed": args.seed,
        "primitive_tolerance": args.primitive_tolerance,
        "junction_tolerance": args.junction_tolerance,
        "instance_iou_threshold": args.instance_iou_threshold,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
