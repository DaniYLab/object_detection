"""Build direct-ArchCAD CDG oracle predictions as a bake-off sanity ceiling."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.drawing_graph import load_archcad_graph, validate_graph  # noqa: E402


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
        "--output-dir",
        type=Path,
        default=Path("outputs/drawing_graph/predictions_archcad_oracle_v0"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.task_manifest.read_text(encoding="utf-8"))
    if manifest.get("purpose") != "task_selection_only_not_an_evaluation_split":
        raise ValueError("task manifest has an unexpected purpose")
    population = manifest.get("subsets", {}).get(args.subset)
    if not population:
        raise ValueError(f"subset is empty or missing: {args.subset!r}")
    if args.sample_count <= 0:
        raise ValueError("sample_count must be positive")
    selected = sorted(
        random.Random(args.seed).sample(
            list(population), min(args.sample_count, len(population))
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for image_id in selected:
        graph = load_archcad_graph(args.data_root / "json" / f"{image_id}.json")
        validation = validate_graph(graph)
        if not validation.is_valid:
            raise ValueError(
                f"oracle compiler produced invalid graph for {image_id}: "
                f"{validation.to_dict()}"
            )
        graph.metadata["prediction_method"] = "archcad_raw_oracle_sanity_ceiling"
        graph.write_json(args.output_dir / f"{image_id}.cdg.json")
    summary = {
        "schema_version": "structure-baseline-predictions-v1",
        "method": "archcad_raw_oracle_sanity_ceiling",
        "subset": args.subset,
        "sample_count": len(selected),
        "seed": args.seed,
        "task_manifest_fingerprint": manifest.get("fingerprint"),
        "task_manifest_is_evaluation_split": False,
        "output_dir": str(args.output_dir),
    }
    summary_path = args.output_dir.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
