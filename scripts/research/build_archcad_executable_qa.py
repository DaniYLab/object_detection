"""Build deterministic CDG queries with executable answers and graph evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.drawing_graph import execute_query, load_archcad_graph, validate_graph  # noqa: E402


QUERY_SPECS: tuple[tuple[str, str, dict[str, Any], str], ...] = (
    ("instance_count", "Có bao nhiêu semantic instances?", {"op": "count_nodes", "node_type": "instance"}, "raw_semantic_oracle"),
    ("wall_line_count", "Có bao nhiêu wall line primitives?", {"op": "count_nodes", "node_type": "line", "semantic_name": "wall"}, "raw_semantic_oracle"),
    ("single_door_count", "Có bao nhiêu single-door instances?", {"op": "count_nodes", "node_type": "instance", "semantic_name": "single_door"}, "raw_semantic_oracle"),
    ("double_door_count", "Có bao nhiêu double-door instances?", {"op": "count_nodes", "node_type": "instance", "semantic_name": "double_door"}, "raw_semantic_oracle"),
    ("parent_child_door_count", "Có bao nhiêu parent-child-door instances?", {"op": "count_nodes", "node_type": "instance", "semantic_name": "parent_child_door"}, "raw_semantic_oracle"),
    ("other_door_count", "Có bao nhiêu other-door instances?", {"op": "count_nodes", "node_type": "instance", "semantic_name": "other_door"}, "raw_semantic_oracle"),
    ("enclosed_face_candidate_count", "Có bao nhiêu enclosed-wall-face candidates?", {"op": "count_nodes", "semantic_name": "enclosed_wall_face_candidate"}, "heuristic_candidate_only"),
    ("door_wall_candidate_count", "Có bao nhiêu door-wall attachment candidates?", {"op": "count_relations", "relation": "candidate_attached_to"}, "heuristic_candidate_only"),
    ("instance_region_candidate_count", "Có bao nhiêu instance-region containment candidates?", {"op": "count_relations", "relation": "candidate_inside"}, "heuristic_candidate_only"),
    ("region_connectivity_candidate_count", "Có bao nhiêu region-connectivity candidates?", {"op": "count_relations", "relation": "candidate_connected_by"}, "heuristic_candidate_only"),
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
        "--output",
        type=Path,
        default=Path("outputs/drawing_graph/oracle_archcad_floor_plan_100_qa.jsonl"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_count <= 0:
        raise ValueError("--sample-count must be positive")
    manifest = json.loads(args.task_manifest.read_text(encoding="utf-8"))
    if manifest.get("purpose") != "task_selection_only_not_an_evaluation_split":
        raise ValueError("task manifest has an unexpected purpose")
    population = manifest.get("subsets", {}).get(args.subset)
    if not population:
        raise ValueError(f"subset is empty or missing: {args.subset!r}")
    selected = random.Random(args.seed).sample(
        population, min(args.sample_count, len(population))
    )
    selected.sort()

    records: list[dict[str, Any]] = []
    for image_id in selected:
        graph = load_archcad_graph(args.data_root / "json" / f"{image_id}.json")
        validation = validate_graph(graph)
        if not validation.is_valid:
            raise ValueError(f"invalid graph for {image_id}: {validation.to_dict()}")
        for query_name, question, program, supervision in QUERY_SPECS:
            result = execute_query(graph, program)
            records.append(
                {
                    "schema_version": "cdg-executable-qa-v0",
                    "id": f"{image_id}:{query_name}",
                    "image_id": image_id,
                    "source_drawing_id": graph.metadata.get("source_drawing_id"),
                    "question": question,
                    "program": program,
                    "answer": result.answer,
                    "evidence_nodes": list(result.evidence_nodes),
                    "evidence_edges": list(result.evidence_edges),
                    "supervision": supervision,
                }
            )

    serialized = [json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records]
    fingerprint = hashlib.sha256("\n".join(serialized).encode("utf-8")).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(serialized) + "\n", encoding="utf-8")
    summary = {
        "schema_version": "cdg-executable-qa-summary-v0",
        "fingerprint": fingerprint,
        "task_manifest_fingerprint": manifest["fingerprint"],
        "selection": args.subset,
        "seed": args.seed,
        "sample_count": len(selected),
        "query_count": len(records),
        "query_types": len(QUERY_SPECS),
        "raw_semantic_oracle_queries": sum(
            record["supervision"] == "raw_semantic_oracle" for record in records
        ),
        "heuristic_candidate_queries": sum(
            record["supervision"] == "heuristic_candidate_only" for record in records
        ),
        "source_drawing_provenance_available": all(
            record["source_drawing_id"] for record in records
        ),
        "publication_benchmark_ready": False,
        "blockers": [
            "source_drawing_id is absent from the local release",
            "heuristic candidate queries are not room/connectivity ground truth",
        ],
        "qa_file": str(args.output),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
