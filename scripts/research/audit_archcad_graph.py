"""Build and audit deterministic CDG-v0 graphs from ArchCAD annotations."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.drawing_graph import execute_query, load_archcad_graph, validate_graph
from src.data.archcad_constants import ARCHCAD_RAW_SEMANTIC_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/ArchCAD"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--endpoint-tolerance", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/drawing_graph/oracle_archcad_100_report.json"),
    )
    parser.add_argument("--sample-graphs", type=int, default=5)
    parser.add_argument("--task-manifest", type=Path)
    parser.add_argument("--subset", default="architectural_reading_candidate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    annotation_dir = args.data_root / "json"
    task_manifest_fingerprint = None
    if args.task_manifest is not None:
        with args.task_manifest.open("r", encoding="utf-8") as handle:
            task_manifest = json.load(handle)
        if task_manifest.get("purpose") != "task_selection_only_not_an_evaluation_split":
            raise ValueError("task manifest has an unexpected purpose")
        if args.subset not in task_manifest.get("subsets", {}):
            raise ValueError(f"unknown task subset: {args.subset!r}")
        candidates = [
            annotation_dir / f"{image_id}.json"
            for image_id in task_manifest["subsets"][args.subset]
        ]
        task_manifest_fingerprint = task_manifest.get("fingerprint")
    else:
        candidates = sorted(annotation_dir.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"no ArchCAD annotations found in {annotation_dir}")

    rng = random.Random(args.seed)
    selected = rng.sample(candidates, min(args.limit, len(candidates)))
    selected.sort()

    error_codes: Counter[str] = Counter()
    warning_codes: Counter[str] = Counter()
    semantic_primitives: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    coverage: Counter[str] = Counter()
    query_checks = 0
    query_failures = 0
    valid_graphs = 0
    graph_dir = args.output.parent / "samples"

    for sample_index, annotation_path in enumerate(selected):
        with annotation_path.open("r", encoding="utf-8") as handle:
            source_annotation = json.load(handle)
        source_entities = [
            entity
            for entity in source_annotation.get("entities", [])
            if isinstance(entity, dict)
        ]
        graph = load_archcad_graph(
            annotation_path,
            endpoint_tolerance=args.endpoint_tolerance,
        )
        validation = validate_graph(graph)
        valid_graphs += int(validation.is_valid)
        error_codes.update(issue.code for issue in validation.errors)
        warning_codes.update(issue.code for issue in validation.warnings)
        totals["nodes"] += len(graph.nodes)
        totals["edges"] += len(graph.edges)
        totals["instances"] += int(graph.metadata["num_instances"])
        totals["junctions"] += int(graph.metadata["num_junctions"])
        totals["region_candidates"] += int(graph.metadata["num_region_candidates"])
        totals["door_wall_link_candidates"] += int(
            graph.metadata["num_door_wall_link_candidates"]
        )
        totals["instance_region_candidates"] += int(
            graph.metadata["num_instance_region_candidates"]
        )
        totals["region_connectivity_candidates"] += int(
            graph.metadata["num_region_connectivity_candidates"]
        )
        semantic_primitives.update(graph.metadata["semantic_primitive_counts"])
        semantic_counts = graph.metadata["semantic_primitive_counts"]
        coverage["samples_with_walls"] += int(semantic_counts.get("wall", 0) > 0)
        coverage["samples_with_doors"] += int(
            sum(
                semantic_counts.get(name, 0)
                for name in ("single_door", "double_door", "parent_child_door", "other_door")
            )
            > 0
        )
        coverage["samples_with_region_candidates"] += int(
            graph.metadata["num_region_candidates"] > 0
        )
        coverage["samples_with_connectivity_candidates"] += int(
            graph.metadata["num_region_connectivity_candidates"] > 0
        )

        # Executable-oracle checks compare query results to the raw annotation, not to
        # graph-derived counts.
        result = execute_query(graph, {"op": "count_nodes", "node_type": "instance"})
        source_instances = {
            str(entity["instance"])
            for entity in source_entities
            if entity.get("instance")
        }
        query_checks += 1
        query_failures += int(result.answer != len(source_instances))
        region_result = execute_query(
            graph,
            {"op": "count_nodes", "semantic_name": "enclosed_wall_face_candidate"},
        )
        query_checks += 1
        query_failures += int(
            region_result.answer != graph.metadata["num_region_candidates"]
        )
        attachment_result = execute_query(
            graph, {"op": "count_relations", "relation": "candidate_attached_to"}
        )
        query_checks += 1
        query_failures += int(
            attachment_result.answer
            != graph.metadata["num_door_wall_link_candidates"]
        )
        connectivity_result = execute_query(
            graph, {"op": "count_relations", "relation": "candidate_connected_by"}
        )
        query_checks += 1
        query_failures += int(
            connectivity_result.answer
            != graph.metadata["num_region_connectivity_candidates"]
        )

        source_line_counts: Counter[str] = Counter()
        for entity in source_entities:
            if str(entity.get("type", "")).upper() != "LINE":
                continue
            raw_id = entity.get("semantic")
            source_line_counts[ARCHCAD_RAW_SEMANTIC_NAMES.get(raw_id, "unknown")] += 1
        for semantic_name, expected_lines in source_line_counts.items():
            result = execute_query(
                graph,
                {
                    "op": "count_nodes",
                    "semantic_name": semantic_name,
                    "node_type": "line",
                },
            )
            query_checks += 1
            query_failures += int(result.answer != expected_lines)

        if sample_index < args.sample_graphs:
            graph.write_json(graph_dir / f"{graph.drawing_id}.cdg.json")

    count = len(selected)
    report = {
        "schema_version": "oracle-archcad-audit-v0",
        "seed": args.seed,
        "selection": args.subset if args.task_manifest is not None else "all_chunks",
        "task_manifest_fingerprint": task_manifest_fingerprint,
        "sample_count": count,
        "endpoint_tolerance": args.endpoint_tolerance,
        "valid_graphs": valid_graphs,
        "valid_graph_pct": round(100.0 * valid_graphs / count, 2),
        "error_codes": dict(sorted(error_codes.items())),
        "warning_codes": dict(sorted(warning_codes.items())),
        "source_drawing_provenance_available": False,
        "publication_split_ready": False,
        "publication_split_blocker": (
            "Local chunk annotations contain no source_drawing_id; recover the upstream "
            "chunk-to-drawing/project mapping before reporting generalization."
        ),
        "query_checks": query_checks,
        "query_failures": query_failures,
        "query_exact_pct": round(100.0 * (query_checks - query_failures) / query_checks, 2),
        "averages": {
            key: round(value / count, 2)
            for key, value in sorted(totals.items())
        },
        "sample_coverage": {
            key: {
                "count": value,
                "pct": round(100.0 * value / count, 2),
            }
            for key, value in sorted(coverage.items())
        },
        "semantic_primitive_counts": dict(sorted(semantic_primitives.items())),
        "sample_ids": [path.stem for path in selected],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
