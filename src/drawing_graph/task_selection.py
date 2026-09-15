"""Task-oriented selection for heterogeneous ArchCAD chunks."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping


DOOR_IDS = {1, 2, 3, 4}
FIXTURE_IDS = {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}
STRUCTURAL_ENGINEERING_IDS = {21, 22, 23, 24, 26, 27, 28}


def archcad_semantic_counts(annotation: Mapping[str, Any]) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for entity in annotation.get("entities", []):
        if not isinstance(entity, Mapping):
            continue
        semantic_id = entity.get("semantic")
        if isinstance(semantic_id, int):
            counts[semantic_id] += 1
    return dict(counts)


def classify_archcad_task(annotation: Mapping[str, Any]) -> dict[str, bool | int]:
    """Return transparent, non-exclusive task tags from raw semantic evidence."""

    counts = archcad_semantic_counts(annotation)
    wall_count = counts.get(20, 0)
    door_count = sum(counts.get(value, 0) for value in DOOR_IDS)
    fixture_count = sum(counts.get(value, 0) for value in FIXTURE_IDS)
    engineering_count = sum(
        counts.get(value, 0) for value in STRUCTURAL_ENGINEERING_IDS
    )
    door_instances = {
        str(entity["instance"])
        for entity in annotation.get("entities", [])
        if isinstance(entity, Mapping)
        and entity.get("semantic") in DOOR_IDS
        and entity.get("instance")
    }
    fixture_instances = {
        str(entity["instance"])
        for entity in annotation.get("entities", [])
        if isinstance(entity, Mapping)
        and entity.get("semantic") in FIXTURE_IDS
        and entity.get("instance")
    }
    wall_graph_candidate = wall_count >= 4
    return {
        "wall_primitives": wall_count,
        "door_primitives": door_count,
        "fixture_primitives": fixture_count,
        "engineering_primitives": engineering_count,
        "door_instances": len(door_instances),
        "fixture_instances": len(fixture_instances),
        "wall_present": wall_count > 0,
        "wall_graph_candidate": wall_graph_candidate,
        "door_connectivity_candidate": wall_graph_candidate and door_count > 0,
        "architectural_reading_candidate": wall_graph_candidate
        and (door_count > 0 or fixture_count > 0),
        "floor_plan_reading_candidate": wall_count >= 8
        and (
            len(door_instances) >= 2
            or (len(door_instances) >= 1 and len(fixture_instances) >= 1)
            or len(fixture_instances) >= 3
        ),
        "structure_engineering_candidate": engineering_count > 0,
    }
