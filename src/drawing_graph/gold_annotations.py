"""Validation contract for manually reviewed room/connectivity gold records."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping

from src.drawing_graph.schema import CanonicalDrawingGraph, DrawingEdge, DrawingNode
from src.drawing_graph.validation import validate_graph


REVIEW_STATUSES = {"unreviewed", "in_progress", "reviewed", "rejected"}
VIEW_TYPES = {"floor_plan", "non_floor_plan", "ambiguous"}
REGION_TYPES = {"room", "corridor", "vertical_core", "shaft", "void", "other"}


def _valid_polygon(points: object) -> bool:
    if not (
        isinstance(points, list)
        and len(points) >= 3
        and all(
            isinstance(point, list)
            and len(point) == 2
            and all(isinstance(value, (int, float)) and math.isfinite(value) for value in point)
            for point in points
        )
    ):
        return False
    vertices = [(float(point[0]), float(point[1])) for point in points]
    if len(set(vertices)) != len(vertices):
        return False
    signed_area_twice = sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(vertices, vertices[1:] + vertices[:1])
    )
    if abs(signed_area_twice) <= 1e-6:
        return False

    def orientation(
        first: tuple[float, float],
        second: tuple[float, float],
        third: tuple[float, float],
    ) -> float:
        return (second[0] - first[0]) * (third[1] - first[1]) - (
            second[1] - first[1]
        ) * (third[0] - first[0])

    def on_segment(
        first: tuple[float, float],
        second: tuple[float, float],
        point: tuple[float, float],
    ) -> bool:
        return (
            min(first[0], second[0]) <= point[0] <= max(first[0], second[0])
            and min(first[1], second[1]) <= point[1] <= max(first[1], second[1])
        )

    def intersects(
        first: tuple[float, float],
        second: tuple[float, float],
        third: tuple[float, float],
        fourth: tuple[float, float],
    ) -> bool:
        o1 = orientation(first, second, third)
        o2 = orientation(first, second, fourth)
        o3 = orientation(third, fourth, first)
        o4 = orientation(third, fourth, second)
        if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
            return True
        epsilon = 1e-9
        return (
            (abs(o1) <= epsilon and on_segment(first, second, third))
            or (abs(o2) <= epsilon and on_segment(first, second, fourth))
            or (abs(o3) <= epsilon and on_segment(third, fourth, first))
            or (abs(o4) <= epsilon and on_segment(third, fourth, second))
        )

    edge_count = len(vertices)
    for first_index in range(edge_count):
        first_edge = (vertices[first_index], vertices[(first_index + 1) % edge_count])
        for second_index in range(first_index + 1, edge_count):
            if second_index in {
                first_index,
                (first_index + 1) % edge_count,
                (first_index - 1) % edge_count,
            }:
                continue
            second_edge = (
                vertices[second_index],
                vertices[(second_index + 1) % edge_count],
            )
            if intersects(*first_edge, *second_edge):
                return False
    return True


def validate_room_gold_record(record: Mapping[str, Any]) -> list[str]:
    """Return deterministic validation errors; an empty list means valid."""

    errors: list[str] = []
    if record.get("schema_version") != "room-gold-v0":
        errors.append("schema_version must be 'room-gold-v0'")
    if not record.get("image_id"):
        errors.append("image_id is required")
    status = record.get("review_status")
    if status not in REVIEW_STATUSES:
        errors.append(f"invalid review_status: {status!r}")
    annotation = record.get("annotation")
    if not isinstance(annotation, Mapping):
        return errors + ["annotation must be a mapping"]
    view_type = annotation.get("view_type")
    if status in {"reviewed", "rejected"} and view_type not in VIEW_TYPES:
        errors.append("reviewed records require a valid view_type")
    if status == "reviewed" and not record.get("reviewer"):
        errors.append("reviewed records require reviewer")

    regions = annotation.get("regions", [])
    if not isinstance(regions, list):
        return errors + ["annotation.regions must be a list"]
    region_ids: set[str] = set()
    for index, region in enumerate(regions):
        if not isinstance(region, Mapping):
            errors.append(f"regions[{index}] must be a mapping")
            continue
        region_id = region.get("id")
        if not isinstance(region_id, str) or not region_id:
            errors.append(f"regions[{index}].id is required")
        elif region_id in region_ids:
            errors.append(f"duplicate region id: {region_id}")
        else:
            region_ids.add(region_id)
        if region.get("type") not in REGION_TYPES:
            errors.append(f"regions[{index}] has invalid type")
        if not _valid_polygon(region.get("polygon")):
            errors.append(f"regions[{index}] has invalid polygon")

    relations = annotation.get("relations", [])
    if not isinstance(relations, list):
        return errors + ["annotation.relations must be a list"]
    for index, relation in enumerate(relations):
        if not isinstance(relation, Mapping):
            errors.append(f"relations[{index}] must be a mapping")
            continue
        if relation.get("relation") != "connected_by":
            errors.append(f"relations[{index}] must use connected_by")
        if relation.get("source") not in region_ids or relation.get("target") not in region_ids:
            errors.append(f"relations[{index}] references unknown region")
        if relation.get("source") == relation.get("target"):
            errors.append(f"relations[{index}] cannot connect a region to itself")
        if not relation.get("door_instance_id"):
            errors.append(f"relations[{index}] requires door_instance_id")
    return errors


def summarize_room_gold_records(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize review progress without treating candidate hints as labels."""

    status_counts = Counter(str(record.get("review_status")) for record in records)
    view_type_counts: Counter[str] = Counter()
    region_type_counts: Counter[str] = Counter()
    relation_count = 0
    reviewed_floor_plan_count = 0
    records_with_preannotations = 0

    for record in records:
        annotation = record.get("annotation")
        if isinstance(annotation, Mapping):
            view_type = annotation.get("view_type")
            if view_type is not None:
                view_type_counts[str(view_type)] += 1
            if record.get("review_status") == "reviewed" and view_type == "floor_plan":
                reviewed_floor_plan_count += 1
            regions = annotation.get("regions")
            if isinstance(regions, list):
                region_type_counts.update(
                    str(region.get("type"))
                    for region in regions
                    if isinstance(region, Mapping) and region.get("type") is not None
                )
            relations = annotation.get("relations")
            if isinstance(relations, list):
                relation_count += len(relations)
        preannotations = record.get("preannotations")
        if isinstance(preannotations, Mapping) and preannotations.get("regions"):
            records_with_preannotations += 1

    reviewed_count = status_counts["reviewed"]
    return {
        "record_count": len(records),
        "reviewed_count": reviewed_count,
        "reviewed_fraction": reviewed_count / len(records) if records else 0.0,
        "reviewed_floor_plan_count": reviewed_floor_plan_count,
        "review_status_counts": dict(sorted(status_counts.items())),
        "view_type_counts": dict(sorted(view_type_counts.items())),
        "region_count": sum(region_type_counts.values()),
        "region_type_counts": dict(sorted(region_type_counts.items())),
        "relation_count": relation_count,
        "records_with_preannotations": records_with_preannotations,
        "preannotations_are_ground_truth": False,
    }


def add_room_gold_annotations(
    graph: CanonicalDrawingGraph, record: Mapping[str, Any]
) -> dict[str, int]:
    """Add one reviewed human record to a CDG as non-candidate truth."""

    errors = validate_room_gold_record(record)
    if errors:
        raise ValueError(f"invalid room gold record: {errors}")
    if record.get("review_status") != "reviewed":
        raise ValueError("only reviewed room gold records can be compiled")
    if record.get("image_id") != graph.drawing_id:
        raise ValueError("room gold image_id must match graph.drawing_id")

    annotation = record["annotation"]
    if annotation.get("view_type") != "floor_plan":
        graph.metadata["room_gold"] = {
            "review_status": "reviewed",
            "view_type": annotation.get("view_type"),
            "reviewer": record.get("reviewer"),
            "region_count": 0,
            "relation_count": 0,
        }
        return {"region_count": 0, "relation_count": 0}

    existing_node_ids = {node.id for node in graph.nodes}
    existing_edge_ids = {edge.id for edge in graph.edges}
    region_node_ids: dict[str, str] = {}
    provenance = {
        "source": "human_room_gold",
        "schema_version": record["schema_version"],
        "reviewer": record.get("reviewer"),
    }
    for region in annotation["regions"]:
        node_id = f"gold_region:{region['id']}"
        if node_id in existing_node_ids:
            raise ValueError(f"gold region node already exists: {node_id}")
        graph.nodes.append(
            DrawingNode(
                id=node_id,
                kind="semantic",
                node_type="region",
                semantic_name=str(region["type"]),
                geometry={"type": "polygon", "points": region["polygon"]},
                confidence=1.0,
                attributes={"annotation_region_id": region["id"]},
                provenance=provenance,
            )
        )
        existing_node_ids.add(node_id)
        region_node_ids[str(region["id"])] = node_id

    for index, relation in enumerate(annotation["relations"]):
        door_node_id = f"instance:{relation['door_instance_id']}"
        if door_node_id not in existing_node_ids:
            raise ValueError(
                f"room gold relation references unknown door instance: {door_node_id}"
            )
        door_node = next(node for node in graph.nodes if node.id == door_node_id)
        if door_node.semantic_id not in {1, 2, 3, 4}:
            raise ValueError(
                "room gold relation door_instance_id must reference an ArchCAD door "
                f"instance, got semantic_id={door_node.semantic_id!r}"
            )
        edge_id = f"gold_connected_by:{index:04d}"
        if edge_id in existing_edge_ids:
            raise ValueError(f"gold relation edge already exists: {edge_id}")
        graph.edges.append(
            DrawingEdge(
                id=edge_id,
                relation="connected_by",
                source=region_node_ids[str(relation["source"])],
                target=region_node_ids[str(relation["target"])],
                confidence=1.0,
                attributes={
                    "door_instance_id": relation["door_instance_id"],
                    "door_node_id": door_node_id,
                },
                provenance=provenance,
            )
        )
        existing_edge_ids.add(edge_id)

    counts = {
        "region_count": len(annotation["regions"]),
        "relation_count": len(annotation["relations"]),
    }
    graph.metadata["room_gold"] = {
        "review_status": "reviewed",
        "view_type": "floor_plan",
        "reviewer": record.get("reviewer"),
        **counts,
    }
    validation = validate_graph(graph)
    if not validation.is_valid:
        raise ValueError(f"room gold produced an invalid graph: {validation.to_dict()}")
    return counts
