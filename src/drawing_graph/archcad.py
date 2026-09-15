"""Deterministic ArchCAD annotation to Canonical Drawing Graph compiler."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.data.archcad_constants import ARCHCAD_RAW_SEMANTIC_NAMES
from src.drawing_graph.schema import CanonicalDrawingGraph, DrawingEdge, DrawingNode


STRUCTURAL_SEMANTIC_IDS = {0, 19, 20, 23, 24, 26, 28}
DOOR_SEMANTIC_IDS = {1, 2, 3, 4}


def _point(value: object) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x, y = value[:2]
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return [float(x), float(y)]


def _primitive_geometry(entity: Mapping[str, Any]) -> dict[str, Any]:
    primitive_type = str(entity.get("type", "")).upper()
    if primitive_type == "LINE":
        return {"type": "line", "start": _point(entity.get("start")), "end": _point(entity.get("end"))}
    if primitive_type == "ARC":
        return {
            "type": "arc",
            "center": _point(entity.get("center")),
            "radius": float(entity.get("radius", 0.0)),
            "start_angle": float(entity.get("start_angle", 0.0)),
            "end_angle": float(entity.get("end_angle", 0.0)),
            "start": _point(entity.get("start")),
            "end": _point(entity.get("end")),
        }
    if primitive_type == "CIRCLE":
        return {
            "type": "circle",
            "center": _point(entity.get("center")),
            "radius": float(entity.get("radius", 0.0)),
        }
    if primitive_type == "ELLIPSE":
        return {
            "type": "ellipse",
            "center": _point(entity.get("center")),
            "major_axis": _point(entity.get("major_axis")),
            "ratio": float(entity.get("ratio", 0.0)),
        }
    return {"type": "none"}


def _geometry_bounds(geometry: Mapping[str, Any]) -> list[float] | None:
    points = [
        list(value)
        for key in ("start", "end", "center")
        if (value := geometry.get(key)) is not None and _point(value) is not None
    ]
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    radius = geometry.get("radius")
    center = _point(geometry.get("center"))
    if center is not None and isinstance(radius, (int, float)) and radius >= 0:
        xs.extend([center[0] - radius, center[0] + radius])
        ys.extend([center[1] - radius, center[1] + radius])
    return [min(xs), min(ys), max(xs), max(ys)]


def _bbox(geometries: Iterable[Mapping[str, Any]]) -> list[float] | None:
    bounds = [bound for geometry in geometries if (bound := _geometry_bounds(geometry))]
    if not bounds:
        return None
    return [
        min(bound[0] for bound in bounds),
        min(bound[1] for bound in bounds),
        max(bound[2] for bound in bounds),
        max(bound[3] for bound in bounds),
    ]


def _endpoint_key(point: list[float], tolerance: float) -> tuple[int, int]:
    return (round(point[0] / tolerance), round(point[1] / tolerance))


def _signed_area(points: list[list[float]]) -> float:
    return 0.5 * sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )


def _point_to_segment_projection(
    point: list[float], start: list[float], end: list[float]
) -> tuple[float, list[float]]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator == 0:
        return math.hypot(point[0] - start[0], point[1] - start[1]), list(start)
    fraction = max(
        0.0,
        min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denominator),
    )
    projection = [start[0] + fraction * dx, start[1] + fraction * dy]
    return math.hypot(point[0] - projection[0], point[1] - projection[1]), projection


def _infer_wall_faces(
    nodes: list[DrawingNode],
    edges: list[DrawingEdge],
    *,
    tolerance: float,
    minimum_area: float,
    edge_counter: int,
) -> tuple[int, int]:
    """Add simple enclosed wall faces as auditable room candidates."""

    wall_lines = [
        node
        for node in nodes
        if node.semantic_id == 20 and node.geometry.get("type") == "line"
    ]
    coordinates: dict[tuple[int, int], list[list[float]]] = defaultdict(list)
    adjacency: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    boundary_primitives: dict[frozenset[tuple[int, int]], list[str]] = defaultdict(list)
    raw_lines: list[
        tuple[DrawingNode, list[float], list[float], tuple[int, int], tuple[int, int]]
    ] = []
    for node in wall_lines:
        start = _point(node.geometry.get("start"))
        end = _point(node.geometry.get("end"))
        if start is None or end is None:
            continue
        start_key = _endpoint_key(start, tolerance)
        end_key = _endpoint_key(end, tolerance)
        if start_key == end_key:
            continue
        coordinates[start_key].append(start)
        coordinates[end_key].append(end)
        raw_lines.append((node, start, end, start_key, end_key))

    vertex_xy = {
        key: [
            sum(point[0] for point in values) / len(values),
            sum(point[1] for point in values) / len(values),
        ]
        for key, values in coordinates.items()
    }
    # Split long wall lines at every annotated endpoint lying on the segment. This
    # converts T-junctions into a planar graph without inventing raster intersections.
    for node, start, end, start_key, end_key in raw_lines:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        denominator = dx * dx + dy * dy
        if denominator == 0:
            continue
        split_vertices: list[tuple[float, tuple[int, int]]] = []
        for candidate_key, candidate in vertex_xy.items():
            fraction = (
                (candidate[0] - start[0]) * dx + (candidate[1] - start[1]) * dy
            ) / denominator
            if fraction < 0.0 or fraction > 1.0:
                continue
            projection = [start[0] + fraction * dx, start[1] + fraction * dy]
            distance = math.hypot(
                candidate[0] - projection[0], candidate[1] - projection[1]
            )
            if distance <= tolerance:
                split_vertices.append((fraction, candidate_key))
        if not split_vertices:
            split_vertices = [(0.0, start_key), (1.0, end_key)]
        ordered_keys = list(dict.fromkeys(key for _, key in sorted(split_vertices)))
        for left, right in zip(ordered_keys, ordered_keys[1:]):
            if left == right:
                continue
            adjacency[left].add(right)
            adjacency[right].add(left)
            boundary_primitives[frozenset((left, right))].append(node.id)
    ordered_neighbors = {
        vertex: sorted(
            neighbors,
            key=lambda neighbor: math.atan2(
                vertex_xy[neighbor][1] - vertex_xy[vertex][1],
                vertex_xy[neighbor][0] - vertex_xy[vertex][0],
            ),
        )
        for vertex, neighbors in adjacency.items()
    }

    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    region_count = 0
    maximum_steps = max(1, 2 * sum(len(value) for value in adjacency.values()))
    for start_vertex in sorted(adjacency):
        for next_vertex in ordered_neighbors[start_vertex]:
            start_half_edge = (start_vertex, next_vertex)
            if start_half_edge in visited:
                continue
            face: list[tuple[int, int]] = []
            current, following = start_half_edge
            closed = False
            for _ in range(maximum_steps):
                if (current, following) in visited:
                    break
                visited.add((current, following))
                face.append(current)
                neighbors = ordered_neighbors[following]
                incoming_index = neighbors.index(current)
                successor = neighbors[(incoming_index - 1) % len(neighbors)]
                current, following = following, successor
                if (current, following) == start_half_edge:
                    closed = True
                    break
            if not closed or len(face) < 3 or len(face) != len(set(face)):
                continue
            polygon = [vertex_xy[vertex] for vertex in face]
            area = _signed_area(polygon)
            if area < minimum_area:
                continue

            region_id = f"region:{region_count:06d}"
            nodes.append(
                DrawingNode(
                    id=region_id,
                    kind="geometry",
                    node_type="region",
                    geometry={"type": "polygon", "points": polygon, "area": area},
                    semantic_name="enclosed_wall_face_candidate",
                    confidence=0.5,
                    attributes={
                        "status": "candidate",
                        "algorithm": "wall_half_edge_face_v0",
                        "minimum_area": minimum_area,
                    },
                    provenance={"dataset": "ArchCAD-400K", "rule": "enclosed_wall_face"},
                )
            )
            for index, vertex in enumerate(face):
                boundary = frozenset((vertex, face[(index + 1) % len(face)]))
                for primitive_id in boundary_primitives[boundary]:
                    edges.append(
                        DrawingEdge(
                            id=f"edge:{edge_counter:07d}",
                            relation="candidate_bounds",
                            source=primitive_id,
                            target=region_id,
                            confidence=0.5,
                            attributes={"status": "candidate"},
                            provenance={"rule": "wall_half_edge_face_v0"},
                        )
                    )
                    edge_counter += 1
            region_count += 1
    return region_count, edge_counter


def _link_doors_to_walls(
    nodes: list[DrawingNode],
    edges: list[DrawingEdge],
    *,
    maximum_distance: float,
    edge_counter: int,
) -> tuple[int, int]:
    """Add distance-backed candidate attachments from door instances to wall lines."""

    node_by_id = {node.id: node for node in nodes}
    wall_lines = [
        node
        for node in nodes
        if node.semantic_id == 20 and node.geometry.get("type") == "line"
    ]
    part_members: dict[str, list[DrawingNode]] = defaultdict(list)
    for edge in edges:
        if edge.relation == "part_of" and edge.source in node_by_id:
            part_members[edge.target].append(node_by_id[edge.source])

    link_count = 0
    door_instances = [
        node
        for node in nodes
        if node.node_type == "instance" and node.semantic_id in DOOR_SEMANTIC_IDS
    ]
    for door in door_instances:
        door_points = [
            point
            for primitive in part_members.get(door.id, [])
            for point in _geometry_points_for_linking(primitive.geometry)
        ]
        candidates: list[tuple[float, str, list[float], list[float]]] = []
        for wall in wall_lines:
            start = _point(wall.geometry.get("start"))
            end = _point(wall.geometry.get("end"))
            if start is None or end is None or not door_points:
                continue
            projected = [
                (*_point_to_segment_projection(point, start, end), point)
                for point in door_points
            ]
            distance, wall_point, door_point = min(projected, key=lambda value: value[0])
            if distance <= maximum_distance:
                candidates.append((distance, wall.id, door_point, wall_point))
        for distance, wall_id, door_point, wall_point in sorted(candidates)[:4]:
            confidence = max(0.0, 1.0 - distance / maximum_distance)
            edges.append(
                DrawingEdge(
                    id=f"edge:{edge_counter:07d}",
                    relation="candidate_attached_to",
                    source=door.id,
                    target=wall_id,
                    confidence=confidence,
                    attributes={
                        "status": "candidate",
                        "distance_px": distance,
                        "door_point": door_point,
                        "wall_point": wall_point,
                    },
                    provenance={"rule": "door_primitive_to_wall_distance_v0"},
                )
            )
            edge_counter += 1
            link_count += 1
    return link_count, edge_counter


def _geometry_points_for_linking(geometry: Mapping[str, Any]) -> list[list[float]]:
    return [
        point
        for key in ("start", "end", "center")
        if (point := _point(geometry.get(key))) is not None
    ]


def _point_in_polygon(point: list[float], polygon: list[list[float]]) -> bool:
    inside = False
    x, y = point
    previous = polygon[-1]
    for current in polygon:
        x0, y0 = previous
        x1, y1 = current
        crosses = (y0 > y) != (y1 > y)
        if crosses:
            intersection_x = (x1 - x0) * (y - y0) / (y1 - y0) + x0
            if x < intersection_x:
                inside = not inside
        previous = current
    return inside


def _infer_candidate_topology(
    nodes: list[DrawingNode],
    edges: list[DrawingEdge],
    *,
    edge_counter: int,
) -> tuple[int, int, int]:
    """Compose candidate containment and door-mediated region connectivity."""

    regions = [
        node
        for node in nodes
        if node.semantic_name == "enclosed_wall_face_candidate"
        and node.geometry.get("type") == "polygon"
    ]
    instances = [node for node in nodes if node.node_type == "instance"]
    containment_count = 0
    for instance in instances:
        xyxy = instance.geometry.get("xyxy")
        if not isinstance(xyxy, (list, tuple)) or len(xyxy) != 4:
            continue
        center = [(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2]
        for region in regions:
            polygon = region.geometry.get("points", [])
            if _point_in_polygon(center, polygon):
                edges.append(
                    DrawingEdge(
                        id=f"edge:{edge_counter:07d}",
                        relation="candidate_inside",
                        source=instance.id,
                        target=region.id,
                        confidence=0.5,
                        attributes={"status": "candidate", "test_point": center},
                        provenance={"rule": "instance_bbox_center_in_wall_face_v0"},
                    )
                )
                edge_counter += 1
                containment_count += 1

    wall_regions: dict[str, set[str]] = defaultdict(set)
    door_walls: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        if edge.relation == "candidate_bounds":
            wall_regions[edge.source].add(edge.target)
        elif edge.relation == "candidate_attached_to":
            door_walls[edge.source].add(edge.target)

    connectivity_count = 0
    emitted: set[tuple[str, str, str]] = set()
    for door_id, wall_ids in sorted(door_walls.items()):
        adjacent_regions = sorted(
            {region_id for wall_id in wall_ids for region_id in wall_regions[wall_id]}
        )
        for left_index, source_region in enumerate(adjacent_regions):
            for target_region in adjacent_regions[left_index + 1 :]:
                key = (source_region, target_region, door_id)
                if key in emitted:
                    continue
                emitted.add(key)
                edges.append(
                    DrawingEdge(
                        id=f"edge:{edge_counter:07d}",
                        relation="candidate_connected_by",
                        source=source_region,
                        target=target_region,
                        confidence=0.5,
                        attributes={"status": "candidate", "door_id": door_id},
                        provenance={"rule": "shared_bounding_wall_with_attached_door_v0"},
                    )
                )
                edge_counter += 1
                connectivity_count += 1
    return containment_count, connectivity_count, edge_counter


def compile_archcad_graph(
    annotation: Mapping[str, Any],
    *,
    drawing_id: str,
    source_path: str | None = None,
    source_drawing_id: str | None = None,
    image_size: tuple[int, int] = (980, 980),
    endpoint_tolerance: float = 1.0,
    infer_topology_candidates: bool = True,
    minimum_region_area: float = 100.0,
    door_wall_distance: float = 5.0,
) -> CanonicalDrawingGraph:
    """Compile all ArchCAD semantics without the detector's 21-class filtering."""

    if endpoint_tolerance <= 0:
        raise ValueError("endpoint_tolerance must be positive")
    if minimum_region_area < 0:
        raise ValueError("minimum_region_area must be non-negative")
    if door_wall_distance <= 0:
        raise ValueError("door_wall_distance must be positive")
    entities = annotation.get("entities", [])
    if not isinstance(entities, list):
        raise TypeError("ArchCAD annotation field 'entities' must be a list")

    nodes: list[DrawingNode] = []
    edges: list[DrawingEdge] = []
    instance_members: dict[str, list[tuple[str, dict[str, Any], int | None]]] = defaultdict(list)
    endpoint_members: dict[tuple[int, int], list[tuple[str, list[float], int]]] = defaultdict(list)

    for entity_index, entity in enumerate(entities):
        if not isinstance(entity, Mapping):
            continue
        semantic_id = entity.get("semantic")
        semantic_name = ARCHCAD_RAW_SEMANTIC_NAMES.get(semantic_id, "unknown")
        geometry = _primitive_geometry(entity)
        primitive_id = f"primitive:{entity_index:06d}"
        kind = "geometry" if semantic_id in STRUCTURAL_SEMANTIC_IDS else "semantic"
        nodes.append(
            DrawingNode(
                id=primitive_id,
                kind=kind,
                node_type=geometry["type"],
                geometry=geometry,
                semantic_id=semantic_id if isinstance(semantic_id, int) else None,
                semantic_name=semantic_name,
                attributes={
                    "linetype": entity.get("linetype"),
                    "line_width": entity.get("line_width"),
                    "rgb": entity.get("rgb"),
                },
                provenance={"dataset": "ArchCAD-400K", "entity_indices": [entity_index]},
            )
        )

        instance_id = entity.get("instance")
        if instance_id:
            instance_members[str(instance_id)].append((primitive_id, geometry, semantic_id))

        if geometry["type"] in {"line", "arc"}:
            for endpoint_index, key in enumerate(("start", "end")):
                point = geometry.get(key)
                if _point(point) is not None:
                    endpoint_members[_endpoint_key(point, endpoint_tolerance)].append(
                        (primitive_id, list(point), endpoint_index)
                    )

    edge_counter = 0
    for raw_instance_id, members in sorted(instance_members.items()):
        semantic_id = next((member[2] for member in members if isinstance(member[2], int)), None)
        semantic_name = ARCHCAD_RAW_SEMANTIC_NAMES.get(semantic_id, "unknown")
        instance_node_id = f"instance:{raw_instance_id}"
        instance_bbox = _bbox(member[1] for member in members)
        nodes.append(
            DrawingNode(
                id=instance_node_id,
                kind="semantic",
                node_type="instance",
                geometry={"type": "bbox", "xyxy": instance_bbox or [0.0, 0.0, 0.0, 0.0]},
                semantic_id=semantic_id,
                semantic_name=semantic_name,
                attributes={"raw_instance_id": raw_instance_id, "primitive_count": len(members)},
                provenance={
                    "dataset": "ArchCAD-400K",
                    "entity_indices": [int(member[0].split(":")[1]) for member in members],
                },
            )
        )
        for primitive_id, _, _ in members:
            edges.append(
                DrawingEdge(
                    id=f"edge:{edge_counter:07d}",
                    relation="part_of",
                    source=primitive_id,
                    target=instance_node_id,
                    provenance={"rule": "shared_archcad_instance_id"},
                )
            )
            edge_counter += 1

    junction_count = 0
    for bucket, members in sorted(endpoint_members.items()):
        unique_primitives = {member[0] for member in members}
        if len(unique_primitives) < 2:
            continue
        mean_point = [
            sum(member[1][0] for member in members) / len(members),
            sum(member[1][1] for member in members) / len(members),
        ]
        junction_id = f"junction:{junction_count:06d}"
        nodes.append(
            DrawingNode(
                id=junction_id,
                kind="junction",
                node_type="junction",
                geometry={"type": "point", "xy": mean_point},
                attributes={"endpoint_tolerance": endpoint_tolerance, "bucket": list(bucket)},
                provenance={"dataset": "ArchCAD-400K", "rule": "quantized_shared_endpoint"},
            )
        )
        for primitive_id in sorted(unique_primitives):
            edges.append(
                DrawingEdge(
                    id=f"edge:{edge_counter:07d}",
                    relation="incident_to",
                    source=primitive_id,
                    target=junction_id,
                    provenance={"rule": "quantized_shared_endpoint"},
                )
            )
            edge_counter += 1
        junction_count += 1

    region_count = 0
    door_wall_link_count = 0
    containment_count = 0
    connectivity_count = 0
    if infer_topology_candidates:
        region_count, edge_counter = _infer_wall_faces(
            nodes,
            edges,
            tolerance=endpoint_tolerance,
            minimum_area=minimum_region_area,
            edge_counter=edge_counter,
        )
        door_wall_link_count, edge_counter = _link_doors_to_walls(
            nodes,
            edges,
            maximum_distance=door_wall_distance,
            edge_counter=edge_counter,
        )
        containment_count, connectivity_count, edge_counter = _infer_candidate_topology(
            nodes,
            edges,
            edge_counter=edge_counter,
        )

    semantic_counts: dict[str, int] = defaultdict(int)
    for node in nodes:
        if node.id.startswith("primitive:") and node.semantic_name:
            semantic_counts[node.semantic_name] += 1

    return CanonicalDrawingGraph(
        drawing_id=drawing_id,
        nodes=nodes,
        edges=edges,
        metadata={
            "dataset": "ArchCAD-400K",
            "source_path": source_path,
            "source_drawing_id": source_drawing_id,
            "image_size": list(image_size),
            "endpoint_tolerance": endpoint_tolerance,
            "num_source_entities": len(entities),
            "num_instances": len(instance_members),
            "num_junctions": junction_count,
            "num_region_candidates": region_count,
            "num_door_wall_link_candidates": door_wall_link_count,
            "num_instance_region_candidates": containment_count,
            "num_region_connectivity_candidates": connectivity_count,
            "semantic_primitive_counts": dict(sorted(semantic_counts.items())),
        },
    )


def load_archcad_graph(
    path: str | Path,
    *,
    source_drawing_id: str | None = None,
    endpoint_tolerance: float = 1.0,
) -> CanonicalDrawingGraph:
    annotation_path = Path(path)
    with annotation_path.open("r", encoding="utf-8") as handle:
        annotation = json.load(handle)
    return compile_archcad_graph(
        annotation,
        drawing_id=annotation_path.stem,
        source_path=str(annotation_path),
        source_drawing_id=source_drawing_id,
        endpoint_tolerance=endpoint_tolerance,
    )
