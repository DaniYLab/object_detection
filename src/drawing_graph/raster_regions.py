"""Raster-assisted free-space candidates from semantic CAD wall primitives."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from src.drawing_graph.schema import CanonicalDrawingGraph, DrawingEdge, DrawingNode


DOOR_SEMANTIC_IDS = {1, 2, 3, 4}
VERTICAL_CORE_IDS = {5, 6}
SERVICE_FIXTURE_IDS = {7, 8, 9, 10, 11, 12, 13, 29}
OCCUPIED_SPACE_IDS = {14, 15, 16, 17}
VOID_IDS = {18}


@dataclass(frozen=True)
class RasterRegionCandidate:
    label: int
    area: int
    bbox: tuple[int, int, int, int]
    perimeter: int
    hydraulic_radius: float
    bbox_fill_ratio: float
    spans: tuple[tuple[int, int, int], ...]

    def geometry(self, image_size: tuple[int, int]) -> dict[str, Any]:
        return {
            "type": "raster_region",
            "image_size": list(image_size),
            "bbox": list(self.bbox),
            "area": self.area,
            "bbox_fill_ratio": self.bbox_fill_ratio,
            "spans": [list(span) for span in self.spans],
        }


def _point(value: object) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x, y = value[:2]
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return float(x), float(y)


def _entity_points(entity: Mapping[str, Any]) -> list[tuple[float, float]]:
    return [
        point
        for key in ("start", "end", "center")
        if (point := _point(entity.get(key))) is not None
    ]


def _projection_to_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[float, tuple[float, float]]:
    dx, dy = end[0] - start[0], end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator == 0:
        return math.dist(point, start), start
    fraction = max(
        0.0,
        min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denominator),
    )
    projection = (start[0] + fraction * dx, start[1] + fraction * dy)
    return math.dist(point, projection), projection


def build_wall_barrier_mask(
    annotation: Mapping[str, Any],
    *,
    image_size: tuple[int, int] = (980, 980),
    wall_width: int = 3,
    close_doors: bool = True,
    door_closure_width: int = 15,
    maximum_door_span: float = 180.0,
) -> tuple[np.ndarray, dict[str, int]]:
    """Rasterize semantic walls and optional evidence-guided door closures."""

    if wall_width <= 0 or door_closure_width <= 0:
        raise ValueError("barrier widths must be positive")
    entities = [
        entity
        for entity in annotation.get("entities", [])
        if isinstance(entity, Mapping)
    ]
    wall_lines: list[tuple[tuple[float, float], tuple[float, float]]] = []
    canvas = Image.new("1", image_size, 0)
    draw = ImageDraw.Draw(canvas)
    for entity in entities:
        if entity.get("semantic") != 20 or str(entity.get("type", "")).upper() != "LINE":
            continue
        start = _point(entity.get("start"))
        end = _point(entity.get("end"))
        if start is None or end is None:
            continue
        wall_lines.append((start, end))
        draw.line([start, end], fill=1, width=wall_width)

    closure_count = 0
    if close_doors and wall_lines:
        door_entities: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for entity in entities:
            instance_id = entity.get("instance")
            if instance_id and entity.get("semantic") in DOOR_SEMANTIC_IDS:
                door_entities[str(instance_id)].append(entity)
        for members in door_entities.values():
            points = [point for entity in members for point in _entity_points(entity)]
            if not points:
                continue
            center = (
                sum(point[0] for point in points) / len(points),
                sum(point[1] for point in points) / len(points),
            )
            distance, projection, wall = min(
                (
                    *_projection_to_segment(center, start, end),
                    (start, end),
                )
                for start, end in wall_lines
            )
            if distance > maximum_door_span:
                continue
            start, end = wall
            dx, dy = end[0] - start[0], end[1] - start[1]
            norm = math.hypot(dx, dy)
            if norm == 0:
                continue
            line_fraction = (
                (center[0] - start[0]) * dx + (center[1] - start[1]) * dy
            ) / (norm * norm)
            projection = (
                start[0] + line_fraction * dx,
                start[1] + line_fraction * dy,
            )
            unit = (dx / norm, dy / norm)
            projected = [
                point[0] * unit[0] + point[1] * unit[1] for point in points
            ]
            span = min(max(max(projected) - min(projected), 20.0), maximum_door_span)
            closure_start = (
                projection[0] - 0.6 * span * unit[0],
                projection[1] - 0.6 * span * unit[1],
            )
            closure_end = (
                projection[0] + 0.6 * span * unit[0],
                projection[1] + 0.6 * span * unit[1],
            )
            draw.line(
                [closure_start, closure_end],
                fill=1,
                width=door_closure_width,
            )
            closure_count += 1

    return np.asarray(canvas, dtype=bool), {
        "wall_line_count": len(wall_lines),
        "door_closure_count": closure_count,
    }


def _component_spans(mask: np.ndarray) -> tuple[tuple[int, int, int], ...]:
    spans: list[tuple[int, int, int]] = []
    for row_index, row in enumerate(mask):
        indices = np.flatnonzero(row)
        if not len(indices):
            continue
        starts = indices[np.r_[True, np.diff(indices) > 1]]
        ends = indices[np.r_[np.diff(indices) > 1, True]]
        spans.extend(
            (int(row_index), int(start), int(end) + 1)
            for start, end in zip(starts, ends)
        )
    return tuple(spans)


def infer_raster_free_space_candidates(
    annotation: Mapping[str, Any],
    *,
    image_size: tuple[int, int] = (980, 980),
    wall_width: int = 3,
    close_doors: bool = True,
    door_closure_width: int = 15,
    minimum_area: int = 1_000,
    minimum_hydraulic_radius: float = 12.0,
    minimum_bbox_fill_ratio: float = 0.35,
    maximum_area_fraction: float = 0.8,
) -> tuple[list[RasterRegionCandidate], dict[str, Any]]:
    """Find bounded free-space components while filtering thin wall strips."""

    if minimum_area <= 0:
        raise ValueError("minimum_area must be positive")
    if minimum_hydraulic_radius < 0:
        raise ValueError("minimum_hydraulic_radius must be non-negative")
    if not 0.0 <= minimum_bbox_fill_ratio <= 1.0:
        raise ValueError("minimum_bbox_fill_ratio must be in [0, 1]")
    if not 0.0 < maximum_area_fraction <= 1.0:
        raise ValueError("maximum_area_fraction must be in (0, 1]")

    barrier, diagnostics = build_wall_barrier_mask(
        annotation,
        image_size=image_size,
        wall_width=wall_width,
        close_doors=close_doors,
        door_closure_width=door_closure_width,
    )
    labels, component_count = ndimage.label(~barrier)
    border_labels = set(np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))))
    maximum_area = int(image_size[0] * image_size[1] * maximum_area_fraction)
    candidates: list[RasterRegionCandidate] = []
    rejected = defaultdict(int)
    for label_value in range(1, component_count + 1):
        if label_value in border_labels:
            rejected["touches_border"] += 1
            continue
        component = labels == label_value
        area = int(component.sum())
        if area < minimum_area:
            rejected["small_area"] += 1
            continue
        if area > maximum_area:
            rejected["large_area"] += 1
            continue
        rows, columns = np.nonzero(component)
        bbox = (int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1)
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        bbox_fill_ratio = area / max(bbox_area, 1)
        if bbox_fill_ratio < minimum_bbox_fill_ratio:
            rejected["sparse_component"] += 1
            continue
        boundary = component & ~ndimage.binary_erosion(component)
        perimeter = int(boundary.sum())
        hydraulic_radius = 2.0 * area / max(perimeter, 1)
        if hydraulic_radius < minimum_hydraulic_radius:
            rejected["thin_component"] += 1
            continue
        candidates.append(
            RasterRegionCandidate(
                label=label_value,
                area=area,
                bbox=bbox,
                perimeter=perimeter,
                hydraulic_radius=hydraulic_radius,
                bbox_fill_ratio=bbox_fill_ratio,
                spans=_component_spans(component),
            )
        )
    diagnostics.update(
        {
            "raw_component_count": int(component_count),
            "candidate_count": len(candidates),
            "rejected": dict(sorted(rejected.items())),
        }
    )
    return candidates, diagnostics


def _candidate_contains(candidate: RasterRegionCandidate, point: tuple[int, int]) -> bool:
    x, y = point
    if not (candidate.bbox[0] <= x < candidate.bbox[2] and candidate.bbox[1] <= y < candidate.bbox[3]):
        return False
    return any(row == y and start <= x < end for row, start, end in candidate.spans)


def _candidate_semantics(semantic_ids: set[int]) -> tuple[str, float]:
    if semantic_ids & VERTICAL_CORE_IDS:
        return "vertical_core_candidate", 0.7
    if semantic_ids & VOID_IDS:
        return "void_or_shaft_candidate", 0.6
    if semantic_ids & SERVICE_FIXTURE_IDS:
        return "service_space_candidate", 0.6
    if semantic_ids & OCCUPIED_SPACE_IDS:
        return "occupied_space_candidate", 0.6
    return "unclassified_free_space_candidate", 0.3


def add_raster_region_candidates(
    graph: CanonicalDrawingGraph,
    annotation: Mapping[str, Any],
    *,
    candidates: list[RasterRegionCandidate] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    **inference_options: Any,
) -> dict[str, Any]:
    """Append conservatively typed raster regions and candidate containment edges."""

    if candidates is None:
        candidates, inferred_diagnostics = infer_raster_free_space_candidates(
            annotation, **inference_options
        )
        diagnostics = inferred_diagnostics
    elif diagnostics is None:
        diagnostics = {}
    instances = [node for node in graph.nodes if node.node_type == "instance"]
    edge_counter = len(graph.edges)
    category_counts: dict[str, int] = defaultdict(int)
    containment_count = 0
    for index, candidate in enumerate(candidates):
        contained = []
        for instance in instances:
            xyxy = instance.geometry.get("xyxy")
            if not isinstance(xyxy, (list, tuple)) or len(xyxy) != 4:
                continue
            center = (round((xyxy[0] + xyxy[2]) / 2), round((xyxy[1] + xyxy[3]) / 2))
            if _candidate_contains(candidate, center):
                contained.append((instance, center))
        semantic_ids = {
            instance.semantic_id
            for instance, _ in contained
            if instance.semantic_id is not None
        }
        semantic_name, confidence = _candidate_semantics(semantic_ids)
        category_counts[semantic_name] += 1
        region_id = f"raster_region:{index:06d}"
        graph.nodes.append(
            DrawingNode(
                id=region_id,
                kind="geometry",
                node_type="region",
                geometry=candidate.geometry(tuple(graph.metadata.get("image_size", [980, 980]))),
                semantic_name=semantic_name,
                confidence=confidence,
                attributes={
                    "status": "candidate",
                    "algorithm": "semantic_wall_raster_free_space_v0",
                    "hydraulic_radius": candidate.hydraulic_radius,
                    "bbox_fill_ratio": candidate.bbox_fill_ratio,
                    "contained_semantic_ids": sorted(semantic_ids),
                },
                provenance={"dataset": "ArchCAD-400K", "rule": "raster_free_space_v0"},
            )
        )
        for instance, center in contained:
            graph.edges.append(
                DrawingEdge(
                    id=f"edge:{edge_counter:07d}",
                    relation="candidate_inside",
                    source=instance.id,
                    target=region_id,
                    confidence=confidence,
                    attributes={"status": "candidate", "test_point": list(center)},
                    provenance={"rule": "instance_bbox_center_in_raster_region_v0"},
                )
            )
            edge_counter += 1
            containment_count += 1
    graph.metadata["num_raster_region_candidates"] = len(candidates)
    graph.metadata["num_raster_region_containments"] = containment_count
    graph.metadata["raster_region_category_counts"] = dict(sorted(category_counts.items()))
    graph.metadata["raster_region_diagnostics"] = dict(diagnostics)
    return {
        "candidate_count": len(candidates),
        "containment_count": containment_count,
        "category_counts": dict(sorted(category_counts.items())),
        "diagnostics": dict(diagnostics),
    }
