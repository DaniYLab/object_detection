"""Structural and semantic validation for Canonical Drawing Graphs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from src.drawing_graph.schema import CDG_SCHEMA_VERSION, CanonicalDrawingGraph


NODE_KINDS = {"geometry", "semantic", "text", "junction"}
GEOMETRY_TYPES = {
    "point",
    "line",
    "arc",
    "circle",
    "ellipse",
    "bbox",
    "polygon",
    "raster_region",
    "none",
}
RELATION_TYPES = {
    "incident_to",
    "part_of",
    "contains",
    "inside",
    "adjacent_to",
    "connected_by",
    "intersects",
    "aligned_with",
    "refers_to",
    "bounds",
    "attached_to",
    "candidate_bounds",
    "candidate_attached_to",
    "candidate_inside",
    "candidate_connected_by",
}


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    entity_id: str | None = None


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def add(self, severity: str, code: str, message: str, entity_id: str | None = None) -> None:
        self.issues.append(ValidationIssue(severity, code, message, entity_id))

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable validation payload.

        Research scripts include this payload when rejecting a graph.  Keeping
        serialization on the report avoids error paths masking the original
        validation failure with an ``AttributeError``.
        """

        return {
            "is_valid": self.is_valid,
            "issues": [
                {
                    "severity": issue.severity,
                    "code": issue.code,
                    "message": issue.message,
                    "entity_id": issue.entity_id,
                }
                for issue in self.issues
            ],
        }


def _finite_point(value: object) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(v, (int, float)) and math.isfinite(v) for v in value)
    )


def _validate_geometry(report: ValidationReport, node_id: str, geometry: object) -> None:
    if not isinstance(geometry, dict):
        report.add("error", "geometry_not_mapping", "geometry must be a mapping", node_id)
        return
    geometry_type = geometry.get("type")
    if geometry_type not in GEOMETRY_TYPES:
        report.add("error", "unknown_geometry_type", f"unknown geometry type {geometry_type!r}", node_id)
        return
    if geometry_type == "point" and not _finite_point(geometry.get("xy")):
        report.add("error", "invalid_point", "point geometry requires finite xy", node_id)
    if geometry_type == "line":
        if not _finite_point(geometry.get("start")) or not _finite_point(geometry.get("end")):
            report.add("error", "invalid_line", "line requires finite start and end", node_id)
    if geometry_type == "bbox":
        xyxy = geometry.get("xyxy")
        valid = (
            isinstance(xyxy, (list, tuple))
            and len(xyxy) == 4
            and all(isinstance(v, (int, float)) and math.isfinite(v) for v in xyxy)
            and xyxy[0] <= xyxy[2]
            and xyxy[1] <= xyxy[3]
        )
        if not valid:
            report.add("error", "invalid_bbox", "bbox requires ordered finite xyxy", node_id)
    if geometry_type == "polygon":
        points = geometry.get("points")
        if not isinstance(points, (list, tuple)) or len(points) < 3 or not all(
            _finite_point(point) for point in points
        ):
            report.add("error", "invalid_polygon", "polygon requires at least three finite points", node_id)
    if geometry_type == "raster_region":
        bbox = geometry.get("bbox")
        spans = geometry.get("spans")
        if not (
            isinstance(bbox, (list, tuple))
            and len(bbox) == 4
            and all(isinstance(value, int) for value in bbox)
            and bbox[0] < bbox[2]
            and bbox[1] < bbox[3]
        ):
            report.add("error", "invalid_raster_region_bbox", "raster region requires integer bbox", node_id)
        if not isinstance(spans, (list, tuple)) or not spans:
            report.add("error", "invalid_raster_region_spans", "raster region requires row spans", node_id)


def validate_graph(graph: CanonicalDrawingGraph) -> ValidationReport:
    """Validate IDs, references, geometry, confidence, and required provenance."""

    report = ValidationReport()
    if graph.schema_version != CDG_SCHEMA_VERSION:
        report.add("error", "schema_version", f"expected {CDG_SCHEMA_VERSION!r}")
    if not graph.drawing_id:
        report.add("error", "drawing_id", "drawing_id must be non-empty")

    node_ids: set[str] = set()
    for node in graph.nodes:
        if node.id in node_ids:
            report.add("error", "duplicate_node_id", "node id is duplicated", node.id)
        node_ids.add(node.id)
        if node.kind not in NODE_KINDS:
            report.add("error", "unknown_node_kind", f"unknown node kind {node.kind!r}", node.id)
        if not 0.0 <= node.confidence <= 1.0:
            report.add("error", "node_confidence", "confidence must be in [0, 1]", node.id)
        if not node.provenance:
            report.add("warning", "missing_node_provenance", "node has no source provenance", node.id)
        _validate_geometry(report, node.id, node.geometry)

    node_by_id = {node.id: node for node in graph.nodes}
    edge_ids: set[str] = set()
    for edge in graph.edges:
        if edge.id in edge_ids:
            report.add("error", "duplicate_edge_id", "edge id is duplicated", edge.id)
        edge_ids.add(edge.id)
        if edge.relation not in RELATION_TYPES:
            report.add("error", "unknown_relation", f"unknown relation {edge.relation!r}", edge.id)
        if edge.source not in node_ids:
            report.add("error", "missing_edge_source", f"unknown source {edge.source!r}", edge.id)
        if edge.target not in node_ids:
            report.add("error", "missing_edge_target", f"unknown target {edge.target!r}", edge.id)
        if edge.source == edge.target:
            report.add("error", "self_edge", "self-relations are not allowed", edge.id)
        if not 0.0 <= edge.confidence <= 1.0:
            report.add("error", "edge_confidence", "confidence must be in [0, 1]", edge.id)
        if not edge.provenance:
            report.add(
                "warning",
                "missing_edge_provenance",
                "edge has no source/rule provenance",
                edge.id,
            )
        source_node = node_by_id.get(edge.source)
        target_node = node_by_id.get(edge.target)
        if edge.relation.startswith("candidate_") and edge.attributes.get("status") != "candidate":
            report.add(
                "error",
                "candidate_status",
                "candidate relations must carry attributes.status='candidate'",
                edge.id,
            )
        if edge.relation in {"connected_by", "candidate_connected_by"} and (
            source_node is not None
            and target_node is not None
            and (source_node.node_type != "region" or target_node.node_type != "region")
        ):
            report.add(
                "error",
                "connectivity_endpoint_type",
                "connected_by endpoints must both be region nodes",
                edge.id,
            )
        if edge.relation == "candidate_attached_to" and (
            source_node is not None
            and target_node is not None
            and (
                source_node.node_type != "instance"
                or source_node.semantic_id not in {1, 2, 3, 4}
                or target_node.semantic_id != 20
            )
        ):
            report.add(
                "error",
                "door_wall_attachment_type",
                "candidate_attached_to must connect a door instance to a wall primitive",
                edge.id,
            )
        if edge.relation == "candidate_bounds" and (
            source_node is not None
            and target_node is not None
            and (source_node.semantic_id != 20 or target_node.node_type != "region")
        ):
            report.add(
                "error",
                "region_boundary_type",
                "candidate_bounds must connect a wall primitive to a region",
                edge.id,
            )

    if not graph.metadata.get("source_drawing_id"):
        report.add(
            "warning",
            "missing_source_drawing_id",
            "chunk provenance is insufficient for source-drawing-disjoint evaluation",
        )
    return report
