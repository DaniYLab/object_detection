"""Canonical Drawing Graph public API."""

from src.drawing_graph.archcad import compile_archcad_graph, load_archcad_graph
from src.drawing_graph.annotation_store import AnnotationConflictError, RoomGoldStore
from src.drawing_graph.agreement import (
    RegionMatch,
    compare_room_gold_records,
    match_regions,
    polygon_iou,
    summarize_room_gold_agreement,
)
from src.drawing_graph.gold_annotations import (
    add_room_gold_annotations,
    summarize_room_gold_records,
    validate_room_gold_record,
)
from src.drawing_graph.metrics import (
    evaluate_structure_prediction,
    summarize_structure_metrics,
)
from src.drawing_graph.queries import QueryResult, execute_query
from src.drawing_graph.raster_regions import (
    RasterRegionCandidate,
    add_raster_region_candidates,
    build_wall_barrier_mask,
    infer_raster_free_space_candidates,
)
from src.drawing_graph.schema import DrawingEdge, DrawingNode, CanonicalDrawingGraph
from src.drawing_graph.task_selection import archcad_semantic_counts, classify_archcad_task
from src.drawing_graph.validation import ValidationIssue, ValidationReport, validate_graph

__all__ = [
    "CanonicalDrawingGraph",
    "AnnotationConflictError",
    "DrawingEdge",
    "DrawingNode",
    "QueryResult",
    "RegionMatch",
    "RasterRegionCandidate",
    "RoomGoldStore",
    "add_raster_region_candidates",
    "add_room_gold_annotations",
    "ValidationIssue",
    "ValidationReport",
    "compile_archcad_graph",
    "build_wall_barrier_mask",
    "archcad_semantic_counts",
    "classify_archcad_task",
    "compare_room_gold_records",
    "execute_query",
    "evaluate_structure_prediction",
    "infer_raster_free_space_candidates",
    "load_archcad_graph",
    "match_regions",
    "polygon_iou",
    "validate_graph",
    "validate_room_gold_record",
    "summarize_room_gold_records",
    "summarize_room_gold_agreement",
    "summarize_structure_metrics",
]
