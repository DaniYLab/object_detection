"""Direct ArchCAD primitive/instance/junction metrics for structural bake-offs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from src.drawing_graph.schema import CanonicalDrawingGraph, DrawingNode
from src.drawing_graph.validation import validate_graph


PRIMITIVE_TYPES = {"line", "arc", "circle", "ellipse"}


def _safe_f1(true_positive: int, predicted: int, gold: int) -> tuple[float, float, float]:
    precision = true_positive / predicted if predicted else (1.0 if not gold else 0.0)
    recall = true_positive / gold if gold else (1.0 if not predicted else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _point(value: object) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return (0.0, 0.0)
    return float(value[0]), float(value[1])


def _primitive_feature(node: DrawingNode) -> tuple[float, ...]:
    geometry = node.geometry
    geometry_type = str(geometry.get("type"))
    if geometry_type == "line":
        start, end = _point(geometry.get("start")), _point(geometry.get("end"))
        first, second = sorted((start, end))
        return (*first, *second)
    if geometry_type == "arc":
        start, end = _point(geometry.get("start")), _point(geometry.get("end"))
        first, second = sorted((start, end))
        center = _point(geometry.get("center"))
        return (*center, float(geometry.get("radius", 0.0)), *first, *second)
    if geometry_type == "circle":
        return (*_point(geometry.get("center")), float(geometry.get("radius", 0.0)))
    if geometry_type == "ellipse":
        return (
            *_point(geometry.get("center")),
            *_point(geometry.get("major_axis")),
            float(geometry.get("ratio", 0.0)),
        )
    raise ValueError(f"unsupported primitive geometry type: {geometry_type!r}")


def _greedy_feature_matches(
    gold_nodes: list[DrawingNode],
    prediction_nodes: list[DrawingNode],
    *,
    tolerance: float,
) -> list[tuple[DrawingNode, DrawingNode, float]]:
    if not gold_nodes or not prediction_nodes:
        return []
    gold_features = np.asarray([_primitive_feature(node) for node in gold_nodes])
    prediction_features = np.asarray(
        [_primitive_feature(node) for node in prediction_nodes]
    )
    tree = cKDTree(gold_features)
    candidates: list[tuple[float, int, int]] = []
    for prediction_index, feature in enumerate(prediction_features):
        for gold_index in tree.query_ball_point(feature, tolerance):
            distance = float(np.linalg.norm(feature - gold_features[gold_index]))
            candidates.append((distance, gold_index, prediction_index))
    used_gold: set[int] = set()
    used_prediction: set[int] = set()
    matches = []
    for distance, gold_index, prediction_index in sorted(candidates):
        if gold_index in used_gold or prediction_index in used_prediction:
            continue
        used_gold.add(gold_index)
        used_prediction.add(prediction_index)
        matches.append(
            (gold_nodes[gold_index], prediction_nodes[prediction_index], distance)
        )
    return matches


def _match_primitives(
    gold_nodes: list[DrawingNode],
    prediction_nodes: list[DrawingNode],
    *,
    tolerance: float,
) -> list[tuple[DrawingNode, DrawingNode, float]]:
    gold_by_type: dict[str, list[DrawingNode]] = defaultdict(list)
    prediction_by_type: dict[str, list[DrawingNode]] = defaultdict(list)
    for node in gold_nodes:
        gold_by_type[str(node.geometry.get("type"))].append(node)
    for node in prediction_nodes:
        prediction_by_type[str(node.geometry.get("type"))].append(node)
    return [
        match
        for geometry_type in sorted(gold_by_type.keys() | prediction_by_type.keys())
        for match in _greedy_feature_matches(
            gold_by_type[geometry_type],
            prediction_by_type[geometry_type],
            tolerance=tolerance,
        )
    ]


def _bbox_iou(first: Iterable[float], second: Iterable[float]) -> float:
    a = list(map(float, first))
    b = list(map(float, second))
    intersection_width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    intersection_height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = intersection_width * intersection_height
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _has_positive_bbox(node: DrawingNode) -> bool:
    xyxy = node.geometry.get("xyxy")
    return (
        isinstance(xyxy, (list, tuple))
        and len(xyxy) == 4
        and float(xyxy[2]) > float(xyxy[0])
        and float(xyxy[3]) > float(xyxy[1])
    )


def _match_instances(
    gold_nodes: list[DrawingNode],
    prediction_nodes: list[DrawingNode],
    *,
    iou_threshold: float,
) -> list[tuple[DrawingNode, DrawingNode, float]]:
    matches = []
    semantic_ids = {
        node.semantic_id for node in [*gold_nodes, *prediction_nodes]
    }
    for semantic_id in semantic_ids:
        gold_class = [node for node in gold_nodes if node.semantic_id == semantic_id]
        prediction_class = [
            node for node in prediction_nodes if node.semantic_id == semantic_id
        ]
        if not gold_class or not prediction_class:
            continue
        scores = np.asarray(
            [
                [
                    _bbox_iou(gold.geometry["xyxy"], prediction.geometry["xyxy"])
                    for prediction in prediction_class
                ]
                for gold in gold_class
            ]
        )
        rows, columns = linear_sum_assignment(-scores)
        matches.extend(
            (gold_class[row], prediction_class[column], float(scores[row, column]))
            for row, column in zip(rows.tolist(), columns.tolist())
            if scores[row, column] >= iou_threshold
        )
    return matches


def _match_junctions(
    gold_nodes: list[DrawingNode],
    prediction_nodes: list[DrawingNode],
    *,
    tolerance: float,
) -> list[tuple[DrawingNode, DrawingNode, float]]:
    if not gold_nodes or not prediction_nodes:
        return []
    gold_points = np.asarray([_point(node.geometry.get("xy")) for node in gold_nodes])
    prediction_points = np.asarray(
        [_point(node.geometry.get("xy")) for node in prediction_nodes]
    )
    tree = cKDTree(gold_points)
    candidates = []
    for prediction_index, point in enumerate(prediction_points):
        for gold_index in tree.query_ball_point(point, tolerance):
            candidates.append(
                (
                    float(np.linalg.norm(point - gold_points[gold_index])),
                    gold_index,
                    prediction_index,
                )
            )
    used_gold: set[int] = set()
    used_prediction: set[int] = set()
    matches = []
    for distance, gold_index, prediction_index in sorted(candidates):
        if gold_index in used_gold or prediction_index in used_prediction:
            continue
        used_gold.add(gold_index)
        used_prediction.add(prediction_index)
        matches.append(
            (gold_nodes[gold_index], prediction_nodes[prediction_index], distance)
        )
    return matches


def evaluate_archcad_prediction(
    prediction: CanonicalDrawingGraph,
    ground_truth: CanonicalDrawingGraph,
    *,
    primitive_tolerance: float = 2.0,
    junction_tolerance: float = 2.0,
    instance_iou_threshold: float = 0.50,
) -> dict[str, Any]:
    """Evaluate a CDG prediction directly against compiled raw ArchCAD GT."""

    if prediction.drawing_id != ground_truth.drawing_id:
        raise ValueError("prediction and ground-truth drawing_id must match")
    if min(primitive_tolerance, junction_tolerance) < 0:
        raise ValueError("geometry tolerances must be non-negative")
    if not 0.0 <= instance_iou_threshold <= 1.0:
        raise ValueError("instance_iou_threshold must be in [0, 1]")

    gold_primitives = [
        node
        for node in ground_truth.nodes
        if node.id.startswith("primitive:") and node.geometry.get("type") in PRIMITIVE_TYPES
    ]
    prediction_primitives = [
        node
        for node in prediction.nodes
        if node.geometry.get("type") in PRIMITIVE_TYPES and node.node_type != "instance"
    ]
    primitive_matches = _match_primitives(
        gold_primitives, prediction_primitives, tolerance=primitive_tolerance
    )
    semantic_matches = [
        match
        for match in primitive_matches
        if match[0].semantic_id == match[1].semantic_id
    ]
    geometry_precision, geometry_recall, geometry_f1 = _safe_f1(
        len(primitive_matches), len(prediction_primitives), len(gold_primitives)
    )
    semantic_precision, semantic_recall, semantic_f1 = _safe_f1(
        len(semantic_matches), len(prediction_primitives), len(gold_primitives)
    )

    all_gold_instances = [
        node for node in ground_truth.nodes if node.node_type == "instance"
    ]
    all_prediction_instances = [
        node for node in prediction.nodes if node.node_type == "instance"
    ]
    gold_instances = [node for node in all_gold_instances if _has_positive_bbox(node)]
    prediction_instances = [
        node for node in all_prediction_instances if _has_positive_bbox(node)
    ]
    instance_matches = _match_instances(
        gold_instances,
        prediction_instances,
        iou_threshold=instance_iou_threshold,
    )
    instance_precision, instance_recall, instance_f1 = _safe_f1(
        len(instance_matches), len(prediction_instances), len(gold_instances)
    )

    gold_junctions = [node for node in ground_truth.nodes if node.kind == "junction"]
    prediction_junctions = [node for node in prediction.nodes if node.kind == "junction"]
    junction_matches = _match_junctions(
        gold_junctions, prediction_junctions, tolerance=junction_tolerance
    )
    junction_precision, junction_recall, junction_f1 = _safe_f1(
        len(junction_matches), len(prediction_junctions), len(gold_junctions)
    )

    prediction_to_gold = {
        prediction_node.id: gold_node.id
        for gold_node, prediction_node, _ in [*primitive_matches, *junction_matches]
    }
    gold_incidence = {
        (edge.source, edge.target)
        for edge in ground_truth.edges
        if edge.relation == "incident_to"
    }
    prediction_incidence = {
        (
            prediction_to_gold.get(edge.source, f"unmatched:{edge.source}"),
            prediction_to_gold.get(edge.target, f"unmatched:{edge.target}"),
        )
        for edge in prediction.edges
        if edge.relation == "incident_to"
    }
    incidence_true_positive = len(gold_incidence & prediction_incidence)
    incidence_precision, incidence_recall, incidence_f1 = _safe_f1(
        incidence_true_positive, len(prediction_incidence), len(gold_incidence)
    )

    semantic_ids = sorted(
        {
            node.semantic_id
            for node in [*gold_primitives, *prediction_primitives]
            if node.semantic_id is not None
        }
    )
    exact_counts = 0
    for semantic_id in semantic_ids:
        gold_count = sum(node.semantic_id == semantic_id for node in gold_primitives)
        prediction_count = sum(
            node.semantic_id == semantic_id for node in prediction_primitives
        )
        exact_counts += gold_count == prediction_count

    validation = validate_graph(prediction)
    return {
        "drawing_id": prediction.drawing_id,
        "prediction_graph_valid": validation.is_valid,
        "prediction_validation": validation.to_dict(),
        "gold_primitive_count": len(gold_primitives),
        "predicted_primitive_count": len(prediction_primitives),
        "matched_primitive_geometry_count": len(primitive_matches),
        "primitive_geometry_precision": geometry_precision,
        "primitive_geometry_recall": geometry_recall,
        "primitive_geometry_f1": geometry_f1,
        "primitive_semantic_precision": semantic_precision,
        "primitive_semantic_recall": semantic_recall,
        "primitive_semantic_f1": semantic_f1,
        "semantic_accuracy_on_geometry_matches": (
            len(semantic_matches) / len(primitive_matches) if primitive_matches else None
        ),
        "semantic_count_exact_fraction": (
            exact_counts / len(semantic_ids) if semantic_ids else 1.0
        ),
        "gold_instance_count": len(gold_instances),
        "predicted_instance_count": len(prediction_instances),
        "unscorable_gold_instance_count": len(all_gold_instances) - len(gold_instances),
        "unscorable_predicted_instance_count": (
            len(all_prediction_instances) - len(prediction_instances)
        ),
        "matched_instance_count": len(instance_matches),
        "instance_precision": instance_precision,
        "instance_recall": instance_recall,
        "instance_f1": instance_f1,
        "mean_matched_instance_iou": (
            sum(match[2] for match in instance_matches) / len(instance_matches)
            if instance_matches
            else None
        ),
        "gold_junction_count": len(gold_junctions),
        "predicted_junction_count": len(prediction_junctions),
        "matched_junction_count": len(junction_matches),
        "junction_precision": junction_precision,
        "junction_recall": junction_recall,
        "junction_f1": junction_f1,
        "incident_edge_precision": incidence_precision,
        "incident_edge_recall": incidence_recall,
        "incident_edge_f1": incidence_f1,
    }


def summarize_archcad_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Macro-average the direct-GT metrics used for model selection."""

    metric_names = (
        "primitive_geometry_f1",
        "primitive_semantic_f1",
        "semantic_count_exact_fraction",
        "instance_f1",
        "junction_f1",
        "incident_edge_f1",
    )
    if not records:
        return {"drawing_count": 0, **{f"macro_{name}": None for name in metric_names}}
    return {
        "drawing_count": len(records),
        "valid_graph_fraction": sum(
            record["prediction_graph_valid"] for record in records
        )
        / len(records),
        **{
            f"macro_{name}": sum(record[name] for record in records) / len(records)
            for name in metric_names
        },
    }
