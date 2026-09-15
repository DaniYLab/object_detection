"""Comparable structural metrics over Canonical Drawing Graph predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

from src.drawing_graph.schema import CanonicalDrawingGraph, DrawingEdge, DrawingNode
from src.drawing_graph.validation import validate_graph


def _safe_f1(true_positive: int, predicted: int, gold: int) -> tuple[float, float, float]:
    precision = true_positive / predicted if predicted else (1.0 if not gold else 0.0)
    recall = true_positive / gold if gold else (1.0 if not predicted else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _is_human_gold(node: DrawingNode) -> bool:
    return node.id.startswith("gold_region:") or node.provenance.get("source") == "human_room_gold"


def _region(node: DrawingNode) -> DrawingNode | None:
    if node.node_type != "region" or node.geometry.get("type") not in {
        "polygon",
        "raster_region",
    }:
        return None
    return node


def _region_mask(node: DrawingNode, image_size: tuple[int, int]) -> np.ndarray:
    geometry_type = node.geometry.get("type")
    if geometry_type == "polygon":
        canvas = Image.new("1", image_size, 0)
        ImageDraw.Draw(canvas).polygon(
            [tuple(map(float, point)) for point in node.geometry.get("points", [])],
            fill=1,
        )
        return np.asarray(canvas, dtype=bool)
    mask = np.zeros((image_size[1], image_size[0]), dtype=bool)
    for row, start, end in node.geometry.get("spans", []):
        if 0 <= row < mask.shape[0]:
            mask[row, max(0, start) : min(mask.shape[1], end)] = True
    return mask


def _match_region_nodes(
    gold_regions: list[DrawingNode],
    prediction_regions: list[DrawingNode],
    image_size: tuple[int, int],
) -> list[tuple[str, str, float]]:
    if not gold_regions or not prediction_regions:
        return []
    gold_masks = [_region_mask(node, image_size) for node in gold_regions]
    prediction_masks = [_region_mask(node, image_size) for node in prediction_regions]
    scores = np.zeros((len(gold_regions), len(prediction_regions)), dtype=np.float64)
    for gold_index, gold_mask in enumerate(gold_masks):
        for prediction_index, prediction_mask in enumerate(prediction_masks):
            union = np.logical_or(gold_mask, prediction_mask).sum()
            scores[gold_index, prediction_index] = (
                np.logical_and(gold_mask, prediction_mask).sum() / union if union else 0.0
            )
    rows, columns = linear_sum_assignment(-scores)
    return [
        (
            gold_regions[row].id,
            prediction_regions[column].id,
            float(scores[row, column]),
        )
        for row, column in zip(rows.tolist(), columns.tolist())
    ]


def _door_id(edge: DrawingEdge) -> str:
    return str(
        edge.attributes.get("door_instance_id")
        or edge.attributes.get("door_id")
        or edge.attributes.get("door_node_id")
        or ""
    ).removeprefix("instance:")


def evaluate_structure_prediction(
    prediction: CanonicalDrawingGraph,
    gold: CanonicalDrawingGraph,
    *,
    region_iou_threshold: float = 0.50,
    image_size: tuple[int, int] = (980, 980),
) -> dict[str, Any]:
    """Score region detection and door connectivity using human-gold CDG nodes."""

    if prediction.drawing_id != gold.drawing_id:
        raise ValueError("prediction and gold drawing_id must match")
    if not 0.0 <= region_iou_threshold <= 1.0:
        raise ValueError("region_iou_threshold must be in [0, 1]")

    prediction_regions = [
        region
        for node in prediction.nodes
        if not _is_human_gold(node) and (region := _region(node)) is not None
    ]
    gold_regions = [
        region
        for node in gold.nodes
        if _is_human_gold(node) and (region := _region(node)) is not None
    ]
    raw_matches = _match_region_nodes(gold_regions, prediction_regions, image_size)
    matches = [match for match in raw_matches if match[2] >= region_iou_threshold]
    region_precision, region_recall, region_f1 = _safe_f1(
        len(matches), len(prediction_regions), len(gold_regions)
    )

    prediction_to_gold = {
        prediction_id: gold_id for gold_id, prediction_id, _ in matches
    }
    gold_relation_set = {
        (*sorted((edge.source, edge.target)), _door_id(edge))
        for edge in gold.edges
        if edge.relation == "connected_by"
        and edge.source in {region.id for region in gold_regions}
        and edge.target in {region.id for region in gold_regions}
    }
    prediction_relation_set: set[tuple[str, str, str]] = set()
    for edge in prediction.edges:
        if edge.relation not in {"connected_by", "candidate_connected_by"}:
            continue
        source = prediction_to_gold.get(edge.source, f"unmatched:{edge.source}")
        target = prediction_to_gold.get(edge.target, f"unmatched:{edge.target}")
        prediction_relation_set.add((*sorted((source, target)), _door_id(edge)))
    relation_true_positive = len(gold_relation_set & prediction_relation_set)
    relation_precision, relation_recall, relation_f1 = _safe_f1(
        relation_true_positive,
        len(prediction_relation_set),
        len(gold_relation_set),
    )
    validation = validate_graph(prediction)
    return {
        "drawing_id": prediction.drawing_id,
        "prediction_graph_valid": validation.is_valid,
        "prediction_validation": validation.to_dict(),
        "region_iou_threshold": region_iou_threshold,
        "gold_region_count": len(gold_regions),
        "predicted_region_count": len(prediction_regions),
        "matched_region_count": len(matches),
        "region_precision": region_precision,
        "region_recall": region_recall,
        "region_f1": region_f1,
        "mean_matched_polygon_iou": (
            sum(match[2] for match in matches) / len(matches) if matches else None
        ),
        "gold_connected_by_count": len(gold_relation_set),
        "predicted_connected_by_count": len(prediction_relation_set),
        "matched_connected_by_count": relation_true_positive,
        "connected_by_precision": relation_precision,
        "connected_by_recall": relation_recall,
        "connected_by_f1": relation_f1,
    }


def summarize_structure_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-drawing bake-off metrics without hiding invalid graphs."""

    if not records:
        return {
            "drawing_count": 0,
            "valid_graph_fraction": None,
            "macro_region_f1": None,
            "macro_connected_by_f1": None,
            "mean_matched_polygon_iou": None,
        }
    matched_ious = [
        record["mean_matched_polygon_iou"]
        for record in records
        if record["mean_matched_polygon_iou"] is not None
    ]
    return {
        "drawing_count": len(records),
        "valid_graph_fraction": sum(
            record["prediction_graph_valid"] for record in records
        )
        / len(records),
        "macro_region_f1": sum(record["region_f1"] for record in records)
        / len(records),
        "macro_connected_by_f1": sum(
            record["connected_by_f1"] for record in records
        )
        / len(records),
        "mean_matched_polygon_iou": (
            sum(matched_ious) / len(matched_ious) if matched_ious else None
        ),
    }
