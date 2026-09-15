"""Independent-review agreement metrics for room/connectivity gold packs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

from src.drawing_graph.gold_annotations import validate_room_gold_record


@dataclass(frozen=True)
class RegionMatch:
    primary_id: str
    secondary_id: str
    region_type: str
    iou: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_id": self.primary_id,
            "secondary_id": self.secondary_id,
            "region_type": self.region_type,
            "iou": self.iou,
        }


def _polygon_mask(
    polygon: Sequence[Sequence[float]], image_size: tuple[int, int]
) -> np.ndarray:
    canvas = Image.new("1", image_size, 0)
    ImageDraw.Draw(canvas).polygon(
        [(float(point[0]), float(point[1])) for point in polygon], fill=1
    )
    return np.asarray(canvas, dtype=bool)


def polygon_iou(
    first: Sequence[Sequence[float]],
    second: Sequence[Sequence[float]],
    *,
    image_size: tuple[int, int] = (980, 980),
) -> float:
    """Compute deterministic pixel-space IoU for two annotation polygons."""

    first_mask = _polygon_mask(first, image_size)
    second_mask = _polygon_mask(second, image_size)
    union = np.logical_or(first_mask, second_mask).sum()
    if not union:
        return 0.0
    return float(np.logical_and(first_mask, second_mask).sum() / union)


def match_regions(
    primary_regions: Sequence[Mapping[str, Any]],
    secondary_regions: Sequence[Mapping[str, Any]],
    *,
    image_size: tuple[int, int] = (980, 980),
    require_same_type: bool = True,
) -> list[RegionMatch]:
    """Maximum-IoU one-to-one matching, constrained to equal region types."""

    if not primary_regions or not secondary_regions:
        return []
    scores = np.full((len(primary_regions), len(secondary_regions)), -1.0)
    for primary_index, primary in enumerate(primary_regions):
        for secondary_index, secondary in enumerate(secondary_regions):
            if require_same_type and primary.get("type") != secondary.get("type"):
                continue
            scores[primary_index, secondary_index] = polygon_iou(
                primary["polygon"], secondary["polygon"], image_size=image_size
            )
    rows, columns = linear_sum_assignment(-scores)
    matches: list[RegionMatch] = []
    for row, column in zip(rows.tolist(), columns.tolist()):
        if scores[row, column] < 0.0:
            continue
        primary = primary_regions[row]
        secondary = secondary_regions[column]
        matches.append(
            RegionMatch(
                primary_id=str(primary["id"]),
                secondary_id=str(secondary["id"]),
                region_type=str(primary.get("type", "region")),
                iou=float(scores[row, column]),
            )
        )
    return matches


def _normalized_relations(
    relations: Sequence[Mapping[str, Any]],
    region_mapping: Mapping[str, str],
    *,
    unmatched_prefix: str,
) -> set[tuple[str, str, str]]:
    normalized: set[tuple[str, str, str]] = set()
    for relation in relations:
        source = region_mapping.get(
            str(relation["source"]), f"{unmatched_prefix}:{relation['source']}"
        )
        target = region_mapping.get(
            str(relation["target"]), f"{unmatched_prefix}:{relation['target']}"
        )
        left, right = sorted((source, target))
        normalized.add((left, right, str(relation["door_instance_id"])))
    return normalized


def _set_f1(first: set[object], second: set[object]) -> tuple[float, float, float]:
    if not first and not second:
        return 1.0, 1.0, 1.0
    true_positive = len(first & second)
    precision = true_positive / len(second) if second else 0.0
    recall = true_positive / len(first) if first else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def compare_room_gold_records(
    primary: Mapping[str, Any],
    secondary: Mapping[str, Any],
    *,
    image_size: tuple[int, int] = (980, 980),
    polygon_iou_threshold: float = 0.90,
    relation_f1_threshold: float = 0.95,
) -> dict[str, Any]:
    """Compare two independently reviewed records for the same image."""

    for name, record in (("primary", primary), ("secondary", secondary)):
        errors = validate_room_gold_record(record)
        if errors:
            raise ValueError(f"invalid {name} record: {errors}")
        if record.get("review_status") != "reviewed":
            raise ValueError(f"{name} record must be reviewed")
    if primary.get("image_id") != secondary.get("image_id"):
        raise ValueError("independent review records must have the same image_id")

    primary_annotation = primary["annotation"]
    secondary_annotation = secondary["annotation"]
    view_type_agreement = (
        primary_annotation.get("view_type") == secondary_annotation.get("view_type")
    )
    matches = match_regions(
        primary_annotation.get("regions", []),
        secondary_annotation.get("regions", []),
        image_size=image_size,
    )
    primary_mapping = {
        match.primary_id: f"matched:{index:04d}" for index, match in enumerate(matches)
    }
    secondary_mapping = {
        match.secondary_id: f"matched:{index:04d}" for index, match in enumerate(matches)
    }
    primary_relations = _normalized_relations(
        primary_annotation.get("relations", []),
        primary_mapping,
        unmatched_prefix="primary-unmatched",
    )
    secondary_relations = _normalized_relations(
        secondary_annotation.get("relations", []),
        secondary_mapping,
        unmatched_prefix="secondary-unmatched",
    )
    relation_precision, relation_recall, relation_f1 = _set_f1(
        primary_relations, secondary_relations
    )
    primary_region_count = len(primary_annotation.get("regions", []))
    secondary_region_count = len(secondary_annotation.get("regions", []))
    all_regions_matched = (
        len(matches) == primary_region_count == secondary_region_count
    )
    minimum_iou = min((match.iou for match in matches), default=1.0)
    mean_iou = (
        sum(match.iou for match in matches) / len(matches) if matches else 1.0
    )
    polygon_gate = all_regions_matched and minimum_iou >= polygon_iou_threshold
    relation_gate = relation_f1 >= relation_f1_threshold
    return {
        "image_id": primary["image_id"],
        "primary_reviewer": primary.get("reviewer"),
        "secondary_reviewer": secondary.get("reviewer"),
        "independent_reviewers": primary.get("reviewer") != secondary.get("reviewer"),
        "view_type_agreement": view_type_agreement,
        "primary_region_count": primary_region_count,
        "secondary_region_count": secondary_region_count,
        "matched_region_count": len(matches),
        "unmatched_primary_regions": primary_region_count - len(matches),
        "unmatched_secondary_regions": secondary_region_count - len(matches),
        "mean_polygon_iou": mean_iou,
        "minimum_polygon_iou": minimum_iou,
        "region_matches": [match.to_dict() for match in matches],
        "relation_precision": relation_precision,
        "relation_recall": relation_recall,
        "relation_f1": relation_f1,
        "polygon_gate_passed": polygon_gate,
        "relation_gate_passed": relation_gate,
        "agreement_gate_passed": (
            view_type_agreement
            and primary.get("reviewer") != secondary.get("reviewer")
            and polygon_gate
            and relation_gate
        ),
    }


def summarize_room_gold_agreement(
    primary_records: Sequence[Mapping[str, Any]],
    secondary_records: Sequence[Mapping[str, Any]],
    *,
    image_size: tuple[int, int] = (980, 980),
    polygon_iou_threshold: float = 0.90,
    relation_f1_threshold: float = 0.95,
) -> dict[str, Any]:
    """Summarize eligible paired reviews without counting unfinished records."""

    primary_by_id = {str(record.get("image_id")): record for record in primary_records}
    secondary_by_id = {str(record.get("image_id")): record for record in secondary_records}
    common_ids = sorted(primary_by_id.keys() & secondary_by_id.keys())
    paired = [
        image_id
        for image_id in common_ids
        if primary_by_id[image_id].get("review_status") == "reviewed"
        and secondary_by_id[image_id].get("review_status") == "reviewed"
    ]
    comparisons = [
        compare_room_gold_records(
            primary_by_id[image_id],
            secondary_by_id[image_id],
            image_size=image_size,
            polygon_iou_threshold=polygon_iou_threshold,
            relation_f1_threshold=relation_f1_threshold,
        )
        for image_id in paired
    ]
    return {
        "schema_version": "room-gold-agreement-v0",
        "primary_record_count": len(primary_records),
        "secondary_record_count": len(secondary_records),
        "common_record_count": len(common_ids),
        "eligible_review_pair_count": len(comparisons),
        "polygon_iou_threshold": polygon_iou_threshold,
        "relation_f1_threshold": relation_f1_threshold,
        "agreement_gate_pass_count": sum(
            comparison["agreement_gate_passed"] for comparison in comparisons
        ),
        "agreement_gate_pass_fraction": (
            sum(comparison["agreement_gate_passed"] for comparison in comparisons)
            / len(comparisons)
            if comparisons
            else 0.0
        ),
        "mean_polygon_iou": (
            sum(comparison["mean_polygon_iou"] for comparison in comparisons)
            / len(comparisons)
            if comparisons
            else None
        ),
        "mean_relation_f1": (
            sum(comparison["relation_f1"] for comparison in comparisons)
            / len(comparisons)
            if comparisons
            else None
        ),
        "comparisons": comparisons,
    }
