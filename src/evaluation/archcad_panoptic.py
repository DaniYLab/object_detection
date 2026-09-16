"""ArchCAD primitive-level semantic, instance, and panoptic benchmark metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.data.archcad_constants import (
    ARCHCAD_IGNORE_LABEL,
    ARCHCAD_RAW_SEMANTIC_NAMES,
    ARCHCAD_STUFF_SEMANTIC_IDS,
    ARCHCAD_THING_SEMANTIC_IDS,
    NUM_ARCHCAD_SEMANTIC_CLASSES,
)
from src.data.archcad_panoptic import ArchCADPanopticSample, ArchCADSegmentTarget


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_ratio(2.0 * precision * recall, precision + recall)


def _semantic_metrics(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    *,
    weights: np.ndarray,
) -> dict[str, Any]:
    valid = ground_truth != ARCHCAD_IGNORE_LABEL
    ground_truth, prediction, weights = ground_truth[valid], prediction[valid], weights[valid]
    class_records = []
    for semantic_id in range(NUM_ARCHCAD_SEMANTIC_CLASSES):
        gt_positive = ground_truth == semantic_id
        pred_positive = prediction == semantic_id
        true_positive = float(weights[np.logical_and(gt_positive, pred_positive)].sum())
        false_positive = float(weights[np.logical_and(~gt_positive, pred_positive)].sum())
        false_negative = float(weights[np.logical_and(gt_positive, ~pred_positive)].sum())
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        class_records.append(
            {
                "semantic_id": semantic_id,
                "semantic_name": ARCHCAD_RAW_SEMANTIC_NAMES[semantic_id],
                "support": float(weights[gt_positive].sum()),
                "precision": precision,
                "recall": recall,
                "f1": _f1(precision, recall),
            }
        )
    present = [record for record in class_records if record["support"] > 0]
    true_positive = sum(
        float(weights[np.logical_and(ground_truth == value, prediction == value)].sum())
        for value in range(NUM_ARCHCAD_SEMANTIC_CLASSES)
    )
    valid_weight = float(weights[ground_truth != ARCHCAD_IGNORE_LABEL].sum())
    predicted_weight = float(weights[prediction != ARCHCAD_IGNORE_LABEL].sum())
    micro = _safe_ratio(2 * true_positive, valid_weight + predicted_weight)
    return {
        "macro_f1": (
            sum(record["f1"] for record in present) / len(present) if present else 0.0
        ),
        "micro_f1": micro,
        "per_class": class_records,
    }


def _mask_iou(first: np.ndarray, second: np.ndarray, weights: np.ndarray) -> float:
    intersection = float(weights[np.logical_and(first, second)].sum())
    union = float(weights[np.logical_or(first, second)].sum())
    return _safe_ratio(intersection, union)


def _validated_prediction(
    sample: ArchCADPanopticSample, prediction: Mapping[str, Any]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if prediction.get("image_id") != sample.image_id:
        raise ValueError("prediction image_id must match ground truth")
    raw_semantic = np.asarray(prediction.get("semantic_labels"))
    if raw_semantic.dtype.kind not in 'iu' or np.any(
        (raw_semantic < 0) | (raw_semantic > ARCHCAD_IGNORE_LABEL)
    ):
        raise ValueError("semantic_labels must contain integer IDs 0..30")
    semantic = raw_semantic.astype(np.int64)
    if semantic.shape != sample.semantic_labels.shape:
        raise ValueError(
            f"semantic prediction must have shape {sample.semantic_labels.shape}, "
            f"got {semantic.shape}"
        )
    instances = []
    for index, instance in enumerate(prediction.get("instances", [])):
        if not isinstance(instance, Mapping):
            raise TypeError(f"instances[{index}] must be a mapping")
        mask = np.asarray(instance.get("mask"))
        if mask.dtype.kind not in 'biu' or not np.isin(mask, [0, 1]).all():
            raise ValueError(f"instances[{index}].mask must be binary")
        mask = mask.astype(bool)
        if mask.shape != sample.semantic_labels.shape:
            raise ValueError(f"instances[{index}].mask has invalid shape {mask.shape}")
        semantic_id = instance.get("semantic_id")
        if not isinstance(semantic_id, (int, np.integer)) or isinstance(semantic_id, bool):
            raise ValueError(f"instances[{index}].semantic_id must be an integer")
        semantic_id = int(semantic_id)
        score = float(instance.get("score"))
        if semantic_id not in range(NUM_ARCHCAD_SEMANTIC_CLASSES):
            raise ValueError(f"instances[{index}].semantic_id is out of range")
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"instances[{index}].score must be in [0, 1]")
        instances.append(
            {"semantic_id": semantic_id, "score": score, "mask": mask}
        )
    return semantic, instances


def _evaluation_weights(sample: ArchCADPanopticSample) -> np.ndarray:
    """Local protocol: void/ignore primitives contribute no matching weight."""
    return np.where(sample.semantic_labels != ARCHCAD_IGNORE_LABEL, sample.length_weights, 0.0)


def _greedy_segment_matches(
    ground_truth: Sequence[ArchCADSegmentTarget],
    predictions: Sequence[Mapping[str, Any]],
    weights: np.ndarray,
    *,
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    candidates = []
    for gt_index, target in enumerate(ground_truth):
        for pred_index, prediction in enumerate(predictions):
            if target.semantic_id != prediction["semantic_id"]:
                continue
            iou = _mask_iou(target.primitive_mask, prediction["mask"], weights)
            if iou > iou_threshold:
                candidates.append((iou, gt_index, pred_index))
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches = []
    for iou, gt_index, pred_index in sorted(candidates, reverse=True):
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
        matches.append((gt_index, pred_index, iou))
    return matches


def _panoptic_metrics(
    samples: Sequence[ArchCADPanopticSample],
    predictions: Sequence[Sequence[Mapping[str, Any]]],
    *,
    score_threshold: float,
    iou_threshold: float,
) -> dict[str, Any]:
    true_positive = np.zeros(NUM_ARCHCAD_SEMANTIC_CLASSES, dtype=np.float64)
    false_positive = np.zeros_like(true_positive)
    false_negative = np.zeros_like(true_positive)
    summed_iou = np.zeros_like(true_positive)
    for sample, sample_predictions in zip(samples, predictions):
        weights = _evaluation_weights(sample)
        kept = [
            prediction
            for prediction in sample_predictions
            if prediction["score"] >= score_threshold and weights[prediction["mask"]].sum() > 0
        ]
        targets = [segment for segment in sample.segments if weights[segment.primitive_mask].sum() > 0]
        matches = _greedy_segment_matches(
            targets,
            kept,
            weights,
            iou_threshold=iou_threshold,
        )
        matched_gt = {match[0] for match in matches}
        matched_pred = {match[1] for match in matches}
        for gt_index, _, iou in matches:
            semantic_id = targets[gt_index].semantic_id
            true_positive[semantic_id] += 1
            summed_iou[semantic_id] += iou
        for index, segment in enumerate(targets):
            if index not in matched_gt:
                false_negative[segment.semantic_id] += 1
        for index, prediction in enumerate(kept):
            if index not in matched_pred:
                false_positive[prediction["semantic_id"]] += 1

    recognition_quality = true_positive / np.maximum(
        true_positive + 0.5 * false_positive + 0.5 * false_negative, 1e-12
    )
    segmentation_quality = summed_iou / np.maximum(true_positive, 1e-12)
    panoptic_quality = recognition_quality * segmentation_quality

    def aggregate(ids: Iterable[int]) -> dict[str, float]:
        indices = np.asarray(list(ids), dtype=np.int64)
        tp = float(true_positive[indices].sum())
        fp = float(false_positive[indices].sum())
        fn = float(false_negative[indices].sum())
        iou = float(summed_iou[indices].sum())
        rq = _safe_ratio(tp, tp + 0.5 * fp + 0.5 * fn)
        sq = _safe_ratio(iou, tp)
        return {"pq": rq * sq, "sq": sq, "rq": rq, "tp": tp, "fp": fp, "fn": fn}

    return {
        "all": aggregate(range(NUM_ARCHCAD_SEMANTIC_CLASSES)),
        "thing": aggregate(ARCHCAD_THING_SEMANTIC_IDS),
        "stuff": aggregate(ARCHCAD_STUFF_SEMANTIC_IDS),
        "per_class": [
            {
                "semantic_id": semantic_id,
                "semantic_name": ARCHCAD_RAW_SEMANTIC_NAMES[semantic_id],
                "pq": float(panoptic_quality[semantic_id]),
                "sq": float(segmentation_quality[semantic_id]),
                "rq": float(recognition_quality[semantic_id]),
                "tp": float(true_positive[semantic_id]),
                "fp": float(false_positive[semantic_id]),
                "fn": float(false_negative[semantic_id]),
            }
            for semantic_id in range(NUM_ARCHCAD_SEMANTIC_CLASSES)
        ],
    }


def _interpolated_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    return float(
        np.mean(
            [
                precisions[recalls >= threshold].max()
                if np.any(recalls >= threshold)
                else 0.0
                for threshold in np.linspace(0.0, 1.0, 101)
            ]
        )
    )


def _instance_ap_at_iou(
    samples: Sequence[ArchCADPanopticSample],
    predictions: Sequence[Sequence[Mapping[str, Any]]],
    *,
    semantic_id: int,
    iou_threshold: float,
) -> float | None:
    ground_truth = {
        sample.image_id: [
            segment
            for segment in sample.segments
            if segment.semantic_id == semantic_id
            and _evaluation_weights(sample)[segment.primitive_mask].sum() > 0
        ]
        for sample in samples
    }
    gt_count = sum(len(value) for value in ground_truth.values())
    if not gt_count:
        return None
    weights = {sample.image_id: _evaluation_weights(sample) for sample in samples}
    ranked = sorted(
        [
            (sample.image_id, prediction)
            for sample, sample_predictions in zip(samples, predictions)
            for prediction in sample_predictions
            if prediction["semantic_id"] == semantic_id
            and weights[sample.image_id][prediction["mask"]].sum() > 0
        ],
        key=lambda value: value[1]["score"],
        reverse=True,
    )
    matched: dict[str, set[int]] = defaultdict(set)
    true_positives = []
    false_positives = []
    for image_id, prediction in ranked:
        candidates = [
            (
                _mask_iou(target.primitive_mask, prediction["mask"], weights[image_id]),
                index,
            )
            for index, target in enumerate(ground_truth[image_id])
            if index not in matched[image_id]
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        is_match = best_iou >= iou_threshold
        if is_match:
            matched[image_id].add(best_index)
        true_positives.append(float(is_match))
        false_positives.append(float(not is_match))
    if not ranked:
        return 0.0
    cumulative_tp = np.cumsum(true_positives)
    cumulative_fp = np.cumsum(false_positives)
    recalls = cumulative_tp / gt_count
    precisions = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
    return _interpolated_ap(recalls, precisions)


def evaluate_archcad_panoptic(
    samples: Sequence[ArchCADPanopticSample],
    predictions: Sequence[Mapping[str, Any]],
    *,
    panoptic_score_threshold: float = 0.10,
    panoptic_iou_threshold: float = 0.50,
) -> dict[str, Any]:
    """Evaluate the three ArchCAD benchmark tracks over aligned primitives."""

    if len(samples) != len(predictions):
        raise ValueError("samples and predictions must have equal length")
    if not samples:
        raise ValueError("at least one sample is required")
    if len({sample.image_id for sample in samples}) != len(samples):
        raise ValueError("duplicate sample image_id")
    if not 0 <= panoptic_score_threshold <= 1 or not 0.5 <= panoptic_iou_threshold < 1:
        raise ValueError("invalid panoptic thresholds")
    validated = [
        _validated_prediction(sample, prediction)
        for sample, prediction in zip(samples, predictions)
    ]
    semantic_predictions = [value[0] for value in validated]
    instance_predictions = [value[1] for value in validated]
    gt_semantic = np.concatenate([sample.semantic_labels for sample in samples])
    pred_semantic = np.concatenate(semantic_predictions)
    unweighted = _semantic_metrics(
        gt_semantic,
        pred_semantic,
        weights=np.ones_like(gt_semantic, dtype=np.float64),
    )
    length_weighted = _semantic_metrics(
        gt_semantic,
        pred_semantic,
        weights=np.concatenate([sample.length_weights for sample in samples]),
    )
    thresholds = np.linspace(0.50, 0.95, 10)
    instance_per_class = []
    for semantic_id in ARCHCAD_THING_SEMANTIC_IDS:
        values = [
            _instance_ap_at_iou(
                samples,
                instance_predictions,
                semantic_id=semantic_id,
                iou_threshold=float(threshold),
            )
            for threshold in thresholds
        ]
        if all(value is None for value in values):
            continue
        numeric = [float(value) for value in values if value is not None]
        instance_per_class.append(
            {
                "semantic_id": semantic_id,
                "semantic_name": ARCHCAD_RAW_SEMANTIC_NAMES[semantic_id],
                "ap50": numeric[0],
                "ap75": numeric[5],
                "map_50_95": sum(numeric) / len(numeric),
            }
        )
    return {
        "schema_version": "archcad-panoptic-benchmark-v1",
        "protocol_status": "local_reconstruction_not_verified_paper_parity",
        "sample_count": len(samples),
        "primitive_count": sum(sample.primitive_count for sample in samples),
        "semantic": {
            "f1_macro": unweighted["macro_f1"],
            "f1_micro": unweighted["micro_f1"],
            "weighted_f1_macro": length_weighted["macro_f1"],
            "weighted_f1_micro": length_weighted["micro_f1"],
            "per_class": unweighted["per_class"],
            "weighted_per_class": length_weighted["per_class"],
        },
        "instance": {
            "ap50": (
                sum(record["ap50"] for record in instance_per_class)
                / len(instance_per_class)
                if instance_per_class
                else 0.0
            ),
            "ap75": (
                sum(record["ap75"] for record in instance_per_class)
                / len(instance_per_class)
                if instance_per_class
                else 0.0
            ),
            "map_50_95": (
                sum(record["map_50_95"] for record in instance_per_class)
                / len(instance_per_class)
                if instance_per_class
                else 0.0
            ),
            "per_class": instance_per_class,
        },
        "panoptic": _panoptic_metrics(
            samples,
            instance_predictions,
            score_threshold=panoptic_score_threshold,
            iou_threshold=panoptic_iou_threshold,
        ),
        "parameters": {
            "panoptic_aggregation": "pooled_counts; not class-macro PQ",
            "matching_policy": "one_to_one; differs from released upstream evaluator",
            "ignore_policy": "exclude GT ignore positions from all metrics",
            "zero_weight_policy": "exclude masks with no positive evaluation weight",
            "ap_interpolation": "101-point; paper parity unverified",
            "length_weight": "round(log(1 + primitive_length), 3)",
            "panoptic_score_threshold": panoptic_score_threshold,
            "panoptic_iou_threshold": panoptic_iou_threshold,
            "thing_semantic_ids": list(ARCHCAD_THING_SEMANTIC_IDS),
            "stuff_semantic_ids": list(ARCHCAD_STUFF_SEMANTIC_IDS),
            "ignore_label": ARCHCAD_IGNORE_LABEL,
        },
        "unscorable": {
            "gt_segments": int(sum(
                _evaluation_weights(s)[seg.primitive_mask].sum() <= 0
                for s in samples for seg in s.segments
            )),
            "prediction_segments": int(sum(
                _evaluation_weights(s)[p['mask']].sum() <= 0
                for s, ps in zip(samples, instance_predictions) for p in ps
            )),
        },
    }
