"""Dependency-free detection IoU and COCO-style AP metrics.

This module intentionally uses only Python and PyTorch.  It implements the
COCO IoU threshold sweep and 101-point interpolated AP, but not COCO's area
ranges, crowd handling, or max-detections variants.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import torch


COCO_IOU_THRESHOLDS: tuple[float, ...] = tuple(
    round(0.50 + 0.05 * index, 2) for index in range(10)
)


def _box_tensor(boxes: Any, *, device: torch.device | None = None) -> torch.Tensor:
    if isinstance(boxes, torch.Tensor):
        tensor = boxes.detach()
        if device is not None:
            tensor = tensor.to(device=device)
        if not tensor.is_floating_point():
            tensor = tensor.to(dtype=torch.float32)
    else:
        tensor = torch.as_tensor(boxes, dtype=torch.float32, device=device)
    if tensor.numel() == 0:
        return tensor.reshape(0, 4)
    if tensor.ndim == 1 and tensor.shape[0] == 4:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[1] != 4:
        raise ValueError(f"boxes must have shape [N,4], got {tuple(tensor.shape)}")
    return tensor


def box_iou(boxes1: Any, boxes2: Any) -> torch.Tensor:
    """Return pairwise IoU for ``xyxy`` boxes as an ``[N,M]`` tensor.

    Degenerate or non-finite pairs receive IoU zero rather than producing NaN.
    Coordinates use continuous geometry (there is no inclusive-pixel ``+1``).
    """
    first = _box_tensor(boxes1)
    second = _box_tensor(boxes2, device=first.device)
    dtype = torch.promote_types(first.dtype, second.dtype)
    if not dtype.is_floating_point:
        dtype = torch.float32
    first = first.to(dtype=dtype)
    second = second.to(dtype=dtype)

    if first.shape[0] == 0 or second.shape[0] == 0:
        return torch.zeros(
            (first.shape[0], second.shape[0]),
            dtype=dtype,
            device=first.device,
        )

    top_left = torch.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = torch.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection_wh = (bottom_right - top_left).clamp(min=0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]

    first_wh = (first[:, 2:] - first[:, :2]).clamp(min=0)
    second_wh = (second[:, 2:] - second[:, :2]).clamp(min=0)
    first_area = first_wh[:, 0] * first_wh[:, 1]
    second_area = second_wh[:, 0] * second_wh[:, 1]
    union = first_area[:, None] + second_area[None, :] - intersection

    valid = torch.isfinite(intersection) & torch.isfinite(union) & union.gt(0)
    return torch.where(valid, intersection / union.clamp(min=torch.finfo(dtype).tiny), 0)


def average_precision_101(recall: Any, precision: Any) -> float:
    """Compute 101-point interpolated AP at recall levels 0.00 through 1.00."""
    recall_tensor = torch.as_tensor(recall, dtype=torch.float64).reshape(-1)
    precision_tensor = torch.as_tensor(precision, dtype=torch.float64).reshape(-1)
    if recall_tensor.numel() != precision_tensor.numel():
        raise ValueError("recall and precision must have equal lengths")
    if recall_tensor.numel() == 0:
        return 0.0
    if not bool(torch.isfinite(recall_tensor).all()) or not bool(
        torch.isfinite(precision_tensor).all()
    ):
        raise ValueError("recall and precision must be finite")

    # Sorting makes this helper useful independently; matching already produces
    # monotonic recall.  Stable sorting preserves precision order at equal recall.
    try:
        order = torch.argsort(recall_tensor, stable=True)
    except TypeError:  # pragma: no cover - compatibility with old PyTorch
        order = torch.argsort(recall_tensor)
    recall_tensor = recall_tensor[order]
    precision_tensor = precision_tensor[order]

    recall_levels = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
    interpolated = torch.zeros_like(recall_levels)
    for index, level in enumerate(recall_levels):
        eligible = precision_tensor[recall_tensor >= level]
        if eligible.numel() > 0:
            interpolated[index] = eligible.max()
    return float(interpolated.mean().item())


def _label_tensor(labels: Any, expected_length: int, field: str) -> torch.Tensor:
    if isinstance(labels, torch.Tensor):
        raw = labels.detach().cpu().reshape(-1)
    else:
        raw = torch.as_tensor(labels).reshape(-1)
    if raw.numel() != expected_length:
        raise ValueError(
            f"{field} has length {raw.numel()}, expected {expected_length}"
        )
    if raw.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    if raw.dtype == torch.bool:
        raise TypeError(f"{field} must contain integer class IDs")
    numeric = raw.to(dtype=torch.float64)
    if not bool(torch.isfinite(numeric).all()) or not bool(numeric.eq(numeric.round()).all()):
        raise ValueError(f"{field} must contain finite integer class IDs")
    if bool(numeric.lt(0).any()):
        raise ValueError(f"{field} must contain non-negative class IDs")
    return numeric.to(dtype=torch.long)


def _normalise_records(
    records: Sequence[Mapping[str, Any]],
    *,
    predictions: bool,
) -> list[dict[str, Any]]:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        kind = "predictions" if predictions else "targets"
        raise TypeError(f"{kind} must be a sequence of per-image mappings")

    normalised: list[dict[str, Any]] = []
    seen_ids: set[Any] = set()
    for image_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"record {image_index} must be a mapping")
        for field in ("image_id", "boxes", "labels"):
            if field not in record:
                raise KeyError(f"record {image_index} is missing {field!r}")
        image_id = record["image_id"]
        try:
            hash(image_id)
        except TypeError as error:
            raise TypeError(f"record {image_index} image_id must be hashable") from error
        if image_id in seen_ids:
            raise ValueError(f"duplicate image_id {image_id!r}")
        seen_ids.add(image_id)

        boxes = _box_tensor(record["boxes"]).to(device="cpu", dtype=torch.float64)
        labels = _label_tensor(record["labels"], boxes.shape[0], f"record {image_index} labels")
        item: dict[str, Any] = {
            "image_id": image_id,
            "boxes": boxes,
            "labels": labels,
            "image_index": image_index,
        }
        if predictions:
            if "scores" not in record:
                raise KeyError(f"record {image_index} is missing 'scores'")
            scores = torch.as_tensor(record["scores"], dtype=torch.float64).reshape(-1)
            if scores.numel() != boxes.shape[0]:
                raise ValueError(
                    f"record {image_index} scores has length {scores.numel()}, "
                    f"expected {boxes.shape[0]}"
                )
            if not bool(torch.isfinite(scores).all()):
                raise ValueError(f"record {image_index} scores must be finite")
            item["scores"] = scores
        normalised.append(item)
    return normalised


def _normalise_thresholds(iou_thresholds: Sequence[float] | None) -> tuple[float, ...]:
    if iou_thresholds is None:
        return COCO_IOU_THRESHOLDS
    if not isinstance(iou_thresholds, Sequence) or isinstance(iou_thresholds, (str, bytes)):
        raise TypeError("iou_thresholds must be a sequence")
    values: list[float] = []
    for threshold in iou_thresholds:
        if not isinstance(threshold, Real) or isinstance(threshold, bool):
            raise TypeError("IoU thresholds must be numbers")
        value = float(threshold)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("IoU thresholds must be finite and in [0,1]")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("iou_thresholds cannot be empty")
    return tuple(values)


def _image_id_sort_key(image_id: Any) -> tuple[int, Any]:
    """Return a deterministic ordering key for supported image identifiers."""
    if isinstance(image_id, Integral) and not isinstance(image_id, bool):
        return 0, int(image_id)
    if isinstance(image_id, str):
        return 1, image_id
    return 2, f"{type(image_id).__module__}.{type(image_id).__qualname__}:{image_id!r}"


def _ap_for_class(
    predictions: list[tuple[float, tuple[int, Any], int, Any, torch.Tensor]],
    ground_truth_by_image: Mapping[Any, torch.Tensor],
    num_ground_truths: int,
    iou_threshold: float,
) -> float:
    if num_ground_truths == 0 or not predictions:
        return 0.0

    # Python's sort is stable. The complete key makes tie behavior explicit:
    # descending score, canonical image ID, then prediction index.
    ordered = sorted(predictions, key=lambda item: (-item[0], item[1], item[2]))
    matched = {
        image_id: torch.zeros(boxes.shape[0], dtype=torch.bool)
        for image_id, boxes in ground_truth_by_image.items()
    }
    true_positives = torch.zeros(len(ordered), dtype=torch.float64)
    false_positives = torch.zeros(len(ordered), dtype=torch.float64)

    for detection_index, (_, _, _, image_id, predicted_box) in enumerate(ordered):
        ground_truth = ground_truth_by_image.get(image_id)
        if ground_truth is None or ground_truth.shape[0] == 0:
            false_positives[detection_index] = 1.0
            continue

        ious = box_iou(predicted_box.unsqueeze(0), ground_truth)[0].to(device="cpu")
        available = ~matched[image_id]
        ious = torch.where(available, ious, torch.full_like(ious, -1.0))
        best_iou, best_index = ious.max(dim=0)
        if float(best_iou.item()) >= iou_threshold:
            true_positives[detection_index] = 1.0
            matched[image_id][int(best_index.item())] = True
        else:
            false_positives[detection_index] = 1.0

    cumulative_tp = true_positives.cumsum(dim=0)
    cumulative_fp = false_positives.cumsum(dim=0)
    recall = cumulative_tp / float(num_ground_truths)
    precision = cumulative_tp / (cumulative_tp + cumulative_fp).clamp(min=1.0)
    return average_precision_101(recall, precision)


def compute_coco_metrics(
    predictions: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    *,
    iou_thresholds: Sequence[float] | None = None,
    class_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Compute class-macro AP50 and AP50:95 for per-image detections.

    Matching is greedy and one-to-one within each image and class. Predictions
    are sorted deterministically by descending score, canonical image ID, and
    prediction index. Macro averages include only classes that have ground
    truth; prediction-only classes are still reported in ``per_class``.

    Empty inputs are explicit: all aggregate AP values are ``0.0`` when there
    are no ground-truth classes, and a GT class with no predictions has AP zero.
    """
    pred_records = _normalise_records(predictions, predictions=True)
    target_records = _normalise_records(targets, predictions=False)
    thresholds = _normalise_thresholds(iou_thresholds)

    ground_truths: dict[int, dict[Any, list[torch.Tensor]]] = defaultdict(
        lambda: defaultdict(list)
    )
    gt_counts: Counter[int] = Counter()
    for record in target_records:
        for box, label_tensor in zip(record["boxes"], record["labels"]):
            label = int(label_tensor.item())
            ground_truths[label][record["image_id"]].append(box)
            gt_counts[label] += 1

    detections: dict[
        int, list[tuple[float, tuple[int, Any], int, Any, torch.Tensor]]
    ] = defaultdict(list)
    prediction_counts: Counter[int] = Counter()
    for record in pred_records:
        for prediction_index, (box, score_tensor, label_tensor) in enumerate(
            zip(record["boxes"], record["scores"], record["labels"])
        ):
            label = int(label_tensor.item())
            score = float(score_tensor.item())
            detections[label].append(
                (
                    score,
                    _image_id_sort_key(record["image_id"]),
                    prediction_index,
                    record["image_id"],
                    box,
                )
            )
            prediction_counts[label] += 1

    classes = set(gt_counts) | set(prediction_counts)
    if class_ids is not None:
        for class_id in class_ids:
            if not isinstance(class_id, Integral) or isinstance(class_id, bool):
                raise TypeError("class_ids must contain integers")
            if int(class_id) < 0:
                raise ValueError("class_ids must be non-negative")
            classes.add(int(class_id))
    ordered_classes = sorted(classes)

    per_class: dict[int, dict[str, Any]] = {}
    for class_id in ordered_classes:
        gt_by_image = {
            image_id: torch.stack(boxes, dim=0)
            for image_id, boxes in ground_truths[class_id].items()
        }
        count_gt = int(gt_counts[class_id])
        count_predictions = int(prediction_counts[class_id])
        ap_by_iou = {
            threshold: _ap_for_class(
                detections[class_id],
                gt_by_image,
                count_gt,
                threshold,
            )
            for threshold in thresholds
        }
        ap50 = ap_by_iou.get(0.5)
        if ap50 is None:
            ap50 = _ap_for_class(detections[class_id], gt_by_image, count_gt, 0.5)
        mean_ap = sum(ap_by_iou.values()) / len(ap_by_iou)
        per_class[class_id] = {
            "num_gt": count_gt,
            "num_predictions": count_predictions,
            "gt_count": count_gt,
            "prediction_count": count_predictions,
            "AP": mean_ap,
            "AP50": ap50,
            "AP50:95": mean_ap,
            "ap_by_iou": ap_by_iou,
            "included_in_macro": count_gt > 0,
        }

    gt_classes = [class_id for class_id in ordered_classes if gt_counts[class_id] > 0]
    if gt_classes:
        macro_ap = sum(per_class[class_id]["AP"] for class_id in gt_classes) / len(gt_classes)
        macro_ap50 = sum(per_class[class_id]["AP50"] for class_id in gt_classes) / len(gt_classes)
    else:
        macro_ap = 0.0
        macro_ap50 = 0.0

    image_ids = {record["image_id"] for record in pred_records}
    image_ids.update(record["image_id"] for record in target_records)
    return {
        "AP": macro_ap,
        "AP50": macro_ap50,
        "AP50:95": macro_ap,
        "mAP": macro_ap,
        "mAP50": macro_ap50,
        "mAP50:95": macro_ap,
        "map": macro_ap,
        "map_50": macro_ap50,
        "map_50_95": macro_ap,
        "iou_thresholds": thresholds,
        "num_images": len(image_ids),
        "num_gt": sum(gt_counts.values()),
        "num_predictions": sum(prediction_counts.values()),
        "num_classes_with_gt": len(gt_classes),
        "classes_with_gt": gt_classes,
        "per_class": per_class,
    }


# Public aliases with common evaluation terminology.
compute_average_precision = average_precision_101
evaluate_detections = compute_coco_metrics
evaluate_predictions = compute_coco_metrics
pairwise_iou = box_iou


__all__ = [
    "COCO_IOU_THRESHOLDS",
    "average_precision_101",
    "box_iou",
    "compute_average_precision",
    "compute_coco_metrics",
    "evaluate_detections",
    "evaluate_predictions",
    "pairwise_iou",
]
