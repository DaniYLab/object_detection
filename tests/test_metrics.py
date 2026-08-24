import pytest
import torch

from src.evaluation.metrics import (
    average_precision_101,
    box_iou,
    compute_coco_metrics,
)


def _record(image_id, boxes, labels, scores=None):
    record = {
        "image_id": image_id,
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    if scores is not None:
        record["scores"] = torch.tensor(scores, dtype=torch.float32)
    return record


def test_box_iou_handles_known_values_and_empty_inputs() -> None:
    first = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    second = torch.tensor(
        [
            [0.0, 0.0, 2.0, 2.0],
            [1.0, 1.0, 3.0, 3.0],
            [3.0, 3.0, 4.0, 4.0],
        ]
    )

    ious = box_iou(first, second)

    assert torch.allclose(ious, torch.tensor([[1.0, 1.0 / 7.0, 0.0]]))
    assert box_iou(torch.empty((0, 4)), second).shape == (0, 3)
    assert box_iou(first, torch.empty((0, 4))).shape == (1, 0)


def test_average_precision_uses_101_point_interpolation() -> None:
    # One false positive before the only true positive gives precision 0.5 at
    # every interpolated recall level.
    assert average_precision_101([0.0, 1.0], [0.0, 0.5]) == pytest.approx(0.5)
    assert average_precision_101([], []) == 0.0


def test_perfect_prediction_has_unit_coco_metrics() -> None:
    predictions = [_record("a", [[1, 1, 5, 5]], [2], [0.8])]
    targets = [_record("a", [[1, 1, 5, 5]], [2])]

    metrics = compute_coco_metrics(predictions, targets)

    assert metrics["AP"] == pytest.approx(1.0)
    assert metrics["AP50"] == pytest.approx(1.0)
    assert metrics["AP50:95"] == pytest.approx(1.0)
    assert metrics["per_class"][2]["num_gt"] == 1
    assert metrics["per_class"][2]["num_predictions"] == 1


def test_matching_order_is_stable_by_score_then_image_and_prediction_index() -> None:
    predictions = [
        _record("no-gt", [[0, 0, 4, 4]], [0], [0.5]),
        _record("with-gt", [[0, 0, 4, 4]], [0], [0.5]),
    ]
    targets = [_record("with-gt", [[0, 0, 4, 4]], [0])]

    metrics = compute_coco_metrics(predictions, targets, iou_thresholds=[0.5])
    reversed_metrics = compute_coco_metrics(
        list(reversed(predictions)), targets, iou_thresholds=[0.5]
    )

    # Equal scores use canonical image ID ordering, independent of record order.
    assert metrics["AP50"] == pytest.approx(0.5)
    assert reversed_metrics["AP50"] == pytest.approx(metrics["AP50"])


def test_iou_sweep_and_one_to_one_class_matching() -> None:
    predictions = [
        _record(
            "a",
            [
                [0, 0, 10, 6],  # IoU 0.60 with class-0 GT
                [20, 20, 30, 30],  # prediction-only class
            ],
            [0, 2],
            [0.9, 0.99],
        )
    ]
    targets = [
        _record(
            "a",
            [
                [0, 0, 10, 10],
                [40, 40, 50, 50],  # class 1 is missed
            ],
            [0, 1],
        )
    ]

    metrics = compute_coco_metrics(predictions, targets)

    assert metrics["per_class"][0]["AP50"] == pytest.approx(1.0)
    assert metrics["per_class"][0]["AP50:95"] == pytest.approx(0.3)
    assert metrics["per_class"][1]["AP50"] == 0.0
    assert metrics["per_class"][2]["num_gt"] == 0
    assert metrics["per_class"][2]["num_predictions"] == 1
    assert metrics["per_class"][2]["included_in_macro"] is False

    # Macro averages include classes 0 and 1 only, never prediction-only class 2.
    assert metrics["AP50"] == pytest.approx(0.5)
    assert metrics["AP50:95"] == pytest.approx(0.15)
    assert metrics["num_classes_with_gt"] == 2


def test_duplicate_detections_match_a_ground_truth_only_once() -> None:
    predictions = [
        _record(
            1,
            [[0, 0, 4, 4], [0, 0, 4, 4]],
            [3, 3],
            [0.9, 0.8],
        )
    ]
    targets = [_record(1, [[0, 0, 4, 4]], [3])]

    metrics = compute_coco_metrics(predictions, targets, iou_thresholds=[0.5])

    # The later duplicate is a false positive, though it cannot reduce the
    # precision envelope after full recall has already been reached.
    assert metrics["AP50"] == pytest.approx(1.0)
    assert metrics["per_class"][3]["num_predictions"] == 2


def test_empty_inputs_have_explicit_zero_metrics() -> None:
    empty = compute_coco_metrics([], [])
    assert empty["AP"] == 0.0
    assert empty["AP50"] == 0.0
    assert empty["num_gt"] == 0
    assert empty["num_predictions"] == 0
    assert empty["per_class"] == {}

    no_predictions = compute_coco_metrics(
        [],
        [_record("a", [[0, 0, 1, 1]], [4])],
    )
    assert no_predictions["AP"] == 0.0
    assert no_predictions["per_class"][4]["num_gt"] == 1
    assert no_predictions["per_class"][4]["num_predictions"] == 0

    prediction_only = compute_coco_metrics(
        [_record("a", [[0, 0, 1, 1]], [4], [0.5])],
        [],
    )
    assert prediction_only["AP"] == 0.0
    assert prediction_only["num_classes_with_gt"] == 0
    assert prediction_only["per_class"][4]["included_in_macro"] is False
