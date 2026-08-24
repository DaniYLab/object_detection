import json

import pytest
import torch

from src.evaluation.prediction_io import (
    PREDICTION_FORMAT,
    PREDICTION_SCHEMA_VERSION,
    make_prediction_payload,
    read_predictions,
    validate_prediction_payload,
    validate_predictions,
    write_predictions,
)


def test_prediction_json_round_trip_preserves_schema_and_tensors(tmp_path) -> None:
    predictions = [
        {
            "image_id": "floor-001",
            "boxes": torch.tensor([[1.5, 2.0, 10.0, 12.0]]),
            "scores": torch.tensor([0.875]),
            "labels": torch.tensor([6]),
        },
        {
            "image_id": 2,
            "boxes": torch.empty((0, 4)),
            "scores": torch.empty((0,)),
            "labels": torch.empty((0,), dtype=torch.long),
        },
    ]
    path = tmp_path / "nested" / "predictions.json"

    returned_path = write_predictions(path, predictions, metadata={"adapter": "yolo"})
    loaded = read_predictions(path)

    assert returned_path == path
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["format"] == PREDICTION_FORMAT
    assert raw["schema_version"] == PREDICTION_SCHEMA_VERSION
    assert raw["box_format"] == "xyxy"
    assert raw["coordinate_space"] == "image"
    assert raw["metadata"] == {"adapter": "yolo"}

    assert loaded[0]["image_id"] == "floor-001"
    assert torch.allclose(loaded[0]["boxes"], predictions[0]["boxes"])
    assert torch.allclose(loaded[0]["scores"], predictions[0]["scores"])
    assert torch.equal(loaded[0]["labels"], predictions[0]["labels"])
    assert loaded[1]["boxes"].shape == (0, 4)
    assert loaded[1]["labels"].dtype == torch.long


def test_write_predictions_accepts_object_first_compatibility_order(tmp_path) -> None:
    predictions = [
        {
            "image_id": 1,
            "boxes": [[0.0, 0.0, 1.0, 1.0]],
            "scores": [0.5],
            "labels": [0],
        }
    ]
    path = tmp_path / "predictions.json"

    write_predictions(predictions, path)

    assert read_predictions(path)[0]["image_id"] == 1


def test_validation_accepts_external_adapter_lists() -> None:
    predictions = [
        {
            "image_id": "image.png",
            "boxes": [[0, 1, 20, 30]],
            "scores": [0.99],
            "labels": [4],
        }
    ]

    assert validate_predictions(predictions) is None
    payload = make_prediction_payload(predictions)
    assert validate_prediction_payload(payload) is None


@pytest.mark.parametrize(
    "record, error",
    [
        (
            {"image_id": "a", "boxes": [[0, 0, 1, 1]], "scores": [], "labels": [0]},
            ValueError,
        ),
        (
            {"image_id": "a", "boxes": [[0, 0, 0, 1]], "scores": [0.5], "labels": [0]},
            ValueError,
        ),
        (
            {"image_id": "a", "boxes": [[0, 0, 1, 1]], "scores": [float("nan")], "labels": [0]},
            ValueError,
        ),
        (
            {"image_id": "a", "boxes": [[0, 0, 1, 1]], "scores": [1.1], "labels": [0]},
            ValueError,
        ),
        (
            {"image_id": "a", "boxes": [[0, 0, 1, 1]], "scores": [0.5], "labels": [0.0]},
            TypeError,
        ),
    ],
)
def test_prediction_validation_rejects_malformed_records(record, error) -> None:
    with pytest.raises(error):
        validate_predictions([record])


def test_prediction_validation_rejects_duplicate_image_ids() -> None:
    record = {
        "image_id": "same",
        "boxes": [],
        "scores": [],
        "labels": [],
    }
    with pytest.raises(ValueError, match="duplicate"):
        validate_predictions([record, record])


def test_read_rejects_incompatible_schema_version(tmp_path) -> None:
    path = tmp_path / "bad-version.json"
    path.write_text(
        json.dumps(
            {
                "format": PREDICTION_FORMAT,
                "schema_version": 99,
                "box_format": "xyxy",
                "coordinate_space": "image",
                "predictions": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema_version"):
        read_predictions(path)
