"""Versioned JSON interchange for per-image object-detection predictions.

The on-disk schema is intentionally model-agnostic so predictions from this
CenterNet model, YOLO, or Faster R-CNN can be evaluated through the same API::

    {
      "format": "object_detection_predictions",
      "schema_version": 1,
      "box_format": "xyxy",
      "coordinate_space": "image",
      "predictions": [
        {
          "image_id": "sample-1",
          "boxes": [[x1, y1, x2, y2]],
          "scores": [0.9],
          "labels": [3]
        }
      ]
    }

``image_id`` may be a string or integer.  Boxes are finite, positive-area
input-image coordinates, scores are finite probabilities in ``[0, 1]``, and
labels are non-negative integers.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import torch


PREDICTION_FORMAT = "object_detection_predictions"
PREDICTION_SCHEMA_VERSION = 1
PREDICTION_BOX_FORMAT = "xyxy"
PREDICTION_COORDINATE_SPACE = "image"


def _records_from_value(value: Any) -> Sequence[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        _validate_payload_header(value)
        return value["predictions"]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("predictions must be a sequence of per-image mappings")
    return value


def _number_list(value: Any, field: str) -> list[float]:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.ndim != 1:
            raise ValueError(f"{field} must be one-dimensional")
        values = tensor.tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
    else:
        raise TypeError(f"{field} must be a one-dimensional sequence")

    result: list[float] = []
    for index, item in enumerate(values):
        if not isinstance(item, Real) or isinstance(item, bool):
            raise TypeError(f"{field}[{index}] must be a number")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{field}[{index}] must be finite")
        result.append(number)
    return result


def _boxes_list(value: Any, field: str) -> list[list[float]]:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.numel() == 0:
            if tensor.ndim not in {1, 2} or (tensor.ndim == 2 and tensor.shape[1] != 4):
                raise ValueError(f"{field} must have shape [N,4]")
            rows: list[Any] = []
        else:
            if tensor.ndim != 2 or tensor.shape[1] != 4:
                raise ValueError(f"{field} must have shape [N,4]")
            rows = tensor.tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        rows = list(value)
    else:
        raise TypeError(f"{field} must be a sequence of xyxy boxes")

    boxes: list[list[float]] = []
    for box_index, row in enumerate(rows):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != 4:
            raise ValueError(f"{field}[{box_index}] must contain four xyxy values")
        box: list[float] = []
        for coordinate_index, item in enumerate(row):
            if not isinstance(item, Real) or isinstance(item, bool):
                raise TypeError(
                    f"{field}[{box_index}][{coordinate_index}] must be a number"
                )
            number = float(item)
            if not math.isfinite(number):
                raise ValueError(f"{field}[{box_index}] coordinates must be finite")
            box.append(number)
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"{field}[{box_index}] must have positive width and height")
        boxes.append(box)
    return boxes


def _labels_list(value: Any, field: str) -> list[int]:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.ndim != 1:
            raise ValueError(f"{field} must be one-dimensional")
        if tensor.dtype == torch.bool or tensor.is_floating_point() or tensor.is_complex():
            raise TypeError(f"{field} must contain integer class IDs")
        values = tensor.tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
    else:
        raise TypeError(f"{field} must be a one-dimensional sequence")

    labels: list[int] = []
    for index, item in enumerate(values):
        if not isinstance(item, Integral) or isinstance(item, bool):
            raise TypeError(f"{field}[{index}] must be an integer")
        label = int(item)
        if label < 0:
            raise ValueError(f"{field}[{index}] must be non-negative")
        labels.append(label)
    return labels


def _normalise_record(record: Mapping[str, Any], record_index: int) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise TypeError(f"prediction record {record_index} must be a mapping")
    missing = {"image_id", "boxes", "scores", "labels"}.difference(record)
    if missing:
        raise KeyError(f"prediction record {record_index} is missing {sorted(missing)}")

    image_id = record["image_id"]
    if isinstance(image_id, bool) or not isinstance(image_id, (str, Integral)):
        raise TypeError(
            f"prediction record {record_index} image_id must be a string or integer"
        )
    if isinstance(image_id, Integral):
        image_id = int(image_id)

    boxes = _boxes_list(record["boxes"], f"predictions[{record_index}].boxes")
    scores = _number_list(record["scores"], f"predictions[{record_index}].scores")
    labels = _labels_list(record["labels"], f"predictions[{record_index}].labels")
    if len(scores) != len(boxes) or len(labels) != len(boxes):
        raise ValueError(
            f"prediction record {record_index} boxes, scores, and labels must have equal lengths"
        )
    for score_index, score in enumerate(scores):
        if not 0.0 <= score <= 1.0:
            raise ValueError(
                f"predictions[{record_index}].scores[{score_index}] must be in [0,1]"
            )

    return {
        "image_id": image_id,
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
    }


def _normalise_predictions(value: Any) -> list[dict[str, Any]]:
    records = _records_from_value(value)
    normalised: list[dict[str, Any]] = []
    seen_image_ids: set[str | int] = set()
    for record_index, record in enumerate(records):
        item = _normalise_record(record, record_index)
        image_id = item["image_id"]
        if image_id in seen_image_ids:
            raise ValueError(f"duplicate prediction image_id {image_id!r}")
        seen_image_ids.add(image_id)
        normalised.append(item)
    return normalised


def validate_predictions(predictions: Any) -> None:
    """Validate per-image records or a complete prediction payload.

    The function returns ``None`` and raises ``TypeError``, ``ValueError``, or
    ``KeyError`` with a field-specific message on invalid input.
    """
    _normalise_predictions(predictions)


def _validate_payload_header(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise TypeError("prediction payload must be a mapping")
    required = {
        "format",
        "schema_version",
        "box_format",
        "coordinate_space",
        "predictions",
    }
    missing = required.difference(payload)
    if missing:
        raise KeyError(f"prediction payload is missing {sorted(missing)}")
    if payload["format"] != PREDICTION_FORMAT:
        raise ValueError(f"format must be {PREDICTION_FORMAT!r}")
    if payload["schema_version"] != PREDICTION_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {PREDICTION_SCHEMA_VERSION}, "
            f"got {payload['schema_version']!r}"
        )
    if payload["box_format"] != PREDICTION_BOX_FORMAT:
        raise ValueError(f"box_format must be {PREDICTION_BOX_FORMAT!r}")
    if payload["coordinate_space"] != PREDICTION_COORDINATE_SPACE:
        raise ValueError(f"coordinate_space must be {PREDICTION_COORDINATE_SPACE!r}")
    predictions = payload["predictions"]
    if not isinstance(predictions, Sequence) or isinstance(predictions, (str, bytes)):
        raise TypeError("payload predictions must be a sequence")


def validate_prediction_payload(payload: Mapping[str, Any]) -> None:
    """Validate a complete payload, including every prediction record."""
    _validate_payload_header(payload)
    _normalise_predictions(payload["predictions"])


def make_prediction_payload(
    predictions: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a validated, JSON-serializable prediction payload."""
    records = _normalise_predictions(predictions)
    payload: dict[str, Any] = {
        "format": PREDICTION_FORMAT,
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "box_format": PREDICTION_BOX_FORMAT,
        "coordinate_space": PREDICTION_COORDINATE_SPACE,
        "predictions": records,
    }
    if metadata is not None:
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        payload["metadata"] = dict(metadata)
    # Strict JSON encoding also validates optional metadata and disallows NaN.
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("prediction payload metadata must be finite and JSON-serializable") from error
    return payload


def write_predictions(
    path: str | Path | Any,
    predictions: Any = None,
    *,
    metadata: Mapping[str, Any] | None = None,
    indent: int | None = 2,
) -> Path:
    """Write predictions to the versioned JSON schema and return the path.

    The canonical call is ``write_predictions(path, predictions)``.  For
    compatibility with object-first save APIs, ``write_predictions(predictions,
    path)`` is also accepted.
    """
    if not isinstance(path, (str, Path)) and isinstance(predictions, (str, Path)):
        path, predictions = predictions, path
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a string or pathlib.Path")
    if predictions is None:
        raise TypeError("predictions are required")

    output_path = Path(path)
    payload = make_prediction_payload(predictions, metadata=metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=indent, allow_nan=False)
        file.write("\n")
    return output_path


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant {value!r} is not allowed")


def read_predictions(path: str | Path) -> list[dict[str, Any]]:
    """Read and validate prediction JSON, returning CPU PyTorch tensors."""
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as file:
        payload = json.load(file, parse_constant=_reject_nonstandard_constant)
    validate_prediction_payload(payload)
    records = _normalise_predictions(payload["predictions"])
    return [
        {
            "image_id": record["image_id"],
            "boxes": torch.tensor(record["boxes"], dtype=torch.float32).reshape(-1, 4),
            "scores": torch.tensor(record["scores"], dtype=torch.float32),
            "labels": torch.tensor(record["labels"], dtype=torch.long),
        }
        for record in records
    ]


def save_predictions(
    predictions: Any,
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    indent: int | None = 2,
) -> Path:
    """Object-first alias for :func:`write_predictions`."""
    return write_predictions(path, predictions, metadata=metadata, indent=indent)


load_predictions = read_predictions


__all__ = [
    "PREDICTION_BOX_FORMAT",
    "PREDICTION_COORDINATE_SPACE",
    "PREDICTION_FORMAT",
    "PREDICTION_SCHEMA_VERSION",
    "load_predictions",
    "make_prediction_payload",
    "read_predictions",
    "save_predictions",
    "validate_prediction_payload",
    "validate_predictions",
    "write_predictions",
]
