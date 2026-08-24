"""Versioned, complete checkpoint serialization for reproducible resumes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from ..models.conditioning import materialize_conditioners_for_state_dict
from .reproducibility import capture_rng_state, restore_rng_state

CHECKPOINT_SCHEMA_VERSION = 2
_PRE_SCHEMA_ERROR = (
    "Unsupported pre-schema checkpoint: checkpoints without the "
    "'floorplan_detector_checkpoint' schema were produced by an older, "
    "incompatible architecture and cannot be migrated automatically. "
    "Use a schema-v2 checkpoint created by the current architecture."
)


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is incomplete or incompatible."""


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def config_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_checkpoint(
    *,
    model: torch.nn.Module,
    epoch: int,
    global_step: int,
    model_config: Any,
    preset: str,
    class_names: list[str] | tuple[str, ...],
    output_stride: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    best_metric: float | None = None,
    metrics: Mapping[str, Any] | None = None,
    runtime_config: Mapping[str, Any] | None = None,
    split_manifest_fingerprint: str | None = None,
    metadata_fingerprint: str | None = None,
    data_generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Build the same complete payload for best, last, and periodic files."""
    plain_model_config = _plain(model_config)
    names = list(class_names)
    return {
        "schema": {
            "name": "floorplan_detector_checkpoint",
            "version": CHECKPOINT_SCHEMA_VERSION,
        },
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_metric": best_metric,
        "metrics": dict(metrics or {}),
        "model_config": plain_model_config,
        "model_config_fingerprint": config_fingerprint(plain_model_config),
        "preset": preset,
        "runtime_config": _plain(dict(runtime_config or {})),
        "output_stride": int(output_stride),
        "class_names": names,
        "class_mapping_fingerprint": config_fingerprint(names),
        "split_manifest_fingerprint": split_manifest_fingerprint,
        "metadata_fingerprint": metadata_fingerprint,
        "rng_state": capture_rng_state(data_generator),
    }


def save_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically save a checkpoint payload to a local trusted path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(destination)
    return destination


def _validate_checkpoint_schema(payload: Mapping[str, Any]) -> None:
    schema = payload.get("schema")
    if not isinstance(schema, Mapping):
        raise CheckpointError(_PRE_SCHEMA_ERROR)
    if schema.get("name") != "floorplan_detector_checkpoint":
        raise CheckpointError(f"Unknown checkpoint schema: {dict(schema)!r}")
    if schema.get("version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError(
            f"Unsupported checkpoint schema version {schema.get('version')}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}"
        )


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load and validate a trusted local schema-v2 checkpoint."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    try:
        payload = torch.load(
            checkpoint_path, map_location=map_location, weights_only=False
        )
    except TypeError:  # PyTorch versions before the weights_only argument.
        payload = torch.load(checkpoint_path, map_location=map_location)

    if not isinstance(payload, dict):
        raise CheckpointError("Checkpoint must be a mapping")
    _validate_checkpoint_schema(payload)
    if "model_state" not in payload:
        raise CheckpointError("Schema-v2 checkpoint must contain 'model_state'")
    return payload


def _validate_compatibility(
    payload: Mapping[str, Any],
    *,
    expected_model_config: Any | None,
    expected_class_names: list[str] | tuple[str, ...] | None,
    expected_output_stride: int | None,
) -> None:
    if expected_model_config is not None:
        expected = config_fingerprint(expected_model_config)
        actual = payload.get("model_config_fingerprint")
        if actual != expected:
            raise CheckpointError(
                "Model configuration mismatch: "
                f"checkpoint={actual}, current={expected}"
            )

    if expected_class_names is not None:
        expected = config_fingerprint(list(expected_class_names))
        actual = payload.get("class_mapping_fingerprint")
        if actual != expected:
            raise CheckpointError(
                "Class mapping mismatch: "
                f"checkpoint={actual}, current={expected}"
            )

    if expected_output_stride is not None:
        actual_stride = payload.get("output_stride")
        if actual_stride != expected_output_stride:
            raise CheckpointError(
                f"Output stride mismatch: checkpoint={actual_stride}, "
                f"current={expected_output_stride}"
            )


def restore_training_state(
    payload: Mapping[str, Any],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    data_generator: torch.Generator | None = None,
    expected_model_config: Any | None = None,
    expected_class_names: list[str] | tuple[str, ...] | None = None,
    expected_output_stride: int | None = None,
    weights_only: bool = False,
    strict: bool = True,
) -> dict[str, Any]:
    """Restore weights or a complete epoch-boundary training state.

    This helper does not execute a training step. Unsupported pre-schema and
    incomplete exact-resume payloads are rejected instead of silently
    restarting state.
    """
    if payload.get("_legacy", False):
        raise CheckpointError(_PRE_SCHEMA_ERROR)
    _validate_checkpoint_schema(payload)

    _validate_compatibility(
        payload,
        expected_model_config=expected_model_config,
        expected_class_names=expected_class_names,
        expected_output_stride=expected_output_stride,
    )

    model_state = payload["model_state"]
    if not isinstance(model_state, Mapping):
        raise CheckpointError("Checkpoint 'model_state' must be a mapping")
    materialize_conditioners_for_state_dict(model, model_state)
    model.load_state_dict(model_state, strict=strict)
    if weights_only:
        return {
            "start_epoch": 1,
            "global_step": 0,
            "best_metric": None,
            "legacy": False,
        }

    missing = []
    if optimizer is not None and payload.get("optimizer_state") is None:
        missing.append("optimizer_state")
    if scheduler is not None and payload.get("scheduler_state") is None:
        missing.append("scheduler_state")
    if payload.get("rng_state") is None:
        missing.append("rng_state")
    if missing:
        raise CheckpointError(
            f"Checkpoint is incomplete for exact resume; missing {missing}"
        )

    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None and payload.get("scaler_state") is not None:
        scaler.load_state_dict(payload["scaler_state"])
    restore_rng_state(payload["rng_state"], data_generator)

    return {
        "start_epoch": int(payload["epoch"]) + 1,
        "global_step": int(payload.get("global_step", 0)),
        "best_metric": payload.get("best_metric"),
        "legacy": False,
    }
