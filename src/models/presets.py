"""Named model presets and preset resolution helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .config import (
    MODEL_PRESETS,
    PRESET_REGISTRY,
    ModelConfig,
    get_model_preset,
    get_preset,
    list_model_presets,
    register_model_preset,
    register_preset,
)


def resolve_model_config(
    config: ModelConfig | Mapping[str, Any] | str | None = None,
    **overrides: Any,
) -> ModelConfig:
    """Resolve a model config, plain dictionary, preset name, or default preset."""
    if config is None:
        resolved = get_model_preset("floorplan_base")
    elif isinstance(config, str):
        resolved = get_model_preset(config)
    elif isinstance(config, ModelConfig):
        resolved = ModelConfig.from_dict(config.to_dict())
    elif isinstance(config, Mapping):
        resolved = ModelConfig.from_dict(dict(config))
    else:
        raise TypeError("config must be ModelConfig, a mapping, a preset name, or None")
    return resolved.with_overrides(**overrides) if overrides else resolved


__all__ = [
    "MODEL_PRESETS",
    "PRESET_REGISTRY",
    "get_model_preset",
    "get_preset",
    "list_model_presets",
    "register_model_preset",
    "register_preset",
    "resolve_model_config",
]
