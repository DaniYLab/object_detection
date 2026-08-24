"""Model construction from serializable configs and named presets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn

from .baseline import SharedCenterNetBaseline, SharedPathwayCenterNet
from .config import ModelConfig
from .detector import FloorPlanDetector
from .presets import resolve_model_config


def build_model(
    config: ModelConfig | Mapping[str, Any] | str | None = None,
    *,
    preset: str | None = None,
    **overrides: Any,
) -> nn.Module:
    """Build a registered architecture from a config, dictionary, or preset.

    Examples:
        ``build_model("floorplan_base")``
        ``build_model(preset="centernet_baseline", num_classes=4)``
        ``build_model(ModelConfig(...))``
        ``build_model(checkpoint["model_config"])``
    """
    if preset is not None:
        if config is not None:
            raise ValueError("pass config or preset, not both")
        config = preset
    resolved = resolve_model_config(config, **overrides)

    if resolved.architecture == "floorplan_detector":
        return FloorPlanDetector(config=resolved)
    if resolved.architecture == "floorplan_unconditioned":
        return SharedPathwayCenterNet(resolved)
    if resolved.architecture == "centernet_baseline":
        model = SharedCenterNetBaseline(
            image_size=resolved.image_size,
            model_dim=resolved.model_dim,
            num_classes=resolved.num_classes,
            depth=resolved.depth_per_class,
            output_stride=resolved.output_stride,
            head_channels=resolved.head_channels,
            image_cfg=resolved.vae,
        )
        model.config = resolved
        return model
    raise ValueError(f"unsupported model architecture: {resolved.architecture}")


create_model = build_model
