from __future__ import annotations

import pytest
import torch

from src.models import CenterNetBaseline, FloorPlanDetector, ModelConfig, VAEConfig


def test_stride_eight_is_the_only_supported_output_stride() -> None:
    with pytest.raises(ValueError, match="output_stride=8"):
        ModelConfig(output_stride=4)
    with pytest.raises(ValueError, match="output_stride=8"):
        FloorPlanDetector(output_stride=4)
    with pytest.raises(ValueError, match="output_stride=8"):
        CenterNetBaseline(output_stride=4)


def test_constructor_and_runtime_image_sizes_validate_stride_contract() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(image_size=30)

    model = FloorPlanDetector(
        image_size=32,
        model_dim=16,
        num_classes=2,
        depth_per_class=0,
        num_heads=4,
        dropout=0.0,
        pathway_mode="shared",
        conditioner="class_embedding",
        head_channels=16,
        vae_cfg=VAEConfig(latent_channels=8, block_out_channels=(8, 12, 16)),
    ).eval()
    with pytest.raises(ValueError, match="divisible"):
        model(torch.randn(1, 3, 30, 32))


def test_rectangular_images_keep_exact_stride_eight_output() -> None:
    model = FloorPlanDetector(
        image_size=32,
        model_dim=16,
        num_classes=2,
        depth_per_class=0,
        num_heads=4,
        dropout=0.0,
        pathway_mode="shared",
        conditioner="class_embedding",
        head_channels=16,
        vae_cfg=VAEConfig(latent_channels=8, block_out_channels=(8, 12, 16)),
    ).eval()
    with torch.no_grad():
        outputs = model(torch.randn(1, 3, 32, 48))
    assert outputs["center_heatmap"].shape == (1, 2, 4, 6)
    assert outputs["size_map"].shape == (1, 4, 4, 6)
    assert outputs["offset_map"].shape == (1, 4, 4, 6)
