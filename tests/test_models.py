from __future__ import annotations

import json

import pytest
import torch

from src.models import (
    CenterNetBaseline,
    ConditionerConfig,
    EarlyFusion,
    FloorPlanDetector,
    ModelConfig,
    SharedPathwayCenterNet,
    VAEConfig,
    build_model,
    get_model_preset,
)
from src.models.conditioning import ConditioningOutput


def tiny_config(*, pathway_mode: str = "shared", conditioner: str = "byte") -> ModelConfig:
    return ModelConfig(
        image_size=32,
        model_dim=16,
        num_classes=3,
        depth_per_class=1,
        num_heads=4,
        dropout=0.0,
        pathway_mode=pathway_mode,
        class_chunk_size=2,
        head_channels=16,
        vae=VAEConfig(latent_channels=8, block_out_channels=(8, 12, 16)),
        conditioner=ConditionerConfig(kind=conditioner, max_length=12, embedding_dim=8),
    )


def assert_output_contract(outputs: dict[str, torch.Tensor], batch: int, classes: int) -> None:
    required = {"center_heatmap", "size_map", "offset_map"}
    assert required.issubset(set(outputs)), f"Missing keys: {required - set(outputs)}"
    assert outputs["center_heatmap"].shape == (batch, classes, 4, 4)
    assert outputs["size_map"].shape == (batch, classes * 2, 4, 4)
    assert outputs["offset_map"].shape == (batch, classes * 2, 4, 4)
    assert torch.all((outputs["center_heatmap"] >= 0) & (outputs["center_heatmap"] <= 1))
    assert torch.all(outputs["size_map"] > 0)
    assert torch.all((outputs["offset_map"] >= 0) & (outputs["offset_map"] <= 1))
    if "center_logits" in outputs:
        assert outputs["center_logits"].shape == outputs["center_heatmap"].shape
        assert torch.isfinite(outputs["center_logits"]).all()


def test_config_round_trip_and_preset_registry_return_independent_configs() -> None:
    config = tiny_config()
    payload = json.loads(json.dumps(config.to_dict()))
    restored = ModelConfig.from_dict(payload)
    assert restored == config
    assert restored.model_dim == 16

    first = get_model_preset("floorplan_tiny")
    second = get_model_preset("floorplan_tiny")
    first.model_dim = 128
    assert second.model_dim == 64
    assert ModelConfig().model_dim == 256
    assert ModelConfig().pathway_mode == "shared"
    assert ModelConfig().conditioner.kind == "class_embedding"
    assert get_model_preset("per_class_no_text").fusion_mode == "none"
    assert get_model_preset("shared_pretrained_text").conditioner.kind == "pretrained_text"
    late = get_model_preset("dual_pathway_late")
    assert late.conditioning_stage == "late"
    assert late.conditioner.kind == "class_embedding"
    assert late.vector.enabled
    assert not late.vector.class_conditioned
    assert ModelConfig.from_dict(late.to_dict()) == late


def test_build_model_dispatches_configs_and_presets() -> None:
    detector = build_model(tiny_config())
    baseline = build_model(
        "centernet_baseline", image_size=32, model_dim=16, num_classes=3,
        depth_per_class=1, head_channels=16,
        vae=VAEConfig(latent_channels=8, block_out_channels=(8, 12, 16)),
    )
    assert isinstance(detector, FloorPlanDetector)
    assert isinstance(baseline, CenterNetBaseline)
    unconditioned = build_model(
        "shared_no_condition", image_size=32, model_dim=16, num_classes=3,
        depth_per_class=1, num_heads=4, head_channels=16,
        vae=VAEConfig(latent_channels=8, block_out_channels=(8, 12, 16)),
    )
    assert isinstance(unconditioned, SharedPathwayCenterNet)
    with torch.no_grad():
        assert_output_contract(unconditioned(torch.randn(2, 3, 32, 32)), 2, 3)
        assert_output_contract(
            unconditioned(torch.randn(2, 3, 32, 32), class_ids=torch.tensor([2, 0])), 2, 1
        )
    lazy = build_model(
        "shared_pretrained_text", image_size=32, model_dim=16, num_classes=3,
        depth_per_class=0, num_heads=4, head_channels=16,
        vae=VAEConfig(latent_channels=8, block_out_channels=(8, 12, 16)),
    )
    assert not lazy.conditioner.is_loaded


@pytest.mark.parametrize("mode", ["none", "add", "film", "cross_attention", "film_cross_attn", "current"])
def test_fusion_modes_accept_masked_conditioning(mode: str) -> None:
    fusion = EarlyFusion(dim=16, num_heads=4, mode=mode, dropout=0.0).eval()
    image_tokens = torch.randn(2, 4, 16)
    tokens = torch.randn(2, 3, 16)
    mask = torch.tensor([[True, True, False], [False, False, False]])
    tokens[~mask] = 0
    conditioning = ConditioningOutput(tokens, mask, masked_pool(tokens, mask))
    with torch.no_grad():
        output = fusion(image_tokens, conditioning)
    assert output.shape == image_tokens.shape
    assert torch.isfinite(output).all()


def masked_pool(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(tokens.dtype)
    return (tokens * weights).sum(1) / weights.sum(1).clamp_min(1)


@pytest.mark.parametrize("pathway_mode", ["shared", "per_class"])
def test_detector_selected_and_all_class_contracts(pathway_mode: str) -> None:
    model = FloorPlanDetector(tiny_config(pathway_mode=pathway_mode)).eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        selected = model(images, class_ids=torch.tensor([2, 0]), texts=["window", "wall"])
        all_classes = model(images, texts=["wall", "door", "window"], class_chunk_size=2)
    assert_output_contract(selected, batch=2, classes=1)
    assert_output_contract(all_classes, batch=2, classes=3)


def test_shared_class_embedding_pathway_and_native_baseline() -> None:
    model = FloorPlanDetector(tiny_config(conditioner="class_embedding")).eval()
    baseline = CenterNetBaseline(
        image_size=32, model_dim=16, num_classes=3, depth=1, head_channels=16,
        image_cfg=VAEConfig(latent_channels=8, block_out_channels=(8, 12, 16)),
    ).eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert_output_contract(model(images), batch=2, classes=3)
        assert_output_contract(baseline(images), batch=2, classes=3)
        assert_output_contract(baseline(images, class_ids=[1, 2]), batch=2, classes=1)


def test_late_conditioning_reuses_shared_trunk_and_preserves_contract() -> None:
    config = ModelConfig(
        image_size=32, model_dim=16, num_classes=3, depth_per_class=1, num_heads=4,
        dropout=0.0, fusion_mode="none", pathway_mode="shared", conditioning_stage="late",
        class_chunk_size=2, head_channels=16,
        vae=VAEConfig(latent_channels=8, block_out_channels=(8, 12, 16)),
        conditioner=ConditionerConfig(kind="class_embedding", embedding_dim=8),
        vector={"enabled": True, "class_conditioned": False, "n_max": 8},
    )
    model = FloorPlanDetector(config).eval()
    images = torch.randn(2, 3, 32, 32)
    strokes = torch.randn(2, 5, 12)
    mask = torch.ones(2, 5, dtype=torch.bool)
    calls = 0

    def count_calls(_module, _inputs, _output) -> None:
        nonlocal calls
        calls += 1

    hook = model.shared_blocks[0].register_forward_hook(count_calls)
    try:
        with torch.no_grad():
            selected = model(images, class_ids=torch.tensor([0, 2]), stroke_tokens=strokes, stroke_mask=mask)
            all_classes = model(images, stroke_tokens=strokes, stroke_mask=mask, class_chunk_size=2)
    finally:
        hook.remove()
    assert calls == 2
    assert_output_contract(selected, batch=2, classes=1)
    assert_output_contract(all_classes, batch=2, classes=3)
    assert model.conditioning_stage == "late"
    assert model.late_head is not None
    assert model.heatmap_head is None


def test_late_conditioning_rejects_class_conditioned_vectors() -> None:
    config = ModelConfig(
        image_size=32, model_dim=16, num_classes=3, num_heads=4, fusion_mode="none",
        conditioning_stage="late", vae=VAEConfig(latent_channels=8, block_out_channels=(8, 12, 16)),
        vector={"enabled": True, "class_conditioned": True},
    )
    with pytest.raises(ValueError, match="class-agnostic"):
        FloorPlanDetector(config)


def test_late_conditioning_rejects_missing_class_conditioner() -> None:
    with pytest.raises(ValueError, match="class-specific conditioner"):
        ModelConfig(
            fusion_mode="none",
            conditioning_stage="late",
            conditioner=ConditionerConfig(kind="none"),
        )


def test_direct_constructor_exposes_late_conditioning_stage() -> None:
    model = FloorPlanDetector(
        image_size=32,
        model_dim=16,
        num_classes=3,
        depth_per_class=1,
        num_heads=4,
        dropout=0.0,
        fusion_mode="none",
        output_stride=8,
        pathway_mode="shared",
        conditioning_stage="late",
        head_channels=16,
        vae_cfg=VAEConfig(latent_channels=8, block_out_channels=(8, 12, 16)),
        conditioner="class_embedding",
    )
    assert model.conditioning_stage == "late"
    assert model.late_head is not None
    assert model.heatmap_head is None


def test_thin_head_upsamples_to_finer_grid_and_round_trips() -> None:
    cfg = ModelConfig(
        image_size=32,
        output_stride=8,
        head_stride=4,
        model_dim=16,
        num_classes=3,
        depth_per_class=1,
        num_heads=4,
        dropout=0.0,
        pathway_mode="shared",
        class_chunk_size=2,
        head_channels=16,
        vae=VAEConfig(latent_channels=8, block_out_channels=(8, 12, 16)),
        conditioner=ConditionerConfig(kind="class_embedding", max_length=12, embedding_dim=8),
    )
    assert cfg.head_size == 8  # 32 // 4
    # Round-trip preserves head_stride.
    assert ModelConfig.from_dict(cfg.to_dict()) == cfg

    model = FloorPlanDetector(config=cfg).eval()
    with torch.no_grad():
        selected = model(
            torch.randn(2, 3, 32, 32),
            class_ids=torch.tensor([0, 1]),
        )
        all_classes = model(torch.randn(2, 3, 32, 32), class_chunk_size=2)
    # Trunk is stride-8 (4x4) but head grid is stride-4 (8x8).
    assert selected["center_heatmap"].shape == (2, 1, 8, 8)
    assert selected["size_map"].shape == (2, 2, 8, 8)
    assert all_classes["center_heatmap"].shape == (2, 3, 8, 8)
    assert all_classes["size_map"].shape == (2, 6, 8, 8)


def test_thin_head_rejects_coarser_grid() -> None:
    with pytest.raises(ValueError, match="head_stride"):
        ModelConfig(output_stride=8, head_stride=16)
