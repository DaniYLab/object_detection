from __future__ import annotations

from src.models import SharedPathwayCenterNet, build_model, get_model_preset


def _parameter_count(preset: str) -> int:
    model = build_model(preset)
    count = sum(parameter.numel() for parameter in model.parameters())
    del model
    return count


def test_direct_conditioning_control_uses_unconditioned_multiclass_pathway() -> None:
    control = get_model_preset("shared_no_condition")
    conditioned = get_model_preset("floorplan_base")

    assert control.architecture == "floorplan_unconditioned"
    assert conditioned.architecture == "floorplan_detector"
    assert control.pathway_mode == conditioned.pathway_mode == "shared"
    assert control.model_dim == conditioned.model_dim
    assert control.depth_per_class == conditioned.depth_per_class
    assert control.conditioner.kind == "none"
    assert control.fusion_mode == "none"
    assert isinstance(build_model(control), SharedPathwayCenterNet)


def test_budget_matched_ablation_presets_are_within_tolerance() -> None:
    shared_wide = _parameter_count("shared_wide")
    per_class = _parameter_count("per_class_no_text")
    assert abs(shared_wide - per_class) / per_class < 0.05

    per_class_small = _parameter_count("per_class_small")
    shared_control = _parameter_count("shared_no_condition")
    assert abs(per_class_small - shared_control) / shared_control < 0.01
