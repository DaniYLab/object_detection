"""Fast smoke checks and optional hardware profiling for model presets."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models import (  # noqa: E402
    FloorPlanDetector,
    VAEConfig,
    build_model,
    get_model_preset,
    list_model_presets,
)


def _check_outputs(
    outputs: dict[str, torch.Tensor],
    *,
    batch: int,
    classes: int,
    output_height: int,
    output_width: int,
) -> None:
    assert set(outputs) == {
        "center_heatmap",
        "center_logits",
        "size_map",
        "offset_map",
    }
    assert outputs["center_heatmap"].shape == (
        batch,
        classes,
        output_height,
        output_width,
    )
    assert outputs["center_logits"].shape == (
        batch,
        classes,
        output_height,
        output_width,
    )
    assert outputs["size_map"].shape == (
        batch,
        classes * 2,
        output_height,
        output_width,
    )
    assert outputs["offset_map"].shape == (
        batch,
        classes * 2,
        output_height,
        output_width,
    )
    assert torch.isfinite(outputs["center_heatmap"]).all()
    assert torch.isfinite(outputs["center_logits"]).all()
    assert torch.all(outputs["size_map"] > 0)
    assert torch.isfinite(outputs["offset_map"]).all()


def _smoke_preset(
    preset: str,
    *,
    device: torch.device,
    image_size: int,
    model_dim: int,
    depth: int,
) -> None:
    num_classes = 3
    num_heads = 4 if model_dim % 4 == 0 else 1
    image_cfg = VAEConfig(
        latent_channels=max(8, model_dim // 2),
        block_out_channels=(8, 12, 16),
        sample_size=image_size,
    )
    model = build_model(
        preset,
        image_size=image_size,
        model_dim=model_dim,
        num_classes=num_classes,
        depth_per_class=depth,
        num_heads=num_heads,
        dropout=0.0,
        class_chunk_size=2,
        head_channels=max(8, model_dim),
        vae=image_cfg,
    ).to(device).eval()
    images = torch.randn(2, 3, image_size, image_size, device=device)
    output_size = image_size // 8
    with torch.no_grad():
        all_classes = model(images)
        selected = model(images, class_ids=torch.tensor([2, 0], device=device))
    _check_outputs(
        all_classes,
        batch=2,
        classes=num_classes,
        output_height=output_size,
        output_width=output_size,
    )
    _check_outputs(
        selected,
        batch=2,
        classes=1,
        output_height=output_size,
        output_width=output_size,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"{preset}: ok ({parameter_count:,} parameters)")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _autocast_context(device: torch.device, amp_dtype: str):
    if amp_dtype == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("fp16 autocast is not supported for CPU profiling")
    return torch.autocast(device_type=device.type, dtype=dtype)


def _profile_preset(
    preset: str,
    *,
    device: torch.device,
    image_size: int,
    batch_size: int,
    class_chunk_size: int,
    warmup_steps: int,
    profile_steps: int,
    amp_dtype: str,
) -> dict[str, Any]:
    """Profile selected-query training and all-class inference."""

    config = get_model_preset(preset).with_overrides(
        image_size=image_size,
        class_chunk_size=class_chunk_size,
    )
    model = build_model(config).to(device)
    images = torch.randn(batch_size, 3, image_size, image_size, device=device)
    class_ids = torch.arange(batch_size, device=device) % config.num_classes
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    def training_step() -> None:
        model.zero_grad(set_to_none=True)
        with _autocast_context(device, amp_dtype):
            outputs = model(images, class_ids=class_ids)
            loss = (
                outputs["center_logits"].float().square().mean()
                + outputs["size_map"].float().mean()
                + outputs["offset_map"].float().mean()
            )
        loss.backward()

    model.train()
    for _ in range(warmup_steps):
        training_step()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(profile_steps):
        training_step()
    _sync(device)
    train_seconds = (time.perf_counter() - started) / profile_steps
    train_peak_allocated = None
    train_peak_reserved = None
    if device.type == "cuda":
        train_peak_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
        train_peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)

    model.zero_grad(set_to_none=True)
    model.eval()

    def inference_step() -> None:
        with torch.no_grad(), _autocast_context(device, amp_dtype):
            if isinstance(model, FloorPlanDetector):
                model(images, class_chunk_size=class_chunk_size)
            else:
                model(images)

    for _ in range(warmup_steps):
        inference_step()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(profile_steps):
        inference_step()
    _sync(device)
    inference_seconds = (time.perf_counter() - started) / profile_steps
    inference_peak_allocated = None
    inference_peak_reserved = None
    if device.type == "cuda":
        inference_peak_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
        inference_peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)

    result = {
        "preset": preset,
        "device": str(device),
        "amp_dtype": amp_dtype,
        "image_size": image_size,
        "batch_size": batch_size,
        "class_chunk_size": class_chunk_size,
        "num_classes": config.num_classes,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "selected_query_training": {
            "seconds_per_step": train_seconds,
            "images_per_second": batch_size / train_seconds,
            "peak_allocated_gib": train_peak_allocated,
            "peak_reserved_gib": train_peak_reserved,
        },
        "all_class_inference": {
            "seconds_per_step": inference_seconds,
            "seconds_per_image": inference_seconds / batch_size,
            "images_per_second": batch_size / inference_seconds,
            "peak_allocated_gib": inference_peak_allocated,
            "peak_reserved_gib": inference_peak_reserved,
        },
    }
    print(
        f"{preset}: train={train_seconds:.3f}s/step, "
        f"all-class={inference_seconds / batch_size:.3f}s/image, "
        f"params={total_params / 1e6:.1f}M"
    )
    if train_peak_allocated is not None:
        print(
            f"  CUDA peak: train={train_peak_allocated:.2f} GiB, "
            f"inference={inference_peak_allocated:.2f} GiB"
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--model-dim", type=int, default=16)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument(
        "--preset",
        action="append",
        dest="presets",
        help="Preset to check/profile. Repeat for multiple presets.",
    )
    parser.add_argument(
        "--all-lightweight-presets",
        action="store_true",
        help="check every preset that does not require pretrained text weights",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Profile actual preset dimensions for selected-query forward/backward "
            "and all-class inference instead of using smoke-size overrides."
        ),
    )
    parser.add_argument("--profile-batch-size", type=int, default=1)
    parser.add_argument("--profile-warmup-steps", type=int, default=1)
    parser.add_argument("--profile-steps", type=int, default=2)
    parser.add_argument("--class-chunk-size", type=int, default=4)
    parser.add_argument(
        "--amp-dtype",
        choices=["fp32", "bf16", "fp16"],
        default="fp32",
    )
    parser.add_argument("--report", default=None, help="Optional JSON profile report")
    return parser.parse_args()


def _selected_presets(args: argparse.Namespace) -> list[str]:
    if args.presets:
        return list(dict.fromkeys(args.presets))
    if args.all_lightweight_presets:
        return [
            name
            for name in list_model_presets()
            if get_model_preset(name).conditioner.kind != "pretrained_text"
        ]
    if args.profile:
        return ["centernet_baseline", "floorplan_base"]
    return ["floorplan_base", "per_class_fixed_byte_text_film", "centernet_shared"]


def main() -> None:
    args = parse_args()
    if args.image_size <= 0 or args.image_size % 8 != 0:
        raise ValueError("--image-size must be positive and divisible by 8")
    if args.model_dim <= 0:
        raise ValueError("--model-dim must be positive")
    if args.depth < 0:
        raise ValueError("--depth cannot be negative")
    if args.profile_batch_size <= 0:
        raise ValueError("--profile-batch-size must be positive")
    if args.profile_warmup_steps < 0:
        raise ValueError("--profile-warmup-steps cannot be negative")
    if args.profile_steps <= 0:
        raise ValueError("--profile-steps must be positive")
    if args.class_chunk_size <= 0:
        raise ValueError("--class-chunk-size must be positive")

    device = torch.device(args.device)
    torch.manual_seed(0)
    presets = _selected_presets(args)

    if args.profile:
        results = [
            _profile_preset(
                preset,
                device=device,
                image_size=args.image_size,
                batch_size=args.profile_batch_size,
                class_chunk_size=args.class_chunk_size,
                warmup_steps=args.profile_warmup_steps,
                profile_steps=args.profile_steps,
                amp_dtype=args.amp_dtype,
            )
            for preset in presets
        ]
        if args.report:
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps({"profiles": results}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"profile report: {report_path}")
        print(f"model profiling passed: {len(results)} preset(s)")
        return

    for preset in presets:
        _smoke_preset(
            preset,
            device=device,
            image_size=args.image_size,
            model_dim=args.model_dim,
            depth=args.depth,
        )
    print(f"model smoke checks passed: {len(presets)} preset(s)")


if __name__ == "__main__":
    main()
