"""Train the FloorPlanCAD detector with reproducible, resumable state.

Examples:
    python train.py --manifest ./data/FloorPlanCAD_original/splits.json
    python train.py --preset floorplan_base --resume ./checkpoints/last.pt
    python train.py --resume ./checkpoints/best.pt --weights-only
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from src.data import (
    CLASS_NAMES,
    NUM_CLASSES,
    FloorPlanImageDataset,
    FloorPlanQueryDataset,
    TargetStats,
    image_collate_fn,
    image_index_fingerprint,
    load_metadata,
    load_split_manifest,
    query_collate_fn,
    resolve_split_manifest_path,
    split_manifest_fingerprint,
    validate_metadata_sources,
)
from src.evaluation import CenterNetDecoder, compute_coco_metrics
from src.models import (
    ModelConfig,
    build_model,
    list_model_presets,
    materialize_pretrained_conditioners,
    resolve_model_config,
)
from src.training.checkpoint import (
    CheckpointError,
    build_checkpoint,
    load_checkpoint,
    restore_training_state,
    save_checkpoint,
)
from src.training.losses import centernet_loss, focal_loss, l1_loss_masked
from src.training.reproducibility import (
    make_generator,
    seed_everything,
    seed_worker,
)

# Imported loss names above intentionally remain module attributes for callers that
# historically used ``from train import focal_loss`` and related helpers.

_CHECKPOINT_INTERVAL = 5
_VAL_AP_INTERVAL_DEFAULT = 5  # Run val AP every N epochs by default.
_TARGET_STAT_FIELDS = (
    "total_boxes",
    "valid_boxes",
    "encoded_boxes",
    "invalid_boxes",
    "out_of_bounds_boxes",
    "collisions",
    "replacements",
    "ignored_collisions",
)
_TEXT_CONDITIONER_KINDS = {
    "byte",
    "hf_pretrained",
    "lightweight_text",
    "pretrained_text",
}
_FUSION_MODES = (
    "none",
    "add",
    "film",
    "cross_attention",
    "film_cross_attention",
    "current",
    # Historical spellings accepted by EarlyFusion.
    "identity",
    "additive",
    "cross_attn",
    "film_cross_attn",
)


def resolve_model_config_from_args(args: argparse.Namespace) -> ModelConfig:
    """Resolve a named preset plus explicitly supplied legacy CLI overrides."""

    requested_stride = getattr(args, "output_stride", None)
    if requested_stride not in {None, 8}:
        raise ValueError(
            "Training requires output_stride=8; use --output_stride 8 or a stride-8 preset"
        )

    overrides: dict[str, Any] = {"num_classes": NUM_CLASSES}
    for argument, config_key in (
        ("image_size", "image_size"),
        ("model_dim", "model_dim"),
        ("depth_per_class", "depth_per_class"),
        ("fusion_mode", "fusion_mode"),
        ("output_stride", "output_stride"),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            overrides[config_key] = value

    config = resolve_model_config(args.preset, **overrides)
    if config.output_stride != 8:
        raise ValueError(
            "Training requires output_stride=8; use --output_stride 8 or a stride-8 preset"
        )
    if config.num_classes != NUM_CLASSES:
        raise ValueError(
            f"Training requires num_classes={NUM_CLASSES}, got {config.num_classes}"
        )
    return config


def _validate_args(args: argparse.Namespace) -> None:
    if args.weights_only and not args.resume:
        raise ValueError("--weights-only requires --resume CHECKPOINT")
    if not 0 <= args.seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.lr <= 0:
        raise ValueError("lr must be positive")
    if args.warmup_steps <= 0:
        raise ValueError("warmup_steps must be positive")
    if not 0 < args.warmup_start_factor <= 1:
        raise ValueError("warmup_start_factor must be in (0, 1]")
    if not 0 <= args.min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    if min(args.focal_weight, args.size_weight, args.offset_weight) < 0:
        raise ValueError("loss weights cannot be negative")
    if args.grad_clip <= 0:
        raise ValueError("grad_clip must be positive")
    if args.balance_power < 0:
        raise ValueError("balance_power cannot be negative")
    if args.limit_train_samples < 0 or args.limit_val_samples < 0:
        raise ValueError("sample limits cannot be negative")
    if args.log_interval <= 0:
        raise ValueError("log_interval must be positive")
    if args.val_ap_interval < 0:
        raise ValueError("val_ap_interval cannot be negative")
    if args.val_ap_chunk_size <= 0:
        raise ValueError("val_ap_chunk_size must be positive")
    if args.limit_val_ap_images < 0:
        raise ValueError("limit_val_ap_images cannot be negative")


def _resolve_training_manifest(
    data_root: str | Path,
    manifest_path: str | Path | None,
) -> tuple[Path, dict[str, Any], str]:
    """Require and load the one image-level manifest used by train and val."""

    resolved = resolve_split_manifest_path(data_root, manifest_path)
    if resolved is None:
        raise RuntimeError(
            "Validation requires an image-level split manifest, but none was found. "
            "Pass --manifest PATH or create data_root/splits.json with "
            "scripts/data/build_splits.py."
        )
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Validation split manifest does not exist: {resolved}. "
            "Pass a valid --manifest PATH."
        )

    manifest = load_split_manifest(resolved)
    val_records = manifest.get("splits", {}).get("val", [])
    if not val_records:
        raise RuntimeError(
            f"Validation split is empty in image-level manifest: {resolved}. "
            "Rebuild the manifest with a non-zero validation fraction."
        )
    fingerprint = manifest.get("fingerprint") or split_manifest_fingerprint(manifest)
    return resolved, manifest, str(fingerprint)


def _metadata_source_path(
    metadata: Mapping[str, Any],
    metadata_path: Path,
    source_name: str,
    fallback: Path,
) -> Path:
    """Resolve a source path recorded in schema-v2 metadata."""

    source = metadata.get("source")
    if isinstance(source, Mapping):
        entry = source.get(source_name)
        if isinstance(entry, Mapping):
            stored_path = entry.get("path")
            if isinstance(stored_path, str) and stored_path:
                candidate = Path(stored_path)
                return candidate if candidate.is_absolute() else metadata_path.parent / candidate
    return fallback


def _validate_source_records(
    data_root: str | Path,
    records: Sequence[Any],
) -> list[str]:
    """Return source-drift errors for the supplied image records."""

    root = Path(data_root)
    errors: list[str] = []
    for record in records:
        image_path = Path(record.image_path)
        if not image_path.is_absolute():
            image_path = root / image_path
        metadata_path = Path(record.metadata_path)
        if not metadata_path.is_absolute():
            metadata_path = root / metadata_path

        try:
            metadata = load_metadata(metadata_path, allow_legacy=True)
        except Exception as exc:
            errors.append(f"{record.image_id}: cannot load metadata: {exc}")
            continue

        metadata_stem = metadata_path.stem
        svg_stem = metadata_stem[:-5] if metadata_stem.endswith("_meta") else metadata_stem
        svg_path = _metadata_source_path(
            metadata,
            metadata_path,
            "svg",
            metadata_path.with_name(f"{svg_stem}.svg"),
        )
        report = validate_metadata_sources(metadata, image_path, svg_path)
        errors.extend(
            f"{record.image_id}: {issue.path}: {issue.message}"
            for issue in report.errors
        )
    return errors


# ── DataLoader helpers ─────────────────────────────────────────────────────────


def _maybe_subset(
    dataset: FloorPlanQueryDataset,
    limit: int,
) -> FloorPlanQueryDataset | Subset:
    if limit <= 0 or limit >= len(dataset):
        return dataset
    return Subset(dataset, list(range(limit)))


def _make_train_loader(
    dataset: Dataset,
    base_dataset: FloorPlanQueryDataset,
    args: argparse.Namespace,
    pin_memory: bool,
    data_generator: torch.Generator | None = None,
) -> DataLoader:
    if data_generator is None:
        data_generator = make_generator(getattr(args, "seed", 1337))
    sampler = None
    shuffle = True

    if args.sampler == "balanced":
        weights = base_dataset.get_sample_weights(balance_power=args.balance_power)
        if isinstance(dataset, Subset):
            weights = weights[dataset.indices]
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(dataset),
            replacement=True,
            generator=data_generator,
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=query_collate_fn,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        generator=data_generator,
        worker_init_fn=seed_worker,
        persistent_workers=args.num_workers > 0 and args.cache_images,
    )


def _make_val_loader(
    dataset: Dataset,
    args: argparse.Namespace,
    pin_memory: bool,
    data_generator: torch.Generator,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=query_collate_fn,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        generator=data_generator,
        worker_init_fn=seed_worker,
        persistent_workers=args.num_workers > 0 and args.cache_images,
    )


def _print_class_stats(dataset: FloorPlanQueryDataset) -> None:
    counts = dataset.class_counts
    if not counts:
        return
    sorted_counts = sorted(counts.items(), key=lambda item: item[1])
    low = ", ".join(f"{CLASS_NAMES[class_id]}={count}" for class_id, count in sorted_counts[:5])
    high = ", ".join(f"{CLASS_NAMES[class_id]}={count}" for class_id, count in sorted_counts[-5:])
    print(f"Class counts: min={sorted_counts[0][1]} | max={sorted_counts[-1][1]}")
    print(f"  rare : {low}")
    print(f"  common: {high}")


def _accumulate_target_stats(
    aggregate: TargetStats,
    batch_stats: Sequence[TargetStats],
) -> None:
    for stats in batch_stats:
        if not isinstance(stats, TargetStats):
            raise TypeError(
                "query_collate_fn must preserve TargetStats objects in batch['target_stats']"
            )
        for field_name in _TARGET_STAT_FIELDS:
            setattr(
                aggregate,
                field_name,
                getattr(aggregate, field_name) + getattr(stats, field_name),
            )


def _target_stat_metrics(stats: TargetStats, prefix: str = "") -> dict[str, float]:
    return {
        f"{prefix}target_total_boxes": float(stats.total_boxes),
        f"{prefix}target_valid_boxes": float(stats.valid_boxes),
        f"{prefix}target_encoded_boxes": float(stats.encoded_boxes),
        f"{prefix}target_invalid_boxes": float(stats.invalid_boxes),
        f"{prefix}target_out_of_bounds_boxes": float(stats.out_of_bounds_boxes),
        f"{prefix}target_collisions": float(stats.collisions),
        f"{prefix}target_replacements": float(stats.replacements),
        f"{prefix}target_ignored_collisions": float(stats.ignored_collisions),
        f"{prefix}target_collision_rate": stats.collision_rate,
    }


def _uses_text_conditioning(model: nn.Module) -> bool:
    config = getattr(model, "config", None)
    conditioner = getattr(config, "conditioner", None)
    return getattr(conditioner, "kind", None) in _TEXT_CONDITIONER_KINDS


def _forward_query_batch(
    model: nn.Module,
    image: torch.Tensor,
    class_ids: torch.Tensor,
    texts: Sequence[str],
    stroke_tokens: torch.Tensor | None = None,
    stroke_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    kwargs: dict[str, Any] = {}
    if getattr(model, "vector_enabled", False):
        if stroke_tokens is None or stroke_mask is None:
            raise ValueError("vector-branch models require stroke_tokens and stroke_mask")
        kwargs["stroke_tokens"] = stroke_tokens
        kwargs["stroke_mask"] = stroke_mask
    if _uses_text_conditioning(model):
        return model(image, class_ids=class_ids, texts=list(texts), **kwargs)
    return model(image, class_ids=class_ids, **kwargs)


def _resolve_precision(requested: str, device: torch.device) -> str:
    """Resolve auto/fp32/bf16/fp16 and reject unsupported combinations."""

    if requested == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return "bf16"
        return "fp32"
    if requested == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 was requested but the CUDA device does not support BF16")
    if requested == "fp16" and device.type != "cuda":
        raise ValueError("FP16 training is supported only on CUDA")
    return requested


def _autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _build_grad_scaler(device: torch.device, precision: str) -> Any | None:
    """Build a scaler for FP16; BF16 has sufficient exponent range without one."""

    if precision != "fp16" or device.type != "cuda":
        return None
    try:
        return torch.amp.GradScaler(device.type)
    except (AttributeError, TypeError):  # pragma: no cover - old PyTorch fallback
        return torch.cuda.amp.GradScaler()


# ── Training and validation loops ──────────────────────────────────────────────


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    epoch: int,
    focal_w: float = 10.0,
    size_w: float = 1.0,
    offset_w: float = 1.0,
    grad_clip: float = 1.0,
    log_interval: int = 20,
    precision: str = "fp32",
    scaler: Any | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = focal_sum = size_sum = offset_sum = pos_sum = grad_sum = 0.0
    target_stats = TargetStats()
    start_time = time.time()

    for step, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=device.type == "cuda")
        targets = {
            "center_heatmap": batch["center_heatmap"].to(
                device, non_blocking=device.type == "cuda"
            ),
            "size_map": batch["size_map"].to(
                device, non_blocking=device.type == "cuda"
            ),
            "offset_map": batch["offset_map"].to(
                device, non_blocking=device.type == "cuda"
            ),
            "mask_map": batch["mask_map"].to(
                device, non_blocking=device.type == "cuda"
            ),
        }
        class_ids = torch.as_tensor(batch["class_ids"], dtype=torch.long, device=device)
        _accumulate_target_stats(target_stats, batch.get("target_stats", ()))
        stroke_tokens = batch.get("stroke_tokens")
        stroke_mask = batch.get("stroke_mask")
        if stroke_tokens is not None:
            stroke_tokens = stroke_tokens.to(device, non_blocking=device.type == "cuda")
            stroke_mask = stroke_mask.to(device, non_blocking=device.type == "cuda")

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, precision):
            predictions = _forward_query_batch(
                model,
                image,
                class_ids,
                batch.get("texts", ()),
                stroke_tokens=stroke_tokens,
                stroke_mask=stroke_mask,
            )
            losses = centernet_loss(
                predictions,
                targets,
                focal_w=focal_w,
                size_w=size_w,
                offset_w=offset_w,
            )

        if scaler is not None:
            previous_scale = float(scaler.get_scale())
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer_stepped = float(scaler.get_scale()) >= previous_scale
        else:
            losses["total"].backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            optimizer_stepped = True
        if optimizer_stepped:
            scheduler.step()

        total_loss += losses["total"].item()
        focal_sum += losses["focal"].item()
        size_sum += losses["size_l1"].item()
        offset_sum += losses["offset_l1"].item()
        pos_sum += losses["num_pos"].item()
        grad_sum += float(grad_norm)

        if (step + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            denominator = step + 1
            print(
                f"  Epoch {epoch} | Step {step + 1:4d}/{len(loader)} | "
                f"Loss {total_loss / denominator:.4f} "
                f"(focal={focal_sum / denominator:.4f}×{focal_w:g}, "
                f"size={size_sum / denominator:.4f}×{size_w:g}, "
                f"offset={offset_sum / denominator:.4f}×{offset_w:g}, "
                f"pos={pos_sum / denominator:.1f}, "
                f"grad={grad_sum / denominator:.3f}) | "
                f"Collisions {target_stats.collisions}/{target_stats.valid_boxes} "
                f"({target_stats.collision_rate:.2%}) | "
                f"LR {scheduler.get_last_lr()[0]:.2e} | "
                f"{elapsed:.1f}s"
            )

    batch_count = len(loader)
    if batch_count == 0:
        raise RuntimeError("Training DataLoader is empty")
    metrics = {
        "loss": total_loss / batch_count,
        "focal": focal_sum / batch_count,
        "size_l1": size_sum / batch_count,
        "offset_l1": offset_sum / batch_count,
        "num_pos": pos_sum / batch_count,
        "grad_norm": grad_sum / batch_count,
    }
    metrics.update(_target_stat_metrics(target_stats))
    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    focal_w: float = 10.0,
    size_w: float = 1.0,
    offset_w: float = 1.0,
    precision: str = "fp32",
) -> dict[str, float]:
    model.eval()
    total_loss = focal_sum = size_sum = offset_sum = pos_sum = 0.0
    target_stats = TargetStats()
    batch_count = 0

    for batch in loader:
        image = batch["image"].to(device, non_blocking=device.type == "cuda")
        targets = {
            "center_heatmap": batch["center_heatmap"].to(
                device, non_blocking=device.type == "cuda"
            ),
            "size_map": batch["size_map"].to(
                device, non_blocking=device.type == "cuda"
            ),
            "offset_map": batch["offset_map"].to(
                device, non_blocking=device.type == "cuda"
            ),
            "mask_map": batch["mask_map"].to(
                device, non_blocking=device.type == "cuda"
            ),
        }
        class_ids = torch.as_tensor(batch["class_ids"], dtype=torch.long, device=device)
        _accumulate_target_stats(target_stats, batch.get("target_stats", ()))
        stroke_tokens = batch.get("stroke_tokens")
        stroke_mask = batch.get("stroke_mask")
        if stroke_tokens is not None:
            stroke_tokens = stroke_tokens.to(device, non_blocking=device.type == "cuda")
            stroke_mask = stroke_mask.to(device, non_blocking=device.type == "cuda")

        with _autocast_context(device, precision):
            predictions = _forward_query_batch(
                model,
                image,
                class_ids,
                batch.get("texts", ()),
                stroke_tokens=stroke_tokens,
                stroke_mask=stroke_mask,
            )
            losses = centernet_loss(
                predictions,
                targets,
                focal_w=focal_w,
                size_w=size_w,
                offset_w=offset_w,
            )
        total_loss += losses["total"].item()
        focal_sum += losses["focal"].item()
        size_sum += losses["size_l1"].item()
        offset_sum += losses["offset_l1"].item()
        pos_sum += losses["num_pos"].item()
        batch_count += 1

    if batch_count == 0:
        raise RuntimeError("Validation DataLoader is empty")
    metrics = {
        "val_loss": total_loss / batch_count,
        "val_focal": focal_sum / batch_count,
        "val_size_l1": size_sum / batch_count,
        "val_offset_l1": offset_sum / batch_count,
        "val_num_pos": pos_sum / batch_count,
    }
    metrics.update(_target_stat_metrics(target_stats, prefix="val_"))
    return metrics


# ── Validation AP (image-level detection metrics) ──────────────────────────────


@torch.no_grad()
def validate_ap(
    model: nn.Module,
    data_root: str | Path,
    manifest_path: Path,
    *,
    device: torch.device,
    image_size: int,
    threshold: float = 0.05,
    topk: int = 100,
    peak_kernel: int = 3,
    class_chunk_size: int = 4,
    batch_size: int = 1,
    num_workers: int = 0,
    limit_images: int = 0,
    precision: str = "fp32",
    vector_n_max: int = 0,
) -> dict[str, float]:
    """Run image-level detection evaluation on the val split.

    Returns AP50 and AP50:95 computed with the same decoder and metrics used
    by evaluate.py.  This is separate from query-level val loss so both
    checkpoint selection criteria are available without running a second
    process.
    """
    from src.models import FloorPlanDetector

    model.eval()
    val_dataset = FloorPlanImageDataset(
        data_root,
        split="val",
        image_size=image_size,
        manifest_path=manifest_path,
        vector_branch=vector_n_max > 0,
        vector_n_max=vector_n_max if vector_n_max > 0 else 1024,
    )
    selected_dataset: Dataset = val_dataset
    if limit_images > 0 and limit_images < len(val_dataset):
        selected_dataset = Subset(val_dataset, list(range(limit_images)))
    loader = DataLoader(
        selected_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=image_collate_fn,
        pin_memory=device.type == "cuda",
    )
    decoder = CenterNetDecoder(
        stride=getattr(model, "output_stride", 8),
        threshold=threshold,
        topk=topk,
        peak_kernel=peak_kernel,
    )
    predictions: list[dict] = []
    targets: list[dict] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=device.type == "cuda")
        stroke_kwargs: dict[str, Any] = {}
        if "stroke_tokens" in batch:
            stroke_kwargs["stroke_tokens"] = batch["stroke_tokens"].to(
                device, non_blocking=device.type == "cuda"
            )
            stroke_kwargs["stroke_mask"] = batch["stroke_mask"].to(
                device, non_blocking=device.type == "cuda"
            )
        with _autocast_context(device, precision):
            if isinstance(model, FloorPlanDetector):
                outputs = model(images, class_chunk_size=class_chunk_size, **stroke_kwargs)
            else:
                outputs = model(images)
        predictions.extend(
            decoder(
                outputs,
                batch["image_ids"],
                image_size=(images.shape[-2], images.shape[-1]),
            )
        )
        targets.extend(
            {
                "image_id": image_id,
                "boxes": boxes.detach().cpu(),
                "labels": labels.detach().cpu(),
            }
            for image_id, boxes, labels in zip(
                batch["image_ids"], batch["boxes"], batch["labels"]
            )
        )
    metrics = compute_coco_metrics(predictions, targets, class_ids=range(NUM_CLASSES))
    return {
        "val_ap50": metrics["AP50"],
        "val_ap50_95": metrics["AP50:95"],
        "val_ap_num_images": float(len(selected_dataset)),
    }


# ── Scheduler and checkpoint wiring ────────────────────────────────────────────


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    steps_per_epoch: int,
) -> tuple[Any, int, int]:
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive")
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = min(args.warmup_steps, max(1, total_steps - 1))
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=args.warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_steps),
        eta_min=args.lr * args.min_lr_ratio,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )
    return scheduler, total_steps, warmup_steps


def _build_runtime_config(
    args: argparse.Namespace,
    *,
    device: torch.device,
    train_size: int,
    val_size: int,
    steps_per_epoch: int,
    total_steps: int,
    warmup_steps: int,
    train_metadata_fingerprint: str,
    val_metadata_fingerprint: str,
    vector_branch: bool = False,
    vector_n_max: int = 0,
) -> dict[str, Any]:
    """Capture every setting that affects an exact epoch-boundary resume."""

    return {
        "training_script_version": 3,
        "reproducibility": {
            "seed": args.seed,
            "deterministic": args.deterministic,
            "device_type": device.type,
        },
        "data": {
            "train_queries": train_size,
            "val_queries": val_size,
            "train_metadata_fingerprint": train_metadata_fingerprint,
            "val_metadata_fingerprint": val_metadata_fingerprint,
            "limit_train_samples": args.limit_train_samples,
            "limit_val_samples": args.limit_val_samples,
            "neg_queries_per_pos": args.neg_queries_per_pos,
            "negative_query_seed": args.seed,
        },
        "loader": {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "sampler": args.sampler,
            "balance_power": args.balance_power,
            "cache_images": args.cache_images,
            "steps_per_epoch": steps_per_epoch,
            "vector_branch": vector_branch,
            "vector_n_max": vector_n_max,
        },        "optimization": {
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": 1e-4,
            "focal_weight": args.focal_weight,
            "size_weight": args.size_weight,
            "offset_weight": args.offset_weight,
            "grad_clip": args.grad_clip,
            "precision": args.precision,
            "grad_scaler": args.precision == "fp16",
        },
        "scheduler": {
            "kind": "linear_warmup_then_cosine",
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "warmup_start_factor": args.warmup_start_factor,
            "min_lr_ratio": args.min_lr_ratio,
        },
        "validation": {
            "val_ap_interval": args.val_ap_interval,
            "val_ap_chunk_size": args.val_ap_chunk_size,
            "limit_val_ap_images": args.limit_val_ap_images,
            "decoder_threshold": 0.05,
            "decoder_topk": 100,
            "decoder_peak_kernel": 3,
        },
        "checkpoint_interval": _CHECKPOINT_INTERVAL,
    }


def validate_exact_resume_compatibility(
    checkpoint: Mapping[str, Any],
    *,
    runtime_config: Mapping[str, Any],
    split_fingerprint: str,
    metadata_fingerprint: str,
) -> None:
    """Reject exact resumes whose data or schedule-affecting settings changed."""

    if checkpoint.get("split_manifest_fingerprint") != split_fingerprint:
        raise CheckpointError(
            "Split manifest fingerprint mismatch: "
            f"checkpoint={checkpoint.get('split_manifest_fingerprint')}, "
            f"current={split_fingerprint}"
        )
    if checkpoint.get("metadata_fingerprint") != metadata_fingerprint:
        raise CheckpointError(
            "Metadata fingerprint mismatch: "
            f"checkpoint={checkpoint.get('metadata_fingerprint')}, "
            f"current={metadata_fingerprint}"
        )

    checkpoint_runtime = checkpoint.get("runtime_config")
    if checkpoint_runtime != dict(runtime_config):
        checkpoint_sections = (
            set(checkpoint_runtime) if isinstance(checkpoint_runtime, Mapping) else set()
        )
        current_sections = set(runtime_config)
        changed_sections = sorted(
            key
            for key in checkpoint_sections | current_sections
            if not isinstance(checkpoint_runtime, Mapping)
            or checkpoint_runtime.get(key) != runtime_config.get(key)
        )
        raise CheckpointError(
            "Runtime configuration mismatch for exact resume; changed sections: "
            f"{changed_sections}. Use --weights-only to start a new optimizer/schedule."
        )


def build_training_checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any | None = None,
    epoch: int,
    global_step: int,
    best_metric: float,
    metrics: Mapping[str, Any],
    model_config: ModelConfig,
    preset: str,
    runtime_config: Mapping[str, Any],
    split_fingerprint: str,
    metadata_fingerprint: str,
    data_generator: torch.Generator,
) -> dict[str, Any]:
    """Build the complete schema-v2 payload shared by all checkpoint files."""

    return build_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=epoch,
        global_step=global_step,
        best_metric=best_metric,
        metrics=metrics,
        model_config=model_config,
        preset=preset,
        runtime_config=runtime_config,
        class_names=CLASS_NAMES,
        output_stride=model_config.output_stride,
        split_manifest_fingerprint=split_fingerprint,
        metadata_fingerprint=metadata_fingerprint,
        data_generator=data_generator,
    )


# ── Main ───────────────────────────────────────────────────────────────────────


def _force_utf8_stdout() -> None:
    """Avoid UnicodeEncodeError on legacy Windows consoles (cp1252)."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def main(args: argparse.Namespace) -> int:
    _force_utf8_stdout()
    _validate_args(args)
    seed_everything(args.seed, deterministic=args.deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
    precision = _resolve_precision(args.precision, device)
    args.precision = precision
    print(f"Precision: {precision}")

    checkpoint_path: Path | None = None
    checkpoint: dict[str, Any] | None = None
    if args.resume:
        checkpoint_path = Path(args.resume)
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")

    model_config = resolve_model_config_from_args(args)
    manifest_path, _manifest, manifest_fingerprint = _resolve_training_manifest(
        args.data_root,
        args.manifest,
    )
    print(f"Preset: {args.preset} | Manifest: {manifest_path}")

    # Both query datasets are derived from the same image-level manifest. Test is
    # intentionally untouched during training and validation.
    vector_branch = bool(getattr(model_config.vector, "enabled", False))
    vector_n_max = int(getattr(model_config.vector, "n_max", 1024))
    train_base = FloorPlanQueryDataset(
        args.data_root,
        split="train",
        image_size=model_config.image_size,
        output_stride=model_config.output_stride,
        manifest_path=manifest_path,
        neg_queries_per_pos=args.neg_queries_per_pos,
        neg_seed=args.seed,
        cache_images=args.cache_images,
        vector_branch=vector_branch,
        vector_n_max=vector_n_max,
        vector_seed=args.seed,
    )
    val_base = FloorPlanQueryDataset(
        args.data_root,
        split="val",
        image_size=model_config.image_size,
        output_stride=model_config.output_stride,
        manifest_path=manifest_path,
        neg_queries_per_pos=0,
        cache_images=args.cache_images,
        vector_branch=vector_branch,
        vector_n_max=vector_n_max,
    )
    if vector_branch:
        with_strokes = 0
        for record in train_base.records:
            metadata_path = Path(record.metadata_path)
            if not metadata_path.is_absolute():
                metadata_path = Path(args.data_root) / metadata_path
            tokens = train_base._load_stroke_tokens(metadata_path)
            if tokens.shape[0] > 0:
                with_strokes += 1
        print(
            f"Vector branch enabled: {with_strokes}/{len(train_base.records)} train images "
            f"carry stroke tokens (n_max={vector_n_max})"
        )
    if train_base.split_manifest_fingerprint not in {None, manifest_fingerprint}:
        raise RuntimeError("Training dataset loaded a different split manifest fingerprint")
    if val_base.split_manifest_fingerprint not in {None, manifest_fingerprint}:
        raise RuntimeError("Validation dataset loaded a different split manifest fingerprint")

    # P2-A: Optional preflight source validation. Re-hashes PNG and SVG for
    # every train/val image to detect source drift before spending compute.
    if args.validate_sources:
        print("Validating source files (--validate-sources)...")
        source_errors = _validate_source_records(
            args.data_root,
            [*train_base.records, *val_base.records],
        )
        if source_errors:
            for message in source_errors[:20]:
                print(f"  [source-error] {message}")
            if len(source_errors) > 20:
                print(f"  ... and {len(source_errors) - 20} more errors")
            raise RuntimeError(
                f"Source validation failed with {len(source_errors)} error(s). "
                "Fix source files or rebuild metadata with scripts/data/build_dataset.py."
            )
        print(f"  Source validation passed for {len(train_base.records) + len(val_base.records):,} images.")

    _print_class_stats(train_base)
    if args.neg_queries_per_pos > 0:
        print(
            f"Negative query sampling: {args.neg_queries_per_pos} absent-class queries per positive "
            f"| Positive: {train_base.num_positive_queries:,} | Negative: {train_base.num_negative_queries:,}"
        )

    train_dataset = _maybe_subset(train_base, args.limit_train_samples)
    val_dataset = _maybe_subset(val_base, args.limit_val_samples)
    data_generator = make_generator(args.seed)
    pin_memory = device.type == "cuda"
    train_loader = _make_train_loader(
        train_dataset,
        train_base,
        args,
        pin_memory=pin_memory,
        data_generator=data_generator,
    )
    val_loader = _make_val_loader(
        val_dataset,
        args,
        pin_memory=pin_memory,
        data_generator=data_generator,
    )
    print(
        f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | "
        f"Steps/epoch: {len(train_loader)} | "
        f"Target: {train_base.output_size}×{train_base.output_size} | "
        f"Sampler: {args.sampler}"
    )

    model = build_model(model_config).to(device)
    materialized_conditioners = materialize_pretrained_conditioners(model)
    if materialized_conditioners:
        print(
            "Materialized pretrained conditioner(s) before optimizer setup: "
            + ", ".join(name or "<root>" for name in materialized_conditioners)
        )
    parameter_count = sum(parameter.numel() for parameter in model.parameters()) / 1e6
    print(f"Model params: {parameter_count:.1f}M")

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler, total_steps, warmup_steps = _build_scheduler(
        optimizer,
        args,
        len(train_loader),
    )
    scaler = _build_grad_scaler(device, precision)
    print(
        f"Loss weights: focal={args.focal_weight:g} | size={args.size_weight:g} | "
        f"offset={args.offset_weight:g} | LR={args.lr:.2e} | "
        f"warmup_steps={warmup_steps}"
    )

    combined_metadata_fingerprint = image_index_fingerprint(
        [*train_base.records, *val_base.records]
    )
    runtime_config = _build_runtime_config(
        args,
        device=device,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        steps_per_epoch=len(train_loader),
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        train_metadata_fingerprint=train_base.metadata_fingerprint,
        val_metadata_fingerprint=val_base.metadata_fingerprint,
        vector_branch=vector_branch,
        vector_n_max=vector_n_max,
    )

    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")
    best_val_ap50_95 = -1.0
    if checkpoint is not None:
        assert checkpoint_path is not None
        if not args.weights_only:
            validate_exact_resume_compatibility(
                checkpoint,
                runtime_config=runtime_config,
                split_fingerprint=manifest_fingerprint,
                metadata_fingerprint=combined_metadata_fingerprint,
            )
        resume_state = restore_training_state(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            data_generator=data_generator,
            expected_model_config=model_config,
            expected_class_names=CLASS_NAMES,
            expected_output_stride=model_config.output_stride,
            weights_only=args.weights_only,
            strict=True,
        )
        start_epoch = int(resume_state["start_epoch"])
        global_step = int(resume_state["global_step"])
        restored_best = resume_state["best_metric"]
        if restored_best is not None:
            best_val_loss = float(restored_best)
        if not args.weights_only:
            stored_metrics = checkpoint.get("metrics")
            if isinstance(stored_metrics, Mapping):
                stored_best_ap = stored_metrics.get(
                    "best_val_ap50_95",
                    stored_metrics.get("val_ap50_95"),
                )
                if stored_best_ap is not None:
                    best_val_ap50_95 = float(stored_best_ap)
        if args.weights_only:
            print(f"Loaded model weights from {checkpoint_path}; optimizer/scheduler restarted")
        else:
            print(
                f"Resumed exact state from {checkpoint_path} at epoch {start_epoch} "
                f"and global_step {global_step}"
            )

    checkpoint_dir = Path(args.ckpt_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if start_epoch > args.epochs:
        print(
            f"Checkpoint already completed epoch {start_epoch - 1}; "
            f"requested epochs={args.epochs}. Nothing to run."
        )
        return 0

    print("\n" + "=" * 60)
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n[Epoch {epoch}/{args.epochs}]")
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            epoch,
            focal_w=args.focal_weight,
            size_w=args.size_weight,
            offset_w=args.offset_weight,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
            precision=precision,
            scaler=scaler,
        )
        global_step += len(train_loader)
        val_metrics = validate(
            model,
            val_loader,
            device,
            focal_w=args.focal_weight,
            size_w=args.size_weight,
            offset_w=args.offset_weight,
            precision=precision,
        )

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"  => Train loss: {train_metrics['loss']:.4f} "
            f"(focal={train_metrics['focal']:.4f}, size={train_metrics['size_l1']:.4f}, "
            f"offset={train_metrics['offset_l1']:.4f}) | "
            f"Val loss: {val_metrics['val_loss']:.4f} "
            f"(focal={val_metrics['val_focal']:.4f}, "
            f"size={val_metrics['val_size_l1']:.4f}, "
            f"offset={val_metrics['val_offset_l1']:.4f}) | "
            f"LR: {current_lr:.2e}"
        )
        print(
            "  => Target collisions: "
            f"train={int(train_metrics['target_collisions'])}/"
            f"{int(train_metrics['target_valid_boxes'])} "
            f"({train_metrics['target_collision_rate']:.2%}), "
            f"val={int(val_metrics['val_target_collisions'])}/"
            f"{int(val_metrics['val_target_valid_boxes'])} "
            f"({val_metrics['val_target_collision_rate']:.2%})"
        )

        is_best = val_metrics["val_loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["val_loss"]

        # P1-A: Periodic val AP evaluation. Runs every --val-ap-interval epochs
        # (default 5). Disabled when val_ap_interval=0. This is the prespecified
        # checkpoint selection metric for held-out test evaluation.
        val_ap_metrics: dict[str, float] = {}
        val_ap_interval = getattr(args, "val_ap_interval", _VAL_AP_INTERVAL_DEFAULT)
        if val_ap_interval > 0 and epoch % val_ap_interval == 0:
            print("  => Computing val AP (image-level)...")
            val_ap_metrics = validate_ap(
                model,
                args.data_root,
                manifest_path,
                device=device,
                image_size=model_config.image_size,
                class_chunk_size=args.val_ap_chunk_size,
                batch_size=1,
                num_workers=0,
                limit_images=args.limit_val_ap_images,
                precision=precision,
                vector_n_max=vector_n_max if vector_branch else 0,
            )
            print(
                f"  => Val AP50={val_ap_metrics['val_ap50']:.4f} | "
                f"AP50:95={val_ap_metrics['val_ap50_95']:.4f}"
            )

        is_best_ap = (
            val_ap_metrics.get("val_ap50_95", -1.0) > best_val_ap50_95
        )
        if is_best_ap:
            best_val_ap50_95 = val_ap_metrics["val_ap50_95"]

        checkpoint_metrics = {
            "train": train_metrics,
            "val": val_metrics,
            "learning_rate": current_lr,
            "best_val_loss": best_val_loss,
            "best_val_ap50_95": best_val_ap50_95,
            **val_ap_metrics,
        }
        payload = build_training_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            best_metric=best_val_loss,
            metrics=checkpoint_metrics,
            model_config=model_config,
            preset=args.preset,
            runtime_config=runtime_config,
            split_fingerprint=manifest_fingerprint,
            metadata_fingerprint=combined_metadata_fingerprint,
            data_generator=data_generator,
        )

        if is_best:
            best_path = save_checkpoint(checkpoint_dir / "best.pt", payload)
            print(
                f"  => Saved best checkpoint "
                f"(val_loss={best_val_loss:.4f}) → {best_path}"
            )
        if is_best_ap:
            best_ap_path = save_checkpoint(checkpoint_dir / "best_val_ap.pt", payload)
            print(
                f"  => Saved best val-AP checkpoint "
                f"(AP50:95={best_val_ap50_95:.4f}) → {best_ap_path}"
            )
        if epoch % _CHECKPOINT_INTERVAL == 0:
            periodic_path = save_checkpoint(
                checkpoint_dir / f"epoch_{epoch:03d}.pt",
                payload,
            )
            print(f"  => Saved periodic checkpoint → {periodic_path}")
        save_checkpoint(checkpoint_dir / "last.pt", payload)

    print(
        f"\nTraining done! Best val_loss: {best_val_loss:.4f} | "
        f"Best val AP50:95: {best_val_ap50_95:.4f}"
    )
    return 0


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FloorPlanCAD Detector Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        "--data_root",
        dest="data_root",
        default="./data/FloorPlanCAD_original",
    )
    parser.add_argument(
        "--ckpt-dir",
        "--ckpt_dir",
        dest="ckpt_dir",
        default="./checkpoints",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Image-level split manifest shared by train and val",
    )
    parser.add_argument(
        "--preset",
        choices=list_model_presets(),
        default="floorplan_base",
        help="Named model configuration",
    )
    parser.add_argument(
        "--image-size",
        "--image_size",
        dest="image_size",
        type=int,
        default=None,
        help="Override the preset input size",
    )
    parser.add_argument(
        "--output-stride",
        "--output_stride",
        dest="output_stride",
        type=int,
        default=None,
        help="Legacy override; only stride 8 is supported",
    )
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=4)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=4)
    parser.add_argument(
        "--cache-images",
        "--cache_images",
        dest="cache_images",
        action="store_true",
        help=(
            "Cache decoded full-resolution images and parsed metadata per worker "
            "so an image is decoded once and reused across its queries. Enables "
            "persistent workers. Uses more RAM; best with few workers on datasets "
            "that fit in memory."
        ),
    )
    parser.add_argument(
        "--model-dim",
        "--model_dim",
        dest="model_dim",
        type=int,
        default=None,
        help="Legacy override for preset model_dim",
    )
    parser.add_argument(
        "--depth-per-class",
        "--depth_per_class",
        dest="depth_per_class",
        type=int,
        default=None,
        help="Legacy override for preset depth_per_class",
    )
    parser.add_argument(
        "--fusion-mode",
        "--fusion_mode",
        dest="fusion_mode",
        choices=_FUSION_MODES,
        default=None,
        help="Legacy override for preset fusion_mode",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--focal-weight",
        "--focal_weight",
        dest="focal_weight",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--size-weight",
        "--size_weight",
        dest="size_weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--offset-weight",
        "--offset_weight",
        dest="offset_weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--warmup-steps",
        "--warmup_steps",
        dest="warmup_steps",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--warmup-start-factor",
        "--warmup_start_factor",
        dest="warmup_start_factor",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--min-lr-ratio",
        "--min_lr_ratio",
        dest="min_lr_ratio",
        type=float,
        default=0.01,
    )
    parser.add_argument("--grad-clip", "--grad_clip", dest="grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--sampler",
        choices=["shuffle", "balanced"],
        default="balanced",
    )
    parser.add_argument(
        "--balance-power",
        "--balance_power",
        dest="balance_power",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--neg-queries-per-pos",
        "--neg_queries_per_pos",
        dest="neg_queries_per_pos",
        type=int,
        default=0,
        help=(
            "Number of absent-class (negative) queries to sample per positive query. "
            "0 disables negative sampling (original behaviour). "
            "Recommended: 1 or 2 to teach the model to output empty predictions."
        ),
    )
    parser.add_argument(
        "--limit-train-samples",
        "--limit_train_samples",
        dest="limit_train_samples",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--limit-val-samples",
        "--limit_val_samples",
        dest="limit_val_samples",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--log-interval",
        "--log_interval",
        dest="log_interval",
        type=int,
        default=20,
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Request deterministic PyTorch algorithms",
    )
    parser.add_argument(
        "--precision",
        choices=["auto", "fp32", "bf16", "fp16"],
        default="fp32",
        help=(
            "Training/inference precision. BF16 is recommended for attention models "
            "on supported CUDA GPUs; FP16 uses gradient scaling."
        ),
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Schema-v2 checkpoint for exact resume, or for --weights-only loading",
    )
    parser.add_argument(
        "--weights-only",
        "--weights_only",
        dest="weights_only",
        action="store_true",
        help="Load --resume model weights but restart optimizer, scheduler, and RNG state",
    )
    # P1-A: Periodic val AP evaluation (image-level AP50 / AP50:95).
    parser.add_argument(
        "--val-ap-interval",
        "--val_ap_interval",
        dest="val_ap_interval",
        type=int,
        default=_VAL_AP_INTERVAL_DEFAULT,
        help=(
            "Run image-level val AP evaluation every N epochs and save "
            "best_val_ap.pt when AP50:95 improves. 0 disables AP checkpointing."
        ),
    )
    parser.add_argument(
        "--val-ap-chunk-size",
        "--val_ap_chunk_size",
        dest="val_ap_chunk_size",
        type=int,
        default=4,
        help="Class chunk size for all-class inference during val AP evaluation.",
    )
    parser.add_argument(
        "--limit-val-ap-images",
        "--limit_val_ap_images",
        dest="limit_val_ap_images",
        type=int,
        default=0,
        help=(
            "Limit image-level val AP to the first N images for pilot runs. "
            "0 evaluates the full val split and should be used for official selection."
        ),
    )
    # P2-A: Preflight source validation — re-hashes PNG/SVG before training.
    parser.add_argument(
        "--validate-sources",
        "--validate_sources",
        dest="validate_sources",
        action="store_true",
        help=(
            "Before training, re-hash every PNG and SVG and compare against "
            "metadata source hashes. Raises an error if any file has changed "
            "since the metadata was built. Slow for large datasets."
        ),
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main(build_argument_parser().parse_args()))
