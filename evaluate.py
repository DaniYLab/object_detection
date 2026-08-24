"""Evaluate internal checkpoints or external detection JSON without training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from src.data.constants import CLASS_NAMES, NUM_CLASSES
from src.data.dataset import FloorPlanImageDataset, image_collate_fn
from src.data.splits import (
    image_index_fingerprint,
    load_split_manifest,
    resolve_split_manifest_path,
)
from src.evaluation import (
    CenterNetDecoder,
    compute_coco_metrics,
    read_predictions,
    write_predictions,
)
from src.models import FloorPlanDetector, ModelConfig, build_model
from src.training.checkpoint import CheckpointError, load_checkpoint, restore_training_state


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _validate_manifest_provenance(
    checkpoint: dict[str, Any],
    manifest_fingerprint: str | None,
    *,
    allow_mismatch: bool,
) -> None:
    """Reject checkpoints trained against another or unspecified manifest."""

    if manifest_fingerprint is None:
        return
    checkpoint_fingerprint = checkpoint.get("split_manifest_fingerprint")
    if checkpoint_fingerprint == manifest_fingerprint:
        return
    message = (
        "Split manifest fingerprint mismatch.\n"
        f"  Checkpoint trained with: {checkpoint_fingerprint}\n"
        f"  Current manifest:        {manifest_fingerprint}\n"
        "The checkpoint was trained on a different split or metadata benchmark. "
        "Pass --allow-manifest-mismatch to override (not recommended)."
    )
    if not allow_mismatch:
        raise CheckpointError(message)
    print(f"WARNING: {message}")


def _load_model(
    checkpoint_path: Path,
    *,
    device: torch.device,
    image_size: int,
    manifest_fingerprint: str | None = None,
    allow_manifest_mismatch: bool = False,
) -> tuple[torch.nn.Module, dict[str, Any], ModelConfig]:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    config_value = checkpoint.get("model_config")
    if not isinstance(config_value, dict):
        raise CheckpointError("Checkpoint does not contain a serializable model_config")
    config = ModelConfig.from_dict(config_value)
    if config.image_size != image_size:
        raise ValueError(
            f"Evaluation image_size={image_size} does not match checkpoint "
            f"image_size={config.image_size}"
        )

    # P1-B: Enforce split manifest provenance to prevent silently comparing
    # checkpoints trained on different splits or metadata policies.
    _validate_manifest_provenance(
        checkpoint,
        manifest_fingerprint,
        allow_mismatch=allow_manifest_mismatch,
    )

    model = build_model(config)
    restore_training_state(
        checkpoint,
        model=model,
        expected_model_config=config,
        expected_class_names=CLASS_NAMES,
        expected_output_stride=config.output_stride,
        weights_only=True,
        strict=True,
    )
    model.to(device).eval()
    return model, checkpoint, config


def _targets_from_loader(loader: DataLoader) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for batch in loader:
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
    return targets


def _align_external_predictions(
    predictions: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    *,
    allow_filter: bool,
) -> list[dict[str, Any]]:
    target_ids = {target["image_id"] for target in targets}
    prediction_ids = {prediction["image_id"] for prediction in predictions}
    unknown = prediction_ids.difference(target_ids)
    if unknown and not allow_filter:
        examples = sorted(str(image_id) for image_id in unknown)[:5]
        raise ValueError(
            "Prediction JSON contains image IDs outside the selected split: "
            f"{examples}"
        )
    return [
        prediction for prediction in predictions if prediction["image_id"] in target_ids
    ]


@torch.inference_mode()
def _predict(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    decoder: CenterNetDecoder,
    class_chunk_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=device.type == "cuda")
        if isinstance(model, FloorPlanDetector):
            outputs = model(images, class_chunk_size=class_chunk_size)
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
    return predictions, targets


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute AP50 and AP50:95 on held-out image-level detections"
    )
    parser.add_argument("--data-root", "--data_root", dest="data_root", default="./data/FloorPlanCAD_original")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--predictions-json")
    parser.add_argument("--report", default=None, help="Output metric JSON path")
    parser.add_argument("--save-predictions", default=None)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="cpu, cuda, or omitted for auto")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--peak-kernel", type=int, default=3)
    parser.add_argument("--class-chunk-size", type=int, default=4)
    parser.add_argument("--limit-images", type=int, default=0)
    parser.add_argument("--strict-metadata", action="store_true")
    parser.add_argument(
        "--allow-manifest-mismatch",
        "--allow_manifest_mismatch",
        dest="allow_manifest_mismatch",
        action="store_true",
        help=(
            "Skip the manifest fingerprint check when evaluating a checkpoint "
            "that was trained on a different split. Not recommended for final reporting."
        ),
    )
    return parser


def _force_utf8_stdout() -> None:
    """Avoid UnicodeEncodeError on legacy Windows consoles (cp1252)."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
    args = build_argument_parser().parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    if args.limit_images < 0:
        raise ValueError("limit_images cannot be negative")
    if args.class_chunk_size <= 0:
        raise ValueError("class_chunk_size must be positive")

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    manifest_path = resolve_split_manifest_path(args.data_root, args.manifest)
    manifest_fingerprint = None
    if manifest_path is not None:
        manifest_fingerprint = load_split_manifest(manifest_path).get("fingerprint")

    dataset = FloorPlanImageDataset(
        args.data_root,
        split=args.split,
        image_size=args.image_size,
        manifest_path=args.manifest,
        strict_metadata=args.strict_metadata,
    )
    selected_dataset = dataset
    if args.limit_images and args.limit_images < len(dataset):
        selected_dataset = Subset(dataset, list(range(args.limit_images)))
    loader = DataLoader(
        selected_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=image_collate_fn,
        pin_memory=device.type == "cuda",
    )

    source_metadata: dict[str, Any]
    if args.predictions_json:
        predictions = read_predictions(args.predictions_json)
        targets = _targets_from_loader(loader)
        predictions = _align_external_predictions(
            predictions,
            targets,
            allow_filter=bool(args.limit_images),
        )
        source_metadata = {
            "type": "external_predictions",
            "path": str(Path(args.predictions_json)),
            "sha256": _sha256(Path(args.predictions_json)),
        }
    else:
        checkpoint_path = Path(args.checkpoint)
        model, checkpoint, config = _load_model(
            checkpoint_path,
            device=device,
            image_size=args.image_size,
            manifest_fingerprint=manifest_fingerprint,
            allow_manifest_mismatch=args.allow_manifest_mismatch,
        )
        decoder = CenterNetDecoder(
            stride=config.output_stride,
            threshold=args.threshold,
            topk=args.topk,
            peak_kernel=args.peak_kernel,
        )
        predictions, targets = _predict(
            model,
            loader,
            device=device,
            decoder=decoder,
            class_chunk_size=args.class_chunk_size,
        )
        source_metadata = {
            "type": "checkpoint",
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "preset": checkpoint.get("preset"),
            "model_config": config.to_dict(),
        }
        if args.save_predictions:
            write_predictions(
                args.save_predictions,
                predictions,
                metadata={
                    "checkpoint_sha256": source_metadata["sha256"],
                    "split": args.split,
                    "class_names": list(CLASS_NAMES),
                },
            )

    metrics = compute_coco_metrics(
        predictions,
        targets,
        class_ids=range(NUM_CLASSES),
    )
    for class_id, values in metrics["per_class"].items():
        values["class_name"] = CLASS_NAMES[class_id] if class_id < NUM_CLASSES else None

    report = {
        "schema": {"name": "floorplan_detection_evaluation", "version": 1},
        "source": source_metadata,
        "data": {
            "root": str(Path(args.data_root)),
            "split": args.split,
            "num_images": len(selected_dataset),
            "manifest": str(manifest_path) if manifest_path is not None else None,
            "manifest_fingerprint": manifest_fingerprint,
            "image_index_fingerprint": image_index_fingerprint(dataset.records),
            "class_names": list(CLASS_NAMES),
        },
        "decoder": {
            "threshold": args.threshold,
            "topk_per_class": args.topk,
            "peak_kernel": args.peak_kernel,
            "output_stride": 8,
        },
        "metrics": metrics,
    }

    output_path = Path(args.report or f"outputs/evaluation_{args.split}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_ready(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Images={metrics['num_images']} | AP50={metrics['AP50']:.4f} | "
        f"AP50:95={metrics['AP50:95']:.4f}"
    )
    print(f"Report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
