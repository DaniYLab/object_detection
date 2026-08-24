"""Generate a reproducible FloorPlanCAD data-audit report from metadata v2."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.constants import CLASS_NAMES
from src.data.splits import load_split_manifest
from src.data.targets import gaussian_radius


def _quantiles(values: Iterable[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None}

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": percentile(0.25),
        "median": percentile(0.50),
        "p75": percentile(0.75),
        "max": ordered[-1],
    }


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _size_bucket(area: float) -> str:
    if area < 32.0**2:
        return "small"
    if area < 96.0**2:
        return "medium"
    return "large"


def audit_dataset(
    data_root: Path,
    manifest_path: Path,
    *,
    image_size: int = 512,
    output_stride: int = 8,
) -> dict[str, Any]:
    manifest = load_split_manifest(manifest_path)
    output_size = image_size // output_stride

    split_summary: dict[str, Any] = {}
    global_class_counts: Counter[str] = Counter()
    class_boxes: Counter[str] = Counter()
    class_collisions: Counter[str] = Counter()
    size_boxes: Counter[str] = Counter()
    size_collisions: Counter[str] = Counter()
    radii: list[float] = []
    radii_by_class: dict[str, list[float]] = defaultdict(list)
    aspect_ratios: list[float] = []
    transform_warning_count = 0
    transform_affected_images = 0
    missing_semantic_paths = 0
    unknown_semantic_paths = 0
    empty_or_invalid_paths = 0

    for split_name in ("train", "val", "test"):
        records = manifest.get("splits", {}).get(split_name, [])
        split_class_counts: Counter[str] = Counter()
        split_instances = 0
        for record in records:
            metadata_path = _resolve(data_root, str(record["metadata_path"]))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            width, height = (float(value) for value in metadata["image_size"])
            aspect_ratios.append(width / height)
            stats = metadata.get("stats", {})
            warnings = stats.get("warnings", []) if isinstance(stats, dict) else []
            image_transform_warnings = sum(
                1 for warning in warnings if "transform" in str(warning).lower()
            )
            transform_warning_count += image_transform_warnings
            transform_affected_images += int(image_transform_warnings > 0)
            missing_semantic_paths += int(stats.get("paths_missing_semantic_id", 0))
            unknown_semantic_paths += int(stats.get("paths_unknown_semantic_id", 0))
            empty_or_invalid_paths += int(stats.get("paths_invalid", 0))

            centers_by_class: dict[str, list[tuple[tuple[int, int], str]]] = defaultdict(list)
            for instance in metadata.get("instances", []):
                class_name = str(instance["class"])
                x0, y0, x1, y1 = (float(value) for value in instance["bbox_px"])
                resized_width = (x1 - x0) * image_size / width
                resized_height = (y1 - y0) * image_size / height
                area = resized_width * resized_height
                bucket = _size_bucket(area)
                center_x = min(
                    output_size - 1,
                    max(0, math.floor(((x0 + x1) / 2.0) * output_size / width)),
                )
                center_y = min(
                    output_size - 1,
                    max(0, math.floor(((y0 + y1) / 2.0) * output_size / height)),
                )
                centers_by_class[class_name].append(((center_x, center_y), bucket))

                radius = max(
                    0,
                    int(
                        gaussian_radius(
                            (
                                math.ceil(resized_height / output_stride),
                                math.ceil(resized_width / output_stride),
                            ),
                            min_overlap=0.7,
                        )
                    ),
                )
                radii.append(float(radius))
                radii_by_class[class_name].append(float(radius))
                split_class_counts[class_name] += 1
                global_class_counts[class_name] += 1
                class_boxes[class_name] += 1
                size_boxes[bucket] += 1
                split_instances += 1

            for class_name, centers in centers_by_class.items():
                seen: set[tuple[int, int]] = set()
                for center, bucket in centers:
                    if center in seen:
                        class_collisions[class_name] += 1
                        size_collisions[bucket] += 1
                    else:
                        seen.add(center)

        split_summary[split_name] = {
            "images": len(records),
            "instances": split_instances,
            "classes_with_gt": sum(1 for count in split_class_counts.values() if count > 0),
            "class_instance_counts": {
                name: split_class_counts.get(name, 0) for name in CLASS_NAMES
            },
        }

    per_class_collision = {
        name: {
            "boxes": class_boxes.get(name, 0),
            "collisions": class_collisions.get(name, 0),
            "collision_rate": (
                class_collisions.get(name, 0) / class_boxes[name]
                if class_boxes.get(name, 0)
                else 0.0
            ),
        }
        for name in CLASS_NAMES
    }
    per_size_collision = {
        bucket: {
            "boxes": size_boxes.get(bucket, 0),
            "collisions": size_collisions.get(bucket, 0),
            "collision_rate": (
                size_collisions.get(bucket, 0) / size_boxes[bucket]
                if size_boxes.get(bucket, 0)
                else 0.0
            ),
        }
        for bucket in ("small", "medium", "large")
    }
    total_boxes = sum(class_boxes.values())
    total_collisions = sum(class_collisions.values())

    return {
        "schema": {"name": "floorplancad_data_audit", "version": 1},
        "data_root": str(data_root),
        "manifest": str(manifest_path),
        "manifest_fingerprint": manifest.get("fingerprint"),
        "settings": {
            "image_size": image_size,
            "output_stride": output_stride,
            "stuff_policy": "exclude",
            "min_size_px": 8.0,
            "collision_policy": "largest",
            "gaussian_min_overlap": 0.7,
        },
        "splits": split_summary,
        "global": {
            "images": sum(values["images"] for values in split_summary.values()),
            "instances": total_boxes,
            "classes_with_gt": sum(1 for count in global_class_counts.values() if count > 0),
            "zero_instance_classes": [
                name for name in CLASS_NAMES if global_class_counts.get(name, 0) == 0
            ],
            "class_instance_counts": {
                name: global_class_counts.get(name, 0) for name in CLASS_NAMES
            },
        },
        "source_parser": {
            "unsupported_transform_warnings": transform_warning_count,
            "transform_affected_images": transform_affected_images,
            "paths_missing_semantic_id": missing_semantic_paths,
            "paths_unknown_semantic_id": unknown_semantic_paths,
            "paths_invalid_or_empty": empty_or_invalid_paths,
        },
        "image_aspect_ratio": _quantiles(aspect_ratios),
        "gaussian_radius": {
            "overall": _quantiles(radii),
            "per_class": {
                name: _quantiles(radii_by_class.get(name, [])) for name in CLASS_NAMES
            },
        },
        "target_collisions": {
            "boxes": total_boxes,
            "collisions": total_collisions,
            "collision_rate": total_collisions / total_boxes if total_boxes else 0.0,
            "per_class": per_class_collision,
            "per_resized_object_size": per_size_collision,
        },
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="./data/FloorPlanCAD_original")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", default="./outputs/data_audit_report.json")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--output-stride", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    data_root = Path(args.data_root)
    manifest_path = Path(args.manifest) if args.manifest else data_root / "splits.json"
    if args.image_size <= 0 or args.image_size % args.output_stride != 0:
        raise ValueError("image_size must be positive and divisible by output_stride")
    report = audit_dataset(
        data_root,
        manifest_path,
        image_size=args.image_size,
        output_stride=args.output_stride,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Images={report['global']['images']:,} | "
        f"Instances={report['global']['instances']:,} | "
        f"Collisions={report['target_collisions']['collisions']:,} "
        f"({report['target_collisions']['collision_rate']:.2%})"
    )
    print(
        "Unsupported SVG transforms: "
        f"{report['source_parser']['unsupported_transform_warnings']:,} warnings in "
        f"{report['source_parser']['transform_affected_images']:,} images"
    )
    print(f"Report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
