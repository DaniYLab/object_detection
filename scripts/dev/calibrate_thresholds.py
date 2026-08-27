"""Per-class decoder threshold calibration on validation predictions.

The global decoder threshold (0.05) treats all classes alike, but rare
classes hallucinate 40–400 predictions per GT on test (see
``docs/sota_gap_plan.md`` §2). This script picks one threshold per class on
the VALIDATION split only (protocol-compliant), maximising per-class AP50,
and reports the calibrated macro AP. Thresholds are saved as JSON so the
decoder can apply them at test time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.data import NUM_CLASSES
from src.data.dataset import FloorPlanImageDataset, image_collate_fn
from src.evaluation import compute_coco_metrics
from src.data.splits import (
    canonical_split_role,
    image_index_fingerprint,
    load_split_manifest,
    manifest_schema_version,
    resolve_split_manifest_path,
)
from torch.utils.data import DataLoader


def _targets_for_split(data_root: str, manifest: str | None, split: str) -> list[dict]:
    dataset = FloorPlanImageDataset(data_root, split=split, manifest_path=manifest)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=image_collate_fn)
    targets = []
    for batch in loader:
        targets.extend(
            {
                "image_id": image_id,
                "boxes": boxes,
                "labels": labels,
            }
            for image_id, boxes, labels in zip(
                batch["image_ids"], batch["boxes"], batch["labels"]
            )
        )
    return targets


def _validate_prediction_provenance(
    payload: object,
    *,
    role: str,
    manifest_schema: int | None,
    manifest_fingerprint: str | None,
    role_fingerprint: str | None,
) -> None:
    if manifest_schema != 2:
        return
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise ValueError(
            "v2 calibration predictions must include metadata with split and role fingerprints"
        )
    metadata = payload["metadata"]
    expected = {
        "split": role,
        "manifest_fingerprint": manifest_fingerprint,
        "role_fingerprint": role_fingerprint,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if value is not None and metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Prediction provenance does not match calibration role: {mismatches}")


def _apply_thresholds(
    predictions: list[dict], thresholds: dict[int, float]
) -> list[dict]:
    filtered = []
    for record in predictions:
        keep = [
            index
            for index, (score, label) in enumerate(zip(record["scores"], record["labels"]))
            if score >= thresholds.get(int(label), 0.05)
        ]
        filtered.append(
            {
                "image_id": record["image_id"],
                "boxes": [record["boxes"][i] for i in keep],
                "scores": [record["scores"][i] for i in keep],
                "labels": [record["labels"][i] for i in keep],
            }
        )
    return filtered


def _select_class_records(
    records: list[dict], class_id: int, *, predictions: bool
) -> list[dict]:
    """Keep one class before metric computation to avoid repeated all-class work."""
    selected = []
    for record in records:
        keep = [
            index
            for index, label in enumerate(record["labels"])
            if int(label) == class_id
        ]
        boxes = record["boxes"]
        if keep and hasattr(boxes, "__getitem__") and not isinstance(boxes, list):
            selected_boxes = boxes[keep].tolist()
        else:
            selected_boxes = [boxes[i] for i in keep]
        item = {
            "image_id": record["image_id"],
            "boxes": selected_boxes,
            "labels": [class_id] * len(keep),
        }
        if predictions:
            item["scores"] = [record["scores"][i] for i in keep]
        selected.append(item)
    return selected


def _class_ap50(
    predictions: list[dict], targets: list[dict], class_id: int, threshold: float
) -> float:
    """Compute AP50 for one class without evaluating the other 34 classes."""
    filtered_predictions = _apply_thresholds(predictions, {class_id: threshold})
    class_predictions = _select_class_records(
        filtered_predictions, class_id, predictions=True
    )
    class_targets = _select_class_records(targets, class_id, predictions=False)
    metrics = compute_coco_metrics(
        class_predictions, class_targets, class_ids=[class_id], iou_thresholds=[0.5]
    )
    return float(metrics["per_class"][class_id]["AP50"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Saved val predictions JSON")
    parser.add_argument("--data-root", default="./data/FloorPlanCAD_original")
    parser.add_argument("--manifest", default=None)
    parser.add_argument(
        "--split",
        default="val_calibration",
        choices=["val", "val_select", "val_calibration", "test", "test_set"],
        help="Calibration role; v2 defaults to val_calibration and never uses test_set.",
    )
    parser.add_argument("--grid", type=float, nargs="*", default=None,
                        help="Threshold grid (default: 0.05..0.60 step 0.05 plus 0.75/0.85)")
    parser.add_argument("--output", default="outputs/class_thresholds.json")
    args = parser.parse_args(argv)

    if args.split in {"test", "test_set"}:
        raise ValueError("Calibration cannot use the test_set role")
    manifest_path = resolve_split_manifest_path(args.data_root, args.manifest)
    manifest_schema = None
    canonical_role = args.split
    manifest_fingerprint = None
    role_fingerprint = None
    if manifest_path is not None:
        manifest = load_split_manifest(manifest_path)
        manifest_schema = manifest_schema_version(manifest)
        requested_role = "val" if manifest_schema == 1 and args.split == "val_calibration" else args.split
        canonical_role = canonical_split_role(manifest, requested_role)
        manifest_fingerprint = manifest.get("fingerprint")
        role_fingerprint = manifest.get("role_fingerprints", {}).get(canonical_role)
    if canonical_role == "test_set":
        raise ValueError("Calibration cannot use the test_set role")

    grid = args.grid or [round(x, 2) for x in np.arange(0.05, 0.65, 0.05)] + [0.75, 0.85]
    payload = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    _validate_prediction_provenance(
        payload,
        role=canonical_role,
        manifest_schema=manifest_schema,
        manifest_fingerprint=manifest_fingerprint,
        role_fingerprint=role_fingerprint,
    )
    predictions = payload["predictions"] if isinstance(payload, dict) else payload
    targets = _targets_for_split(args.data_root, args.manifest, canonical_role)

    baseline = compute_coco_metrics(predictions, targets, class_ids=range(NUM_CLASSES))
    print(f"Baseline (global 0.05, as decoded): AP50={baseline['AP50']:.4f} AP50:95={baseline['AP50:95']:.4f}")

    # Per-class grid search: maximise that class's AP50.
    per_class_best: dict[int, dict] = {}
    for class_id in range(NUM_CLASSES):
        best = {"threshold": 0.05, "AP50": -1.0}
        for threshold in grid:
            ap = _class_ap50(predictions, targets, class_id, threshold)
            if ap > best["AP50"]:
                best = {"threshold": threshold, "AP50": ap}
        per_class_best[class_id] = best

    calibrated_thresholds = {str(cid): entry["threshold"] for cid, entry in per_class_best.items()}
    filtered = _apply_thresholds(predictions, {int(k): v for k, v in calibrated_thresholds.items()})
    calibrated = compute_coco_metrics(filtered, targets, class_ids=range(NUM_CLASSES))
    print(f"Calibrated (per-class):            AP50={calibrated['AP50']:.4f} AP50:95={calibrated['AP50:95']:.4f}")
    print(f"Predictions: {baseline['num_predictions']} -> {calibrated['num_predictions']}")

    output = {
        "thresholds": calibrated_thresholds,
        "per_class": {
            cid: {
                "threshold": entry["threshold"],
                "best_AP50_val": entry["AP50"],
                "class_name": baseline["per_class"].get(cid, {}).get(
                    "class_name", baseline["per_class"].get(str(cid), {}).get("class_name")
                ),
            }
            for cid, entry in per_class_best.items()
        },
        "baseline": {"AP50": baseline["AP50"], "AP50_95": baseline["AP50:95"]},
        "calibrated": {"AP50": calibrated["AP50"], "AP50_95": calibrated["AP50:95"]},
        "grid": grid,
        "provenance": {
            "manifest": str(manifest_path) if manifest_path is not None else None,
            "manifest_schema_version": manifest_schema,
            "manifest_fingerprint": manifest_fingerprint,
            "role": canonical_role,
            "role_fingerprint": role_fingerprint,
            "image_index_fingerprint": image_index_fingerprint(
                FloorPlanImageDataset(
                    args.data_root,
                    split=canonical_role,
                    manifest_path=args.manifest,
                ).records
            ),
        },
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Written: {args.output}")

    changed = {c: v for c, v in calibrated_thresholds.items() if abs(v - 0.05) > 1e-9}
    print(f"\nClasses with non-default threshold: {len(changed)}")
    for cid, threshold in sorted(changed.items(), key=lambda item: -item[1]):
        info = output["per_class"][cid]
        print(f"  {info['class_name']}: {threshold:.2f} (val AP50 {info['best_AP50_val']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
