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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Saved val predictions JSON")
    parser.add_argument("--data-root", default="./data/FloorPlanCAD_original")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split", default="val", choices=["val"])
    parser.add_argument("--grid", type=float, nargs="*", default=None,
                        help="Threshold grid (default: 0.05..0.60 step 0.05 plus 0.75/0.85)")
    parser.add_argument("--output", default="outputs/class_thresholds.json")
    args = parser.parse_args(argv)

    grid = args.grid or [round(x, 2) for x in np.arange(0.05, 0.65, 0.05)] + [0.75, 0.85]
    payload = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    predictions = payload["predictions"] if isinstance(payload, dict) else payload
    targets = _targets_for_split(args.data_root, args.manifest, args.split)

    baseline = compute_coco_metrics(predictions, targets, class_ids=range(NUM_CLASSES))
    print(f"Baseline (global 0.05, as decoded): AP50={baseline['AP50']:.4f} AP50:95={baseline['AP50:95']:.4f}")

    # Per-class grid search: maximise that class's AP50.
    per_class_best: dict[int, dict] = {}
    for class_id in range(NUM_CLASSES):
        best = {"threshold": 0.05, "AP50": -1.0}
        for threshold in grid:
            thresholds = {class_id: threshold}
            filtered = _apply_thresholds(predictions, thresholds)
            metrics = compute_coco_metrics(filtered, targets, class_ids=range(NUM_CLASSES))
            ap = metrics["per_class"].get(str(class_id), {}).get("AP50", -1.0)
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
                "class_name": baseline["per_class"].get(str(cid), {}).get("class_name"),
            }
            for cid, entry in per_class_best.items()
        },
        "baseline": {"AP50": baseline["AP50"], "AP50_95": baseline["AP50:95"]},
        "calibrated": {"AP50": calibrated["AP50"], "AP50_95": calibrated["AP50:95"]},
        "grid": grid,
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
