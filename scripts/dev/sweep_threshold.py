"""Threshold sweep over saved predictions (protocol-compliant calibration).

Re-scores an existing predictions JSON at multiple decoder thresholds and
reports AP50/AP50:95 at each. This calibrates the decoder threshold on the
validation split only — never on test — satisfying the frozen-before-test
protocol. Uses the same COCO-style metric implementation as evaluate.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import CLASS_NAMES, NUM_CLASSES
from src.evaluation import compute_coco_metrics
from src.data.dataset import FloorPlanImageDataset, image_collate_fn
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--data-root", default="./data/FloorPlanCAD_original")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=[0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    predictions = payload["predictions"] if isinstance(payload, dict) else payload
    targets = _targets_for_split(args.data_root, args.manifest, args.split)

    rows = []
    for threshold in args.thresholds:
        filtered = []
        for record in predictions:
            keep = [
                index
                for index, score in enumerate(record["scores"])
                if score >= threshold
            ]
            filtered.append(
                {
                    "image_id": record["image_id"],
                    "boxes": [record["boxes"][i] for i in keep],
                    "scores": [record["scores"][i] for i in keep],
                    "labels": [record["labels"][i] for i in keep],
                }
            )
        metrics = compute_coco_metrics(filtered, targets, class_ids=range(NUM_CLASSES))
        rows.append(
            {
                "threshold": threshold,
                "AP50": metrics["AP50"],
                "AP50_95": metrics["AP50:95"],
                "num_predictions": metrics["num_predictions"],
            }
        )
        print(
            f"threshold={threshold:.2f} | AP50={metrics['AP50']:.4f} | "
            f"AP50:95={metrics['AP50:95']:.4f} | preds={metrics['num_predictions']}"
        )

    best = max(rows, key=lambda row: row["AP50_95"])
    print(f"\nBest by AP50:95: threshold={best['threshold']} AP50={best['AP50']:.4f} AP50:95={best['AP50_95']:.4f}")

    if args.output:
        Path(args.output).write_text(
            json.dumps({"rows": rows, "best": best}, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
