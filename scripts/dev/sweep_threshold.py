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
from src.data.splits import (
    canonical_split_role,
    image_index_fingerprint,
    load_split_manifest,
    manifest_schema_version,
    resolve_split_manifest_path,
)
from torch.utils.data import DataLoader


def _targets_for_split(data_root: str, manifest: str | None, split: str) -> tuple[list[dict], FloorPlanImageDataset]:
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
    return targets, dataset


def _validate_prediction_provenance(
    payload: object,
    *,
    role: str,
    manifest_schema: int | None,
    manifest_fingerprint: str | None,
    role_fingerprint: str | None,
) -> None:
    if not isinstance(payload, dict):
        return
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        if manifest_schema == 2:
            raise ValueError("v2 calibration predictions must include provenance metadata")
        return
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--data-root", default="./data/FloorPlanCAD_original")
    parser.add_argument("--manifest", default=None)
    parser.add_argument(
        "--split",
        default="val_calibration",
        choices=["val", "val_select", "val_calibration", "test", "test_set"],
        help="Calibration role; v2 defaults to val_calibration and never uses test_set.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=[0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    if args.split in {"test", "test_set"}:
        raise ValueError("Threshold calibration cannot use the test_set role")
    manifest_path = resolve_split_manifest_path(args.data_root, args.manifest)
    manifest_schema = None
    canonical_role = args.split
    manifest_fingerprint = None
    role_fingerprint = None
    if manifest_path is not None:
        manifest = load_split_manifest(manifest_path)
        manifest_schema = manifest_schema_version(manifest)
        requested_role = (
            "val" if manifest_schema == 1 and args.split == "val_calibration" else args.split
        )
        canonical_role = canonical_split_role(manifest, requested_role)
        manifest_fingerprint = manifest.get("fingerprint")
        role_fingerprint = manifest.get("role_fingerprints", {}).get(canonical_role)
    if canonical_role == "test_set":
        raise ValueError("Threshold calibration cannot use the test_set role")

    payload = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    _validate_prediction_provenance(
        payload,
        role=canonical_role,
        manifest_schema=manifest_schema,
        manifest_fingerprint=manifest_fingerprint,
        role_fingerprint=role_fingerprint,
    )
    predictions = payload["predictions"] if isinstance(payload, dict) else payload
    targets, dataset = _targets_for_split(args.data_root, args.manifest, canonical_role)

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
            json.dumps(
                {
                    "rows": rows,
                    "best": best,
                    "provenance": {
                        "requested_role": args.split,
                        "role": canonical_role,
                        "manifest": str(manifest_path) if manifest_path is not None else None,
                        "manifest_schema_version": manifest_schema,
                        "manifest_fingerprint": manifest_fingerprint,
                        "role_fingerprint": role_fingerprint,
                        "image_index_fingerprint": image_index_fingerprint(dataset.records),
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
