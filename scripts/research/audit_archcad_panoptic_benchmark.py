"""Verify the ArchCAD panoptic benchmark contract with identity predictions."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.archcad_panoptic import (  # noqa: E402
    oracle_panoptic_prediction,
    parse_archcad_panoptic_annotation,
)
from src.evaluation.archcad_panoptic import evaluate_archcad_panoptic  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/ArchCAD"))
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/panoptic/archcad_oracle_benchmark_v0.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted((args.data_root / "json").glob("*.json"))
    if args.sample_count <= 0:
        raise ValueError("sample_count must be positive")
    selected = random.Random(args.seed).sample(
        paths, min(args.sample_count, len(paths))
    )
    selected.sort()
    samples = []
    issue_counts: Counter[str] = Counter()
    for path in selected:
        annotation = json.loads(path.read_text(encoding="utf-8"))
        sample = parse_archcad_panoptic_annotation(annotation, image_id=path.stem)
        samples.append(sample)
        issue_counts.update(
            "thing_without_instance"
            if "thing semantic" in issue
            else "invalid_entity"
            for issue in sample.issues
        )
    predictions = [oracle_panoptic_prediction(sample) for sample in samples]
    report = evaluate_archcad_panoptic(samples, predictions)
    report.update(
        {
            "audit": "identity_prediction_oracle",
            "seed": args.seed,
            "issue_counts": dict(sorted(issue_counts.items())),
            "sample_ids": [sample.image_id for sample in samples],
            "expected_identity_score": 1.0,
            "identity_is_only_sanity_check": True,
            "paper_metric_parity_verified": False,
            "official_release_notes": {
                "semantic_classes": 30,
                "thing_classes": 21,
                "stuff_classes": 9,
                "upstream_data_loader_present": False,
                "upstream_pointops_present": False,
                "paper_split_reproducible_from_public_40k": False,
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "sample_count": report["sample_count"],
                "primitive_count": report["primitive_count"],
                "semantic": {
                    key: value
                    for key, value in report["semantic"].items()
                    if key not in {"per_class", "weighted_per_class"}
                },
                "instance": {
                    key: value
                    for key, value in report["instance"].items()
                    if key != "per_class"
                },
                "panoptic": {
                    key: value
                    for key, value in report["panoptic"].items()
                    if key != "per_class"
                },
                "issue_counts": report["issue_counts"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
