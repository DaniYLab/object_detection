"""Stream-audit every public ArchCAD annotation against the panoptic contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.archcad_constants import (  # noqa: E402
    ARCHCAD_IGNORE_LABEL,
    ARCHCAD_RAW_SEMANTIC_NAMES,
    ARCHCAD_STUFF_SEMANTIC_IDS,
    ARCHCAD_THING_SEMANTIC_IDS,
)
from src.data.archcad_panoptic import (  # noqa: E402
    parse_archcad_panoptic_annotation,
)


def audit_annotation(annotation: Mapping[str, Any], *, image_id: str) -> dict[str, Any]:
    sample = parse_archcad_panoptic_annotation(annotation, image_id=image_id)
    semantic_counts = Counter(map(int, sample.semantic_labels.tolist()))
    segment_counts = Counter(segment.semantic_id for segment in sample.segments)
    primitive_types = Counter(
        str(geometry.get("type", "UNKNOWN")).upper() for geometry in sample.geometries
    )
    thing_without_instance = sum("thing semantic" in issue for issue in sample.issues)
    zero_length = int((sample.primitive_lengths <= 0).sum())
    zero_weight = int((sample.length_weights <= 0).sum())
    raw_instance_semantics: dict[str, set[int]] = {}
    for entity in annotation.get("entities", []):
        if not isinstance(entity, Mapping) or not entity.get("instance"):
            continue
        raw_instance_semantics.setdefault(str(entity["instance"]), set()).add(
            int(entity.get("semantic", ARCHCAD_IGNORE_LABEL))
        )
    conflicts = sum(len(values) > 1 for values in raw_instance_semantics.values())
    return {
        "image_id": image_id,
        "primitive_count": sample.primitive_count,
        "segment_count": len(sample.segments),
        "semantic_counts": dict(semantic_counts),
        "segment_counts": dict(segment_counts),
        "primitive_types": dict(primitive_types),
        "thing_without_instance": thing_without_instance,
        "zero_length_primitives": zero_length,
        "zero_weight_primitives": zero_weight,
        "cross_semantic_instance_id_conflicts": conflicts,
        "issue_count": len(sample.issues),
    }


def _scan(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    annotation = json.loads(payload)
    return audit_annotation(annotation, image_id=path.stem), hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/ArchCAD"))
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 4, 16))
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/panoptic/archcad_public40k_audit_v0.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted((args.data_root / "json").glob("*.json"))
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        paths = paths[: args.limit]
    if not paths:
        raise ValueError("no ArchCAD JSON annotations found")
    semantic_counts: Counter[int] = Counter()
    segment_counts: Counter[int] = Counter()
    primitive_types: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    maximum_primitives = 0
    maximum_segments = 0
    fingerprint = hashlib.sha256()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for record, source_hash in executor.map(_scan, paths):
            semantic_counts.update(record["semantic_counts"])
            segment_counts.update(record["segment_counts"])
            primitive_types.update(record["primitive_types"])
            for key in (
                "primitive_count",
                "segment_count",
                "thing_without_instance",
                "zero_length_primitives",
                "zero_weight_primitives",
                "cross_semantic_instance_id_conflicts",
                "issue_count",
            ):
                totals[key] += int(record[key])
            maximum_primitives = max(maximum_primitives, record["primitive_count"])
            maximum_segments = max(maximum_segments, record["segment_count"])
            fingerprint.update(f"{record['image_id']}:{source_hash}\n".encode("ascii"))

    def semantic_table(counter: Counter[int]) -> list[dict[str, Any]]:
        return [
            {
                "semantic_id": semantic_id,
                "semantic_name": (
                    "ignore_background"
                    if semantic_id == ARCHCAD_IGNORE_LABEL
                    else ARCHCAD_RAW_SEMANTIC_NAMES[semantic_id]
                ),
                "count": counter[semantic_id],
            }
            for semantic_id in sorted(counter)
        ]

    report = {
        "schema_version": "archcad-public40k-panoptic-audit-v0",
        "image_count": len(paths),
        "source_fingerprint": fingerprint.hexdigest(),
        "primitive_count": totals["primitive_count"],
        "segment_count": totals["segment_count"],
        "mean_primitives_per_image": totals["primitive_count"] / len(paths),
        "mean_segments_per_image": totals["segment_count"] / len(paths),
        "maximum_primitives_per_image": maximum_primitives,
        "maximum_segments_per_image": maximum_segments,
        "primitive_type_counts": dict(sorted(primitive_types.items())),
        "semantic_primitive_counts": semantic_table(semantic_counts),
        "semantic_segment_counts": semantic_table(segment_counts),
        "thing_semantic_ids": list(ARCHCAD_THING_SEMANTIC_IDS),
        "stuff_semantic_ids": list(ARCHCAD_STUFF_SEMANTIC_IDS),
        "ignore_label": ARCHCAD_IGNORE_LABEL,
        "anomalies": {
            "thing_primitives_without_instance": totals["thing_without_instance"],
            "zero_length_primitives": totals["zero_length_primitives"],
            "zero_weight_primitives": totals["zero_weight_primitives"],
            "cross_semantic_instance_id_conflicts": totals[
                "cross_semantic_instance_id_conflicts"
            ],
            "parser_issue_count": totals["issue_count"],
        },
        "protocol": {
            "scope": "ArchCAD initial public 40K release",
            "paper_table_comparable": False,
            "reason": "public subset and source-drawing split differ from the full paper benchmark",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
