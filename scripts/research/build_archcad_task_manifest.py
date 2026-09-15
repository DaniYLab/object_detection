"""Build a non-split task manifest for heterogeneous ArchCAD chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.drawing_graph import classify_archcad_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/ArchCAD"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/drawing_graph/archcad_task_manifest.json"),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def _scan(path: Path) -> tuple[str, dict[str, bool | int]]:
    with path.open("r", encoding="utf-8") as handle:
        return path.stem, classify_archcad_task(json.load(handle))


def main() -> None:
    args = parse_args()
    annotation_paths = sorted((args.data_root / "json").glob("*.json"))
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        annotation_paths = annotation_paths[: args.limit]
    if not annotation_paths:
        raise FileNotFoundError("no ArchCAD JSON annotations found")

    subsets: dict[str, list[str]] = {
        "wall_present": [],
        "wall_graph_candidate": [],
        "door_connectivity_candidate": [],
        "architectural_reading_candidate": [],
        "floor_plan_reading_candidate": [],
        "structure_engineering_candidate": [],
    }
    totals: Counter[str] = Counter()
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        scanned = executor.map(_scan, annotation_paths)
        for index, (image_id, tags) in enumerate(scanned, start=1):
            for key in subsets:
                if tags[key]:
                    subsets[key].append(image_id)
            totals.update(
                {
                    "wall_primitives": int(tags["wall_primitives"]),
                    "door_primitives": int(tags["door_primitives"]),
                    "fixture_primitives": int(tags["fixture_primitives"]),
                    "engineering_primitives": int(tags["engineering_primitives"]),
                    "door_instances": int(tags["door_instances"]),
                    "fixture_instances": int(tags["fixture_instances"]),
                }
            )
            records.append({"image_id": image_id, **tags})
            if index % 5000 == 0:
                print(f"scanned {index}/{len(annotation_paths)}")

    fingerprint_payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()
    sample_count = len(records)
    manifest = {
        "schema_version": "archcad-task-manifest-v0",
        "purpose": "task_selection_only_not_an_evaluation_split",
        "source_drawing_disjoint": False,
        "publication_split_ready": False,
        "sample_count": sample_count,
        "fingerprint": fingerprint,
        "criteria": {
            "wall_graph_candidate": "wall_primitives >= 4",
            "door_connectivity_candidate": "wall_graph_candidate and door_primitives > 0",
            "architectural_reading_candidate": (
                "wall_graph_candidate and (door_primitives > 0 or fixture_primitives > 0)"
            ),
            "floor_plan_reading_candidate": (
                "wall_primitives >= 8 and (door_instances >= 2 or "
                "(door_instances >= 1 and fixture_instances >= 1) or fixture_instances >= 3)"
            ),
            "structure_engineering_candidate": "engineering_primitives > 0",
        },
        "subset_counts": {key: len(value) for key, value in subsets.items()},
        "subset_pct": {
            key: round(100.0 * len(value) / sample_count, 2)
            for key, value in subsets.items()
        },
        "primitive_totals": dict(sorted(totals.items())),
        "subsets": subsets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: manifest[key] for key in ("sample_count", "fingerprint", "subset_counts", "subset_pct")}, indent=2))


if __name__ == "__main__":
    main()
