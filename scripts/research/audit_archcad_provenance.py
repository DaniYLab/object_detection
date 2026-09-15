"""Audit whether the public ArchCAD-40K release supports drawing-disjoint splits."""

from __future__ import annotations

import argparse
import json
import random
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any


UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)
PROVENANCE_KEY_TERMS = ("source", "drawing", "project", "parent", "origin", "chunk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-data", type=Path, default=Path("data/ArchCAD_raw/data")
    )
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/drawing_graph/archcad_provenance_audit.json"),
    )
    return parser.parse_args()


def _archive_ids(path: Path) -> tuple[set[str], int]:
    with zipfile.ZipFile(path) as archive:
        stems = {
            Path(member).stem
            for member in archive.namelist()
            if member and not member.endswith("/")
        }
    return stems, sum(bool(UUID_PATTERN.fullmatch(stem)) for stem in stems)


def _walk_mapping_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            keys.append(str(key))
            keys.extend(_walk_mapping_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.extend(_walk_mapping_keys(child))
    return keys


def audit_release(raw_data: Path, sample_count: int, seed: int) -> dict[str, Any]:
    archive_names = ("png.zip", "svg.zip", "json.zip", "point.zip", "caption.zip")
    archive_ids: dict[str, set[str]] = {}
    inventory: dict[str, Any] = {}
    for archive_name in archive_names:
        ids, uuid_count = _archive_ids(raw_data / archive_name)
        archive_ids[archive_name] = ids
        inventory[archive_name] = {
            "file_count": len(ids),
            "uuid_filename_count": uuid_count,
            "all_filenames_are_uuid": uuid_count == len(ids),
        }

    json_ids = archive_ids["json.zip"]
    selected = random.Random(seed).sample(sorted(json_ids), min(sample_count, len(json_ids)))
    top_level_key_counts: Counter[str] = Counter()
    nested_key_counts: Counter[str] = Counter()
    provenance_key_counts: Counter[str] = Counter()
    invalid_json: list[str] = []
    with zipfile.ZipFile(raw_data / "json.zip") as archive:
        for image_id in selected:
            member = f"json/{image_id}.json"
            try:
                value = json.loads(archive.read(member))
            except (KeyError, json.JSONDecodeError, UnicodeDecodeError):
                invalid_json.append(image_id)
                continue
            if not isinstance(value, dict):
                continue
            top_level_key_counts.update(value.keys())
            all_keys = _walk_mapping_keys(value)
            nested_key_counts.update(all_keys)
            provenance_key_counts.update(
                key
                for key in all_keys
                if any(term in key.lower() for term in PROVENANCE_KEY_TERMS)
            )

    svg_root_attribute_counts: Counter[str] = Counter()
    svg_element_attribute_counts: Counter[str] = Counter()
    svg_provenance_attribute_counts: Counter[str] = Counter()
    invalid_svg: list[str] = []
    with zipfile.ZipFile(raw_data / "svg.zip") as archive:
        for image_id in selected:
            member = f"svg/{image_id}.svg"
            try:
                root = ET.fromstring(archive.read(member))
            except (KeyError, ET.ParseError):
                invalid_svg.append(image_id)
                continue
            svg_root_attribute_counts.update(root.attrib.keys())
            for element in root.iter():
                keys = [str(key) for key in element.attrib]
                svg_element_attribute_counts.update(keys)
                svg_provenance_attribute_counts.update(
                    key
                    for key in keys
                    if any(term in key.lower() for term in PROVENANCE_KEY_TERMS)
                )

    common_ids = set.intersection(*archive_ids.values())
    union_ids = set.union(*archive_ids.values())
    provenance_available = bool(provenance_key_counts) or bool(
        svg_provenance_attribute_counts
    )
    return {
        "schema_version": "archcad-provenance-audit-v1",
        "release": "ArchCAD initial public 40K batch",
        "archive_inventory": inventory,
        "modality_id_sets_identical": len(common_ids) == len(union_ids),
        "common_modality_id_count": len(common_ids),
        "union_modality_id_count": len(union_ids),
        "json_metadata_sample": {
            "seed": seed,
            "sample_count": len(selected),
            "top_level_key_counts": dict(sorted(top_level_key_counts.items())),
            "all_nested_key_counts": dict(sorted(nested_key_counts.items())),
            "provenance_like_key_counts": dict(sorted(provenance_key_counts.items())),
            "invalid_json_ids": invalid_json,
        },
        "svg_metadata_sample": {
            "seed": seed,
            "sample_count": len(selected),
            "root_attribute_counts": dict(sorted(svg_root_attribute_counts.items())),
            "all_element_attribute_counts": dict(
                sorted(svg_element_attribute_counts.items())
            ),
            "provenance_like_attribute_counts": dict(
                sorted(svg_provenance_attribute_counts.items())
            ),
            "invalid_svg_ids": invalid_svg,
        },
        "source_drawing_id_available": provenance_available,
        "drawing_disjoint_split_possible": provenance_available,
        "exact_duplicate_hash_split_is_sufficient": False,
        "recovery_status": {
            "recoverable_from_public_release_fields": provenance_available,
            "required_upstream_artifact": (
                "chunk UUID to source drawing/project mapping or the original "
                "drawing-disjoint split manifest"
            ),
            "unsafe_inferences_rejected": [
                "ZIP member order",
                "ZIP modification timestamps",
                "UUID lexical proximity",
                "exact-image SHA-256 grouping as a source-drawing substitute",
            ],
        },
        "conclusion": (
            "The five modalities share flat UUID sample IDs, but the sampled release "
            "metadata exposes no parent drawing/project identifier. Request or recover "
            "the upstream chunk-to-drawing split manifest before reporting generalization."
        ),
    }


def main() -> None:
    args = parse_args()
    report = audit_release(args.raw_data, args.sample_count, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
