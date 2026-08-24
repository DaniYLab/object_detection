from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from src.data.metadata import (
    METADATA_SCHEMA_VERSION,
    MetadataValidationError,
    adapt_legacy_metadata,
    load_metadata,
    parse_svg_metadata,
    validate_metadata,
    validate_metadata_sources,
)


def _write_sample(tmp_path: Path) -> tuple[Path, Path]:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (200, 100), "white").save(image_path)
    svg_path = tmp_path / "sample.svg"
    svg_path.write_text(
        """<svg xmlns="http://www.w3.org/2000/svg" viewBox="10 20 100 50">
        <path semantic-id="2" instance-id="7" d="M 10 20 L 30 20 L 30 40 L 10 40 Z"/>
        <path semantic-id="1" instance-id="-1" d="M 40 20 L 50 20 L 50 30 L 40 30 Z"/>
        <path semantic-id="1" instance-id="-1" d="M 60 30 L 70 30 L 70 40 L 60 40 Z"/>
        </svg>""",
        encoding="utf-8",
    )
    return image_path, svg_path


def test_schema_v2_respects_viewbox_origin_and_provenance(tmp_path: Path) -> None:
    image_path, svg_path = _write_sample(tmp_path)
    metadata = parse_svg_metadata(
        svg_path,
        image_path,
        min_size=0,
        stuff_policy="exclude",
    )

    assert metadata["schema_version"] == METADATA_SCHEMA_VERSION
    assert metadata["svg_viewbox"] == [10.0, 20.0, 100.0, 50.0]
    assert metadata["instances"][0]["bbox_px"] == [0.0, 0.0, 40.0, 40.0]
    assert metadata["stats"]["stuff_paths_excluded"] == 2
    assert len(metadata["source"]["image_sha256"]) == 64
    assert len(metadata["source"]["svg_sha256"]) == 64
    assert len(metadata["source"]["fingerprint"]) == 64
    assert metadata["build"]["stuff_policy"] == "exclude"
    assert validate_metadata(metadata).valid


def test_stuff_policies_are_explicit(tmp_path: Path) -> None:
    image_path, svg_path = _write_sample(tmp_path)
    merged = parse_svg_metadata(
        svg_path, image_path, min_size=0, stuff_policy="merge_by_class"
    )
    path_instances = parse_svg_metadata(
        svg_path, image_path, min_size=0, stuff_policy="path_instances"
    )

    merged_walls = [item for item in merged["instances"] if item["class"] == "wall"]
    path_walls = [item for item in path_instances["instances"] if item["class"] == "wall"]
    assert len(merged_walls) == 1
    assert merged_walls[0]["bbox_px"] == [60.0, 0.0, 120.0, 40.0]
    assert len(path_walls) == 2


def test_validation_rejects_non_finite_and_out_of_bounds_boxes(tmp_path: Path) -> None:
    image_path, svg_path = _write_sample(tmp_path)
    metadata = parse_svg_metadata(svg_path, image_path, min_size=0)
    metadata["instances"][0]["bbox_px"] = [0.0, 0.0, float("inf"), 20.0]
    report = validate_metadata(metadata)
    assert not report.valid
    assert any("finite" in issue.message for issue in report.errors)

    metadata["instances"][0]["bbox_px"] = [0.0, 0.0, 201.0, 20.0]
    report = validate_metadata(metadata)
    assert not report.valid
    assert any("within" in issue.message for issue in report.errors)


def test_legacy_adapter_and_loader_do_not_rewrite_source(tmp_path: Path) -> None:
    path = tmp_path / "legacy_meta.json"
    legacy = {
        "image_size": [20, 10],
        "svg_viewbox": [5, 6, 20, 10],
        "num_instances": 1,
        "instances": [
            {
                "class": "door_single",
                "class_id": 10,
                "instance_id": 3,
                "bbox_px": [1, 2, 8, 9],
            }
        ],
    }
    original = json.dumps(legacy, separators=(",", ":"))
    path.write_text(original, encoding="utf-8")

    adapted = load_metadata(path)
    assert adapted["schema_version"] == 2
    assert adapted["build"]["parser"] == "legacy_adapter"
    assert adapted["instances"][0]["semantic_id"] == 2
    assert path.read_text(encoding="utf-8") == original
    assert validate_metadata(adapt_legacy_metadata(legacy)).valid


def test_source_validation_detects_stale_input(tmp_path: Path) -> None:
    image_path, svg_path = _write_sample(tmp_path)
    metadata = parse_svg_metadata(svg_path, image_path, min_size=0)
    assert validate_metadata_sources(metadata, image_path, svg_path).valid

    svg_path.write_text('<svg viewBox="0 0 10 10"/>', encoding="utf-8")
    report = validate_metadata_sources(metadata, image_path, svg_path)
    assert not report.valid
    assert any(issue.path == "source.svg_sha256" for issue in report.errors)


def test_unknown_semantic_policy_warns_or_rejects(tmp_path: Path) -> None:
    image_path = tmp_path / "unknown.png"
    Image.new("RGB", (20, 20), "white").save(image_path)
    svg_path = tmp_path / "unknown.svg"
    svg_path.write_text(
        '<svg viewBox="0 0 20 20"><path semantic-id="999" instance-id="1" '
        'd="M 1 1 L 10 1 L 10 10 L 1 10 Z"/></svg>',
        encoding="utf-8",
    )

    metadata = parse_svg_metadata(
        svg_path,
        image_path,
        min_size=0,
        unknown_policy="warn",
    )
    assert metadata["num_instances"] == 0
    assert metadata["stats"]["paths_unknown_semantic_id"] == 1
    assert metadata["build"]["unknown_policy"] == "warn"
    assert any("unknown semantic-id 999" in item for item in metadata["stats"]["warnings"])

    with pytest.raises(MetadataValidationError, match="Unknown semantic IDs"):
        parse_svg_metadata(
            svg_path,
            image_path,
            min_size=0,
            unknown_policy="error",
        )


def test_invalid_viewbox_fails_before_writing_any_metadata(tmp_path: Path) -> None:
    image_path = tmp_path / "bad.png"
    Image.new("RGB", (10, 10)).save(image_path)
    svg_path = tmp_path / "bad.svg"
    svg_path.write_text('<svg viewBox="0 0 inf 10"/>', encoding="utf-8")
    with pytest.raises(MetadataValidationError):
        parse_svg_metadata(svg_path, image_path)
    assert not (tmp_path / "bad_meta.json").exists()
