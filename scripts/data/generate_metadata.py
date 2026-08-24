"""Deprecated copied-layout metadata generator.

New code should use ``scripts/data/build_dataset.py``. This compatibility command
retains the old output layout but delegates SVG parsing and validation entirely
to the canonical schema-v2 parser in ``src.data.metadata``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.metadata import (  # noqa: E402
    STUFF_POLICIES,
    UNKNOWN_POLICIES,
    StuffPolicy,
    UnknownPolicy,
    load_metadata,
    parse_path_bbox,
    parse_svg_metadata,
    validate_metadata_sources,
)

ORIGINAL_ROOT = Path(os.environ.get("ORIGINAL_ROOT", "./data/FloorPlanCAD_original"))
DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", "./data/FloorPlanCAD_dataset"))

SPLITS = {
    "train": ["train_set_1", "train_set_2"],
    "test": ["test_set"],
}


def parse_svg(
    svg_path: str | Path,
    *,
    min_size: float = 0.0,
    stuff_policy: StuffPolicy = "merge_by_class",
    unknown_policy: UnknownPolicy = "warn",
    strict: bool = False,
) -> dict[str, Any]:
    """Compatibility wrapper around :func:`parse_svg_metadata`."""

    return parse_svg_metadata(
        svg_path,
        Path(svg_path).with_suffix(".png"),
        min_size=min_size,
        stuff_policy=stuff_policy,
        unknown_policy=unknown_policy,
        strict=strict,
    )


# Old helper name now points at the canonical path parser.
path_bbox = parse_path_bbox


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(metadata, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def generate_metadata(
    original_root: str | Path,
    dataset_root: str | Path,
    *,
    min_size: float = 0.0,
    stuff_policy: StuffPolicy = "merge_by_class",
    unknown_policy: UnknownPolicy = "warn",
    force: bool = False,
    validate_only: bool = False,
    strict: bool = False,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    original_root = Path(original_root)
    dataset_root = Path(dataset_root)
    report: dict[str, Any] = {
        "deprecated": True,
        "original_root": str(original_root.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "settings": {
            "min_size": min_size,
            "stuff_policy": stuff_policy,
            "unknown_policy": unknown_policy,
            "force": force,
            "validate_only": validate_only,
            "strict": strict,
        },
        "processed": 0,
        "validated": 0,
        "skipped_existing": 0,
        "missing_output_sample": 0,
        "failed": 0,
        "errors": [],
    }

    for split_name, source_dirs in SPLITS.items():
        output_split = dataset_root / split_name
        for source_dir_name in source_dirs:
            source_dir = original_root / source_dir_name
            if not source_dir.exists():
                continue
            for svg_path in sorted(source_dir.glob("*.svg")):
                output_dir = output_split / svg_path.stem
                metadata_path = output_dir / "metadata.json"
                if not output_dir.is_dir() and not validate_only:
                    report["missing_output_sample"] += 1
                    if strict:
                        report["failed"] += 1
                        report["errors"].append(
                            {"sample": str(svg_path), "error": "copied-layout sample directory is missing"}
                        )
                    continue
                try:
                    if metadata_path.exists() and not force:
                        if validate_only or strict:
                            existing = load_metadata(
                                metadata_path, allow_legacy=True, strict=strict
                            )
                            source_report = validate_metadata_sources(
                                existing, svg_path.with_suffix(".png"), svg_path
                            )
                            if source_report.errors or (strict and source_report.warnings):
                                issues = source_report.errors + (
                                    source_report.warnings if strict else []
                                )
                                raise ValueError(
                                    "; ".join(
                                        f"{issue.path}: {issue.message}" for issue in issues
                                    )
                                )
                            report["validated"] += 1
                        else:
                            report["skipped_existing"] += 1
                        continue
                    metadata = parse_svg(
                        svg_path,
                        min_size=min_size,
                        stuff_policy=stuff_policy,
                        unknown_policy=unknown_policy,
                        strict=strict,
                    )
                    if validate_only:
                        report["validated"] += 1
                    else:
                        _write_metadata(metadata_path, metadata)
                        report["processed"] += 1
                except Exception as exc:
                    report["failed"] += 1
                    report["errors"].append({"sample": str(svg_path), "error": str(exc)})

    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deprecated copied-layout generator using canonical metadata parser"
    )
    parser.add_argument("--original-root", default=str(ORIGINAL_ROOT))
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT))
    parser.add_argument("--min-size", type=float, default=0.0)
    parser.add_argument(
        "--stuff-policy", "--stuff_policy", dest="stuff_policy",
        choices=STUFF_POLICIES,
        default="merge_by_class",
    )
    parser.add_argument(
        "--unknown-policy", "--unknown_policy", dest="unknown_policy",
        choices=UNKNOWN_POLICIES,
        default="warn",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate-only", "--validate_only", dest="validate_only", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--report", "--report-json", "--report_json", dest="report",
        nargs="?", const="generate_metadata_report.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    print("WARNING: generate_metadata.py is deprecated; use build_dataset.py for new data.")
    report = generate_metadata(
        args.original_root,
        args.dataset_root,
        min_size=args.min_size,
        stuff_policy=args.stuff_policy,
        unknown_policy=args.unknown_policy,
        force=args.force,
        validate_only=args.validate_only,
        strict=args.strict,
        report_path=args.report,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
