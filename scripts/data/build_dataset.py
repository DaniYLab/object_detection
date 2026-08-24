"""Build canonical FloorPlanCAD schema-v2 metadata beside source PNG files.

No images are copied. Existing metadata is never replaced unless ``--force`` is
explicitly supplied. ``--validate-only`` performs no writes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.metadata import (  # noqa: E402
    STUFF_POLICIES,
    UNKNOWN_POLICIES,
    MetadataValidationError,
    StuffPolicy,
    UnknownPolicy,
    load_metadata,
    parse_path_bbox,
    parse_svg_metadata,
    validate_metadata,
    validate_metadata_sources,
)

SPLITS = {
    "train": ["train_set_1", "train_set_2"],
    "test": ["test_set"],
}


@dataclass
class BuildReport:
    data_root: str
    settings: dict[str, Any]
    discovered: int = 0
    processed: int = 0
    validated: int = 0
    skipped_existing: int = 0
    missing_svg: int = 0
    failed: int = 0
    instances: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(metadata, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def process_sample(
    png_path: Path,
    svg_path: Path,
    min_size: float = 8,
    *,
    stuff_policy: StuffPolicy = "exclude",
    unknown_policy: UnknownPolicy = "warn",
    force: bool = False,
    validate_only: bool = False,
    strict: bool = False,
) -> int:
    """Process one sample and return instances, or ``-1`` when skipped.

    This preserves the old helper's integer return contract while delegating all
    parsing and validation to the canonical implementation.
    """

    meta_path = png_path.with_name(f"{png_path.stem}_meta.json")
    if meta_path.exists() and not force:
        if validate_only or strict:
            metadata = load_metadata(meta_path, allow_legacy=True, strict=strict)
            source_report = validate_metadata_sources(metadata, png_path, svg_path)
            if source_report.errors or (strict and source_report.warnings):
                issues = source_report.errors + (source_report.warnings if strict else [])
                raise MetadataValidationError(
                    "; ".join(f"{issue.path}: {issue.message}" for issue in issues),
                    source_report,
                )
            return int(metadata["num_instances"])
        return -1

    metadata = parse_svg_metadata(
        svg_path,
        png_path,
        min_size=min_size,
        stuff_policy=stuff_policy,
        unknown_policy=unknown_policy,
        strict=strict,
    )
    validation = validate_metadata(metadata)
    if not validation.valid:
        raise MetadataValidationError(f"Generated invalid metadata for {png_path}", validation)
    if not validate_only:
        _write_metadata(meta_path, metadata)
    return int(metadata["num_instances"])


def build_dataset(
    data_root: Path,
    min_size: float = 8,
    *,
    stuff_policy: StuffPolicy = "exclude",
    unknown_policy: UnknownPolicy = "warn",
    force: bool = False,
    validate_only: bool = False,
    strict: bool = False,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Build or validate all source metadata and return a structured report."""

    data_root = Path(data_root)
    report = BuildReport(
        data_root=str(data_root.resolve()),
        settings={
            "min_size": float(min_size),
            "stuff_policy": stuff_policy,
            "unknown_policy": unknown_policy,
            "force": force,
            "validate_only": validate_only,
            "strict": strict,
        },
    )
    print("=" * 60)
    print("  FloorPlanCAD Metadata Builder (schema v2)")
    print(f"  Data root     : {data_root.resolve()}")
    print(f"  min_size       : {min_size}px")
    print(f"  stuff_policy   : {stuff_policy}")
    print(f"  unknown_policy : {unknown_policy}")
    print(f"  validate_only  : {validate_only}")
    print(f"  force           : {force}")
    print("=" * 60)

    for split_name, source_dirs in SPLITS.items():
        print(f"\n[{split_name.upper()}]")
        for source_dir_name in source_dirs:
            source_dir = data_root / source_dir_name
            if not source_dir.exists():
                print(f"  [SKIP] {source_dir} not found")
                continue
            png_files = sorted(source_dir.glob("*.png"))
            print(f"  {source_dir_name}: {len(png_files)} images")
            for png_path in png_files:
                report.discovered += 1
                svg_path = png_path.with_suffix(".svg")
                metadata_path = png_path.with_name(f"{png_path.stem}_meta.json")
                if not svg_path.is_file() and not (validate_only and metadata_path.is_file()):
                    report.missing_svg += 1
                    message = "matching SVG is missing"
                    if strict:
                        report.failed += 1
                        report.errors.append({"sample": str(png_path), "error": message})
                    continue
                try:
                    if metadata_path.exists() and not force:
                        if validate_only or strict:
                            metadata = load_metadata(
                                metadata_path,
                                allow_legacy=True,
                                strict=strict,
                            )
                            source_report = validate_metadata_sources(
                                metadata, png_path, svg_path
                            )
                            if source_report.errors or (strict and source_report.warnings):
                                issues = source_report.errors + (
                                    source_report.warnings if strict else []
                                )
                                raise MetadataValidationError(
                                    "; ".join(
                                        f"{issue.path}: {issue.message}" for issue in issues
                                    ),
                                    source_report,
                                )
                            report.validated += 1
                            report.instances += int(metadata["num_instances"])
                        else:
                            report.skipped_existing += 1
                        continue

                    metadata = parse_svg_metadata(
                        svg_path,
                        png_path,
                        min_size=min_size,
                        stuff_policy=stuff_policy,
                        unknown_policy=unknown_policy,
                        strict=strict,
                    )
                    report.instances += int(metadata["num_instances"])
                    if validate_only:
                        report.validated += 1
                    else:
                        _write_metadata(metadata_path, metadata)
                        report.processed += 1
                except Exception as exc:
                    report.failed += 1
                    report.errors.append({"sample": str(png_path), "error": str(exc)})
                    print(f"    [ERROR] {png_path.name}: {exc}")

    result = report.to_dict()
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")

    print(f"\n{'=' * 60}")
    print(f"  Processed       : {report.processed}")
    print(f"  Validated       : {report.validated}")
    print(f"  Skipped existing: {report.skipped_existing}")
    print(f"  Missing SVG     : {report.missing_svg}")
    print(f"  Failed          : {report.failed}")
    print(f"  Instances       : {report.instances}")
    print("=" * 60)
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate canonical *_meta.json files without copying images"
    )
    parser.add_argument(
        "--data-root",
        "--data_root",
        dest="data_root",
        default=os.environ.get("DATA_ROOT", "./data/FloorPlanCAD_original"),
        help="Directory containing train_set_1, train_set_2, and test_set",
    )
    parser.add_argument(
        "--min-size",
        "--min_size",
        dest="min_size",
        type=float,
        default=8.0,
        help="Drop boxes smaller than this many pixels",
    )
    parser.add_argument(
        "--stuff-policy",
        "--stuff_policy",
        dest="stuff_policy",
        choices=STUFF_POLICIES,
        default="exclude",
        help="How instance-id=-1 paths are represented",
    )
    parser.add_argument(
        "--unknown-policy",
        "--unknown_policy",
        dest="unknown_policy",
        choices=UNKNOWN_POLICIES,
        default="warn",
        help="Skip unknown semantic IDs with warnings or reject the SVG",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing metadata")
    parser.add_argument(
        "--validate-only",
        "--validate_only",
        dest="validate_only",
        action="store_true",
        help="Parse/validate without writing metadata",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat source warnings, missing files, and validation warnings as failures",
    )
    parser.add_argument(
        "--report",
        "--report-json",
        "--report_json",
        dest="report",
        nargs="?",
        const="build_metadata_report.json",
        default=None,
        metavar="PATH",
        help="Write a JSON build report (default: build_metadata_report.json)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = build_dataset(
        data_root=Path(args.data_root),
        min_size=args.min_size,
        stuff_policy=args.stuff_policy,
        unknown_policy=args.unknown_policy,
        force=args.force,
        validate_only=args.validate_only,
        strict=args.strict,
        report_path=Path(args.report) if args.report else None,
    )
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
