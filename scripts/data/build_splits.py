"""Build a deterministic image-level train/val/test split manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.splits import (  # noqa: E402
    DEFAULT_SEED,
    DEFAULT_VAL_FRACTION,
    build_split_manifest,
    write_split_manifest,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Split physical images before query expansion: 10% validation from "
            "train directories only; test_set remains untouched"
        )
    )
    parser.add_argument(
        "--data-root",
        "--data_root",
        dest="data_root",
        default=os.environ.get("DATA_ROOT", "./data/FloorPlanCAD_original"),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Manifest path (default: <data-root>/splits.json)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--val-fraction",
        "--val_fraction",
        dest="val_fraction",
        type=float,
        default=DEFAULT_VAL_FRACTION,
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing or invalid metadata",
    )
    parser.add_argument(
        "--validate-only",
        "--validate_only",
        dest="validate_only",
        action="store_true",
        help="Build and validate the manifest in memory without writing it",
    )
    parser.add_argument(
        "--report",
        nargs="?",
        const="-",
        default=None,
        metavar="PATH",
        help="Print class distribution ('-') or write it to PATH",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing manifest",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    data_root = Path(args.data_root)
    output_path = Path(args.output) if args.output else data_root / "splits.json"
    if output_path.exists() and not args.force and not args.validate_only:
        print(f"Refusing to replace existing manifest without --force: {output_path}", file=sys.stderr)
        return 1

    manifest = build_split_manifest(
        data_root,
        seed=args.seed,
        val_fraction=args.val_fraction,
        strict=args.strict,
    )
    if not args.validate_only:
        write_split_manifest(manifest, output_path)
        print(f"Wrote image-level split manifest: {output_path}")
    else:
        print("Manifest validated in memory; no files written.")

    distribution = manifest["class_distribution"]
    for split_name in ("train", "val", "test"):
        counts = distribution[split_name]
        print(
            f"{split_name:>5}: images={counts['images']} "
            f"queries={counts['queries']} instances={counts['instances']}"
        )
    if args.report:
        payload = json.dumps(distribution, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        if args.report == "-":
            print(payload, end="")
        else:
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
