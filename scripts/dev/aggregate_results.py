"""Aggregate FloorPlanCAD evaluation reports into a comparison table.

Reads evaluation JSON reports (produced by evaluate.py) and prints a Markdown
table of AP50 / AP50:95 plus parameter counts, so ablation runs can be compared
side by side. Optionally writes the aggregate to JSON.
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

from src.models import build_model


def _preset_param_count(preset: str | None) -> int | None:
    if not preset:
        return None
    try:
        model = build_model(preset)
    except Exception:
        return None
    count = sum(parameter.numel() for parameter in model.parameters())
    del model
    return count


def _load_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    source = data.get("source", {})
    metrics = data.get("metrics", {})
    dataset = data.get("data", {})
    preset = source.get("preset")
    return {
        "report": str(path),
        "preset": preset,
        "split": dataset.get("split"),
        "num_images": metrics.get("num_images"),
        "num_gt": metrics.get("num_gt"),
        "num_predictions": metrics.get("num_predictions"),
        "num_classes_with_gt": metrics.get("num_classes_with_gt"),
        "AP50": metrics.get("AP50"),
        "AP50_95": metrics.get("AP50:95"),
        "parameters": _preset_param_count(preset),
        "manifest_fingerprint": dataset.get("manifest_fingerprint"),
    }


def _format_row(entry: dict[str, Any]) -> str:
    def fmt(value: Any, spec: str = "") -> str:
        if value is None:
            return "—"
        if spec:
            return format(value, spec)
        return str(value)

    params = entry["parameters"]
    params_m = f"{params / 1e6:.1f}M" if params is not None else "—"
    return (
        f"| {entry['preset'] or '—'} | {entry['split'] or '—'} | {params_m} | "
        f"{fmt(entry['AP50'], '.4f')} | {fmt(entry['AP50_95'], '.4f')} | "
        f"{fmt(entry['num_images'])} | {fmt(entry['num_classes_with_gt'])} |"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports",
        nargs="*",
        default=None,
        help="Explicit report JSON paths. Defaults to outputs/evaluation_val_*.json.",
    )
    parser.add_argument("--outputs-dir", default="./outputs")
    parser.add_argument("--output-json", default="./outputs/ablation_summary.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.reports:
        report_paths = [Path(path) for path in args.reports]
    else:
        report_paths = sorted(Path(args.outputs_dir).glob("evaluation_val_*.json"))

    entries = [_load_report(path) for path in report_paths if path.is_file()]
    entries.sort(key=lambda item: (item["AP50_95"] or -1.0), reverse=True)

    print("| Preset | Split | Params | AP50 | AP50:95 | Images | Classes w/ GT |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for entry in entries:
        print(_format_row(entry))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"runs": entries}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\nAggregate written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
