"""Audit SVG stroke geometry for the dual-pathway (image + vector) design.

Answers the three design questions raised in the research brainstorm:

1. Which path commands actually occur in FloorPlanCAD? (assumption: only
   M/L/A — if Bezier or H/V commands exist we must handle them in the
   primitive tokenizer)
2. How many primitives does one drawing have? (drives ``N_max`` and the
   truncate-vs-pad policy)
3. How long are connected stroke chains (M followed by L runs)? (decides
   whether polyline grouping is worth it)

Read-only: no source annotation is modified. Run while training jobs are
active — this script never imports ``src.data.dataset`` mutations.

Usage:
    python scripts/data/audit_strokes.py --manifest data/FloorPlanCAD_original/splits.json
        [--limit N] [--split val] [--output outputs/stroke_audit.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.metadata import _get_attribute, _parse_viewbox  # noqa: E402

# Same tokenizer contract as metadata.py
_PATH_TOKEN_RE = re.compile(
    r"[AaCcHhLlMmQqSsTtVvZz]|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)
_PATH_ARITY = {
    "M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7, "Z": 0,
}


def iter_path_elements(svg_path: Path):
    tree = ET.parse(svg_path)
    root = tree.getroot()
    for element in root.iter():
        if _get_attribute(element, "d") is not None:
            yield element


def tokenize_path_commands(path_data: str) -> list[tuple[str, list[float]]]:
    """Flatten an SVG path ``d`` string into (command, values) pairs.

    Implicit repeats are expanded (``M x y L ...`` style) so counts reflect
    actual primitives, and ``Z`` is kept as its own command.
    """

    tokens = _PATH_TOKEN_RE.findall(path_data)
    result: list[tuple[str, list[float]]] = []
    command: str | None = None
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.isalpha():
            command = token
            index += 1
            if command.upper() == "Z":
                result.append(("Z", []))
                command = None
                continue
        if command is None:
            break
        upper = command.upper()
        arity = _PATH_ARITY[upper]
        if index + arity > len(tokens):
            break
        values = [float(v) for v in tokens[index : index + arity]]
        result.append((command, values))
        index += arity
        if upper == "M":
            command = "l" if command.islower() else "L"
    return result


def stroke_stats(svg_path: Path) -> dict[str, Any]:
    viewbox = _parse_viewbox(
        _get_attribute(svg_root(svg_path), "viewBox") or _get_attribute(svg_root(svg_path), "viewbox")
    )
    command_counts: Counter[str] = Counter()
    num_elements = 0
    num_primitives = 0
    polyline_lengths: list[int] = []  # consecutive L-run lengths (incl. the M)
    current_run = 0
    parse_failures = 0

    for element in iter_path_elements(svg_path):
        path_data = _get_attribute(element, "d")
        if path_data is None:
            continue
        num_elements += 1
        try:
            commands = tokenize_path_commands(path_data)
        except Exception:
            parse_failures += 1
            continue
        current_run = 0
        for command, _values in commands:
            command_counts[command.upper()] += 1
            num_primitives += 1
            if command.upper() in ("L", "M"):
                current_run += 1
            else:
                if current_run > 1:
                    polyline_lengths.append(current_run)
                current_run = 0
        if current_run > 1:
            polyline_lengths.append(current_run)

    return {
        "num_path_elements": num_elements,
        "num_primitives": num_primitives,
        "command_counts": dict(command_counts),
        "polyline_run_lengths": polyline_lengths,
        "parse_failures": parse_failures,
        "viewbox": list(viewbox),
    }


_cache_root: dict[Path, ET.Element] = {}


def svg_root(svg_path: Path) -> ET.Element:
    if svg_path not in _cache_root:
        _cache_root[svg_path] = ET.parse(svg_path).getroot()
    return _cache_root[svg_path]


def quantiles(values: list[int], points=(0.5, 0.9, 0.95, 0.99, 1.0)) -> dict[str, int]:
    if not values:
        return {}
    ordered = sorted(values)
    out = {}
    for point in points:
        idx = min(int(point * len(ordered)) - 1, len(ordered) - 1)
        out[f"p{int(point * 100)}"] = ordered[max(idx, 0)]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/FloorPlanCAD_original/splits.json")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=0, help="Audit at most N SVGs (0 = all)")
    parser.add_argument("--output", default="outputs/stroke_audit.json")
    args = parser.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    data_root = Path(args.manifest).parent
    # Manifest source splits are train/test; "val" maps to the train split
    # (val is carved out of train at dataset-construction time).
    source_split = "train" if args.split == "val" else args.split
    svg_paths = [
        data_root / Path(entry["image_path"]).with_suffix(".svg")
        for entry in manifest["image_index"]
        if entry["source_split"] == source_split
    ]
    if args.limit:
        svg_paths = svg_paths[: args.limit]

    global_command_counts: Counter[str] = Counter()
    all_primitive_counts: list[int] = []
    all_element_counts: list[int] = []
    all_polyline_runs: list[int] = list()
    total_parse_failures = 0
    files_failed = 0

    for svg_path in svg_paths:
        try:
            stats = stroke_stats(svg_path)
        except Exception:
            files_failed += 1
            continue
        global_command_counts.update(stats["command_counts"])
        all_primitive_counts.append(stats["num_primitives"])
        all_element_counts.append(stats["num_path_elements"])
        all_polyline_runs.extend(stats["polyline_run_lengths"])
        total_parse_failures += stats["parse_failures"]

    total_svgs = len(svg_paths)
    primitive_q = quantiles(all_primitive_counts)
    # Coverage of N_max candidates
    coverage = {}
    for n_max in (128, 256, 512, 768, 1024, 1536, 2048):
        covered = sum(1 for c in all_primitive_counts if c <= n_max)
        coverage[str(n_max)] = covered / total_svgs if total_svgs else 0.0

    report = {
        "split": args.split,
        "num_svgs": total_svgs,
        "files_failed": files_failed,
        "total_parse_failures": total_parse_failures,
        "command_counts": dict(global_command_counts),
        "primitives_per_svg": {
            "mean": sum(all_primitive_counts) / total_svgs if total_svgs else 0.0,
            **primitive_q,
        },
        "path_elements_per_svg": {
            "mean": sum(all_element_counts) / total_svgs if total_svgs else 0.0,
            **quantiles(all_element_counts),
        },
        "polyline_run_length": {
            "count": len(all_polyline_runs),
            **quantiles(all_polyline_runs),
        },
        "n_max_coverage": coverage,
    }

    print(json.dumps(report, indent=2))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nAudit written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
