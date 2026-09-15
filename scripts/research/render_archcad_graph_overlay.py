"""Render CDG-v0 room and door-wall candidates over an ArchCAD PNG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.drawing_graph import CanonicalDrawingGraph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _center(geometry: dict) -> tuple[float, float] | None:
    if geometry.get("type") == "bbox":
        x0, y0, x1, y1 = geometry["xyxy"]
        return ((x0 + x1) / 2, (y0 + y1) / 2)
    if geometry.get("type") == "line":
        x0, y0 = geometry["start"]
        x1, y1 = geometry["end"]
        return ((x0 + x1) / 2, (y0 + y1) / 2)
    return None


def main() -> None:
    args = parse_args()
    graph = CanonicalDrawingGraph.read_json(args.graph)
    with Image.open(args.image) as source:
        image = source.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    nodes = {node.id: node for node in graph.nodes}

    for node in graph.nodes:
        if (
            node.semantic_name == "enclosed_wall_face_candidate"
            and node.geometry.get("type") == "polygon"
        ):
            points = [tuple(point) for point in node.geometry["points"]]
            draw.polygon(points, fill=(255, 80, 30, 55), outline=(220, 30, 20, 230), width=3)
        elif node.semantic_name == "wall" and node.geometry.get("type") == "line":
            draw.line(
                [tuple(node.geometry["start"]), tuple(node.geometry["end"])],
                fill=(0, 150, 255, 170),
                width=2,
            )

    for edge in graph.edges:
        if edge.relation != "candidate_attached_to":
            continue
        source_center = edge.attributes.get("door_point") or _center(
            dict(nodes[edge.source].geometry)
        )
        target_center = edge.attributes.get("wall_point") or _center(
            dict(nodes[edge.target].geometry)
        )
        if source_center is None or target_center is None:
            continue
        draw.line([source_center, target_center], fill=(40, 190, 70, 230), width=3)
        radius = 4
        draw.ellipse(
            [
                source_center[0] - radius,
                source_center[1] - radius,
                source_center[0] + radius,
                source_center[1] + radius,
            ],
            fill=(40, 190, 70, 255),
        )

    result = Image.alpha_composite(image, overlay).convert("RGB")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
