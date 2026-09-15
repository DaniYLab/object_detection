"""Serializable schema for the Canonical Drawing Graph (CDG)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


CDG_SCHEMA_VERSION = "cdg-v0"


@dataclass(frozen=True)
class DrawingNode:
    """A geometry, semantic, text, or junction entity in a drawing."""

    id: str
    kind: str
    node_type: str
    geometry: Mapping[str, Any]
    semantic_id: int | None = None
    semantic_name: str | None = None
    confidence: float = 1.0
    attributes: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DrawingEdge:
    """A typed relation between two drawing nodes."""

    id: str
    relation: str
    source: str
    target: str
    confidence: float = 1.0
    attributes: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalDrawingGraph:
    """Versioned, JSON-serializable representation of a parsed drawing."""

    drawing_id: str
    nodes: list[DrawingNode]
    edges: list[DrawingEdge]
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = CDG_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "drawing_id": self.drawing_id,
            "metadata": dict(self.metadata),
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [asdict(edge) for edge in self.edges],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CanonicalDrawingGraph":
        return cls(
            schema_version=str(value.get("schema_version", CDG_SCHEMA_VERSION)),
            drawing_id=str(value["drawing_id"]),
            metadata=dict(value.get("metadata", {})),
            nodes=[DrawingNode(**dict(node)) for node in value.get("nodes", [])],
            edges=[DrawingEdge(**dict(edge)) for edge in value.get("edges", [])],
        )

    def write_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)

    @classmethod
    def read_json(cls, path: str | Path) -> "CanonicalDrawingGraph":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
