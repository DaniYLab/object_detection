"""Canonical primitive-level ArchCAD panoptic ground-truth contract."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from src.data.archcad_constants import (
    ARCHCAD_IGNORE_LABEL,
    ARCHCAD_STUFF_SEMANTIC_IDS,
    ARCHCAD_THING_SEMANTIC_IDS,
)


@dataclass(frozen=True)
class ArchCADSegmentTarget:
    """One thing instance or one per-class stuff segment over primitives."""

    semantic_id: int
    instance_key: str
    primitive_mask: np.ndarray

    def __post_init__(self) -> None:
        if self.primitive_mask.ndim != 1 or self.primitive_mask.dtype != np.bool_:
            raise ValueError("primitive_mask must be a one-dimensional bool array")


@dataclass(frozen=True)
class ArchCADPanopticSample:
    """All benchmark supervision for one ArchCAD chunk."""

    image_id: str
    semantic_labels: np.ndarray
    instance_ids: np.ndarray
    primitive_lengths: np.ndarray
    segments: tuple[ArchCADSegmentTarget, ...]
    geometries: tuple[Mapping[str, Any], ...]
    issues: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        primitive_count = len(self.semantic_labels)
        if self.semantic_labels.shape != (primitive_count,):
            raise ValueError("semantic_labels must be one-dimensional")
        if self.instance_ids.shape != (primitive_count,):
            raise ValueError("instance_ids must align with semantic_labels")
        if self.primitive_lengths.shape != (primitive_count,):
            raise ValueError("primitive_lengths must align with semantic_labels")
        if len(self.geometries) != primitive_count:
            raise ValueError("geometries must align with semantic_labels")
        if not np.isfinite(self.primitive_lengths).all() or np.any(
            self.primitive_lengths < 0
        ):
            raise ValueError("primitive_lengths must be finite and non-negative")
        if any(segment.primitive_mask.shape != (primitive_count,) for segment in self.segments):
            raise ValueError("segment masks must align with sample primitives")

    @property
    def primitive_count(self) -> int:
        return len(self.semantic_labels)

    @property
    def length_weights(self) -> np.ndarray:
        """Match the released DPSS evaluator's round(log(1 + length), 3)."""

        return np.round(np.log1p(self.primitive_lengths), 3)


def _point(value: object) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x, y = value[:2]
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return float(x), float(y)


def primitive_length(entity: Mapping[str, Any]) -> float:
    """Return geometric primitive length in native 980-coordinate units."""

    primitive_type = str(entity.get("type", "")).upper()
    if primitive_type == "LINE":
        start, end = _point(entity.get("start")), _point(entity.get("end"))
        return math.dist(start, end) if start is not None and end is not None else 0.0
    if primitive_type == "ARC":
        radius = float(entity.get("radius", 0.0))
        start_angle = float(entity.get("start_angle", 0.0))
        end_angle = float(entity.get("end_angle", 0.0))
        delta = abs(end_angle - start_angle) % 360.0
        if bool(entity.get("is_ccw")) and delta:
            delta = 360.0 - delta
        return max(radius, 0.0) * math.radians(delta)
    if primitive_type == "CIRCLE":
        return 2.0 * math.pi * max(float(entity.get("radius", 0.0)), 0.0)
    if primitive_type == "ELLIPSE":
        major = _point(entity.get("major_axis"))
        major_radius = math.hypot(*major) if major is not None else float(
            entity.get("radius_a", 0.0)
        )
        ratio = abs(float(entity.get("ratio", 0.0)))
        minor_radius = major_radius * ratio
        if major_radius <= 0 or minor_radius <= 0:
            return 0.0
        h = ((major_radius - minor_radius) ** 2) / (
            (major_radius + minor_radius) ** 2
        )
        return math.pi * (major_radius + minor_radius) * (
            1.0 + 3.0 * h / (10.0 + math.sqrt(4.0 - 3.0 * h))
        )
    return 0.0


def _canonical_geometry(entity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: entity[key]
        for key in (
            "type",
            "start",
            "end",
            "center",
            "radius",
            "start_angle",
            "end_angle",
            "major_axis",
            "ratio",
        )
        if key in entity
    }


def parse_archcad_panoptic_annotation(
    annotation: Mapping[str, Any], *, image_id: str
) -> ArchCADPanopticSample:
    """Parse raw entities without dropping stuff or exposing labels as inputs."""

    entities = annotation.get("entities")
    if not isinstance(entities, list):
        raise TypeError("ArchCAD annotation field 'entities' must be a list")

    semantic_labels: list[int] = []
    raw_instances: list[str | None] = []
    lengths: list[float] = []
    geometries: list[Mapping[str, Any]] = []
    issues: list[str] = []
    thing_ids = set(ARCHCAD_THING_SEMANTIC_IDS)
    stuff_ids = set(ARCHCAD_STUFF_SEMANTIC_IDS)
    for entity_index, entity in enumerate(entities):
        if not isinstance(entity, Mapping):
            issues.append(f"entity[{entity_index}] is not a mapping")
            continue
        raw_semantic = entity.get("semantic")
        semantic_id = (
            int(raw_semantic)
            if isinstance(raw_semantic, int)
            and (raw_semantic in thing_ids or raw_semantic in stuff_ids)
            else ARCHCAD_IGNORE_LABEL
        )
        raw_instance = entity.get("instance")
        instance_key = (
            str(raw_instance) if raw_instance is not None and raw_instance != "" else None
        )
        if semantic_id in thing_ids and instance_key is None:
            issues.append(
                f"entity[{entity_index}] thing semantic {semantic_id} has no instance"
            )
        semantic_labels.append(semantic_id)
        raw_instances.append(instance_key)
        lengths.append(primitive_length(entity))
        geometries.append(_canonical_geometry(entity))

    semantic_array = np.asarray(semantic_labels, dtype=np.int64)
    instance_array = np.full(len(semantic_array), -1, dtype=np.int64)
    instance_lookup: dict[tuple[int, str], int] = {}
    next_instance_id = 0
    for index, (semantic_id, instance_key) in enumerate(
        zip(semantic_labels, raw_instances)
    ):
        if semantic_id not in thing_ids or instance_key is None:
            continue
        key = (semantic_id, instance_key)
        if key not in instance_lookup:
            instance_lookup[key] = next_instance_id
            next_instance_id += 1
        instance_array[index] = instance_lookup[key]

    segment_keys: list[tuple[int, str]] = []
    for semantic_id in sorted(stuff_ids):
        if np.any(semantic_array == semantic_id):
            segment_keys.append((semantic_id, "stuff"))
    # The released DPSS `prepare_targets` groups every unique `(semantic_id,
    # instance_id)` pair and skips only `(background, -1)`. Preserve its
    # behavior for anomalous thing primitives with no instance: all such
    # primitives of a class in one chunk form the explicit `unassigned:-1`
    # segment, while the audit still reports the annotation defect.
    for semantic_id in sorted(thing_ids):
        if any(
            label == semantic_id and raw is None
            for label, raw in zip(semantic_labels, raw_instances)
        ):
            segment_keys.append((semantic_id, "unassigned:-1"))
    segment_keys.extend(
        (semantic_id, instance_key)
        for semantic_id, instance_key in sorted(instance_lookup)
    )
    segments = tuple(
        ArchCADSegmentTarget(
            semantic_id=semantic_id,
            instance_key=instance_key,
            primitive_mask=(
                semantic_array == semantic_id
                if instance_key == "stuff"
                else np.asarray(
                    [
                        label == semantic_id
                        and (
                            raw is None
                            if instance_key == "unassigned:-1"
                            else raw == instance_key
                        )
                        for label, raw in zip(semantic_labels, raw_instances)
                    ],
                    dtype=bool,
                )
            ),
        )
        for semantic_id, instance_key in segment_keys
    )
    return ArchCADPanopticSample(
        image_id=image_id,
        semantic_labels=semantic_array,
        instance_ids=instance_array,
        primitive_lengths=np.asarray(lengths, dtype=np.float64),
        segments=segments,
        geometries=tuple(geometries),
        issues=tuple(issues),
    )


def oracle_panoptic_prediction(sample: ArchCADPanopticSample) -> dict[str, Any]:
    """Return an identity prediction used only to verify benchmark metrics."""

    return {
        "image_id": sample.image_id,
        "semantic_labels": sample.semantic_labels.copy(),
        "instances": [
            {
                "semantic_id": segment.semantic_id,
                "score": 1.0,
                "mask": segment.primitive_mask.copy(),
            }
            for segment in sample.segments
        ],
    }
