"""Canonical FloorPlanCAD metadata parsing and validation.

Schema v2 keeps the legacy ``image_size``, ``svg_viewbox`` and ``instances``
fields so older consumers continue to work, while adding reproducibility data
(source hashes, a build fingerprint/settings, and parser statistics).
Bounding boxes use finite, clipped, half-open ``[x0, y0, x1, y1)`` pixels.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from PIL import Image

from src.data.constants import CLASS_TO_IDX, SEMANTIC_ID_TO_NAME

METADATA_SCHEMA_VERSION = 2
BBOX_CONVENTION = "xyxy_half_open"
StuffPolicy = Literal["exclude", "merge_by_class", "path_instances"]
STUFF_POLICIES: tuple[str, ...] = ("exclude", "merge_by_class", "path_instances")
UnknownPolicy = Literal["warn", "error"]
UNKNOWN_POLICIES: tuple[str, ...] = ("warn", "error")
CLASS_MAPPING_FINGERPRINT = hashlib.sha256(
    json.dumps(
        {str(key): value for key, value in sorted(SEMANTIC_ID_TO_NAME.items())},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


@dataclass(frozen=True)
class ValidationIssue:
    """One metadata validation issue."""

    path: str
    message: str
    severity: Literal["error", "warning"] = "error"


@dataclass
class MetadataValidationReport:
    """Structured result returned by :func:`validate_metadata`."""

    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def is_valid(self) -> bool:
        return self.valid

    def add_error(self, path: str, message: str) -> None:
        self.errors.append(ValidationIssue(path, message, "error"))

    def add_warning(self, path: str, message: str) -> None:
        self.warnings.append(ValidationIssue(path, message, "warning"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": [asdict(issue) for issue in self.errors],
            "warnings": [asdict(issue) for issue in self.warnings],
        }


class MetadataValidationError(ValueError):
    """Raised when metadata or source annotations fail validation."""

    def __init__(self, message: str, report: MetadataValidationReport | None = None) -> None:
        super().__init__(message)
        self.report = report


@dataclass
class ParseStats:
    """Counters recorded in every schema-v2 metadata file."""

    paths_total: int = 0
    paths_encoded: int = 0
    paths_invalid: int = 0
    paths_missing_semantic_id: int = 0
    paths_unknown_semantic_id: int = 0
    stuff_paths_excluded: int = 0
    groups_total: int = 0
    instances_emitted: int = 0
    instances_too_small: int = 0
    instances_outside_image: int = 0
    instances_clipped: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _json_fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def build_fingerprint(
    image_sha256: str | None,
    svg_sha256: str | None,
    settings: Mapping[str, Any],
) -> str:
    """Fingerprint source bytes and all target-affecting build settings.

    The emitted instance summary is stored and validated separately by split
    manifests; keeping this fingerprint source/build based preserves readability
    of earlier schema-v2 files.
    """

    return _json_fingerprint(
        {
            "schema_version": METADATA_SCHEMA_VERSION,
            "image_sha256": image_sha256,
            "svg_sha256": svg_sha256,
            "settings": dict(settings),
        }
    )


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _parse_viewbox(value: str | None) -> tuple[float, float, float, float]:
    if value is None:
        raise MetadataValidationError("SVG root is missing a viewBox")
    parts = [part for part in re.split(r"[\s,]+", value.strip()) if part]
    if len(parts) != 4:
        raise MetadataValidationError(f"SVG viewBox must contain four numbers, got {value!r}")
    try:
        x, y, width, height = (float(part) for part in parts)
    except ValueError as exc:
        raise MetadataValidationError(f"SVG viewBox contains a non-number: {value!r}") from exc
    if not all(math.isfinite(v) for v in (x, y, width, height)):
        raise MetadataValidationError("SVG viewBox values must be finite")
    if width <= 0 or height <= 0:
        raise MetadataValidationError("SVG viewBox width and height must be positive")
    return x, y, width, height


def _get_attribute(element: ET.Element, name: str) -> str | None:
    direct = element.get(name)
    if direct is not None:
        return direct
    for key, value in element.attrib.items():
        if key.rsplit("}", 1)[-1] == name:
            return value
    return None


_PATH_TOKEN_RE = re.compile(
    r"[AaCcHhLlMmQqSsTtVvZz]|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)
_PATH_ARITY = {
    "M": 2,
    "L": 2,
    "H": 1,
    "V": 1,
    "C": 6,
    "S": 4,
    "Q": 4,
    "T": 2,
    "A": 7,
    "Z": 0,
}


def _fallback_path_bbox(path_data: str) -> tuple[float, float, float, float] | None:
    """Dependency-free fallback for simple SVG paths.

    ``svgpathtools`` is used when installed and computes exact curve extrema. This
    fallback fully tracks line/end/control coordinates and conservatively bounds
    curves and arcs. It exists so metadata validation and synthetic tests do not
    require an optional parser dependency.
    """

    tokens = _PATH_TOKEN_RE.findall(path_data)
    if not tokens:
        return None

    points: list[tuple[float, float]] = []
    current = (0.0, 0.0)
    subpath_start = current
    command: str | None = None
    index = 0

    def add(x: float, y: float) -> None:
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("path has non-finite coordinates")
        points.append((x, y))

    while index < len(tokens):
        token = tokens[index]
        if token.isalpha():
            command = token
            index += 1
            if command.upper() == "Z":
                current = subpath_start
                add(*current)
                command = None
                continue
        if command is None:
            raise ValueError("path data is missing a command")

        upper = command.upper()
        relative = command.islower()
        arity = _PATH_ARITY[upper]
        if index + arity > len(tokens) or any(t.isalpha() for t in tokens[index : index + arity]):
            raise ValueError(f"path command {command!r} has incomplete coordinates")
        values = [float(v) for v in tokens[index : index + arity]]
        index += arity
        cx, cy = current

        def absolute_pair(x: float, y: float) -> tuple[float, float]:
            return (x + cx, y + cy) if relative else (x, y)

        if upper in {"M", "L", "T"}:
            current = absolute_pair(values[0], values[1])
            add(*current)
            if upper == "M":
                subpath_start = current
                command = "l" if relative else "L"
        elif upper == "H":
            current = (values[0] + cx if relative else values[0], cy)
            add(*current)
        elif upper == "V":
            current = (cx, values[0] + cy if relative else values[0])
            add(*current)
        elif upper == "C":
            p1 = absolute_pair(values[0], values[1])
            p2 = absolute_pair(values[2], values[3])
            current = absolute_pair(values[4], values[5])
            add(*p1)
            add(*p2)
            add(*current)
        elif upper in {"S", "Q"}:
            control = absolute_pair(values[0], values[1])
            current = absolute_pair(values[2], values[3])
            add(*control)
            add(*current)
        elif upper == "A":
            rx, ry = abs(values[0]), abs(values[1])
            end = absolute_pair(values[5], values[6])
            # A conservative bound; exact extrema are supplied by svgpathtools.
            add(cx - rx, cy - ry)
            add(cx + rx, cy + ry)
            add(end[0] - rx, end[1] - ry)
            add(end[0] + rx, end[1] + ry)
            current = end
            add(*current)
        else:  # pragma: no cover - guarded by token regex/arity table
            raise ValueError(f"unsupported path command {command!r}")

    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def parse_path_bbox(path_data: str) -> tuple[float, float, float, float] | None:
    """Return an SVG path bounding box as ``(x0, y0, x1, y1)``."""

    try:
        from svgpathtools import parse_path  # type: ignore[import-not-found]
    except ImportError:
        return _fallback_path_bbox(path_data)

    path = parse_path(path_data)
    if not path:
        return None
    xmin, xmax, ymin, ymax = path.bbox()
    bbox = (float(xmin), float(ymin), float(xmax), float(ymax))
    if not all(math.isfinite(value) for value in bbox):
        raise ValueError("path bounding box contains non-finite values")
    return bbox


def _merge_bboxes(bboxes: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    return (
        min(float(box[0]) for box in bboxes),
        min(float(box[1]) for box in bboxes),
        max(float(box[2]) for box in bboxes),
        max(float(box[3]) for box in bboxes),
    )


def svg_bbox_to_pixels(
    bbox: Sequence[float],
    viewbox: Sequence[float],
    image_size: Sequence[int],
) -> tuple[float, float, float, float]:
    """Map an SVG bbox into pixels, including a non-zero viewBox origin."""

    x0, y0, x1, y1 = (float(value) for value in bbox)
    vb_x, vb_y, vb_w, vb_h = (float(value) for value in viewbox)
    image_w, image_h = (int(value) for value in image_size)
    scale_x = image_w / vb_w
    scale_y = image_h / vb_h
    return (
        (x0 - vb_x) * scale_x,
        (y0 - vb_y) * scale_y,
        (x1 - vb_x) * scale_x,
        (y1 - vb_y) * scale_y,
    )


def _source_entry(path: Path, sha256: str) -> dict[str, str]:
    return {"path": path.name, "sha256": sha256}


def parse_svg_metadata(
    svg_path: str | Path,
    image_path: str | Path | None = None,
    *,
    min_size: float = 8.0,
    stuff_policy: StuffPolicy = "exclude",
    unknown_policy: UnknownPolicy = "warn",
    strict: bool = False,
) -> dict[str, Any]:
    """Parse one FloorPlanCAD SVG into canonical schema-v2 metadata.

    Thing paths are merged by ``(semantic-id, instance-id)``. Paths with an
    instance id of ``-1`` follow ``stuff_policy``:

    - ``exclude``: omit them from detection targets;
    - ``merge_by_class``: emit one union bbox per semantic class;
    - ``path_instances``: emit one bbox per SVG path.

    Unknown semantic IDs are skipped and recorded when ``unknown_policy="warn"``;
    ``unknown_policy="error"`` rejects the source SVG.
    """

    svg_path = Path(svg_path)
    image_path = Path(image_path) if image_path is not None else svg_path.with_suffix(".png")
    if stuff_policy not in STUFF_POLICIES:
        raise ValueError(f"Unknown stuff_policy {stuff_policy!r}; expected one of {STUFF_POLICIES}")
    if unknown_policy not in UNKNOWN_POLICIES:
        raise ValueError(
            f"Unknown unknown_policy {unknown_policy!r}; expected one of {UNKNOWN_POLICIES}"
        )
    if not _is_finite_number(min_size) or float(min_size) < 0:
        raise ValueError("min_size must be a finite, non-negative number")
    if not svg_path.is_file():
        raise FileNotFoundError(svg_path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    try:
        tree = ET.parse(svg_path)
    except ET.ParseError as exc:
        raise MetadataValidationError(f"Invalid SVG XML in {svg_path}: {exc}") from exc
    root = tree.getroot()
    viewbox = _parse_viewbox(root.get("viewBox") or root.get("viewbox"))

    try:
        with Image.open(image_path) as image:
            image_w, image_h = image.size
            image.verify()
    except Exception as exc:
        raise MetadataValidationError(f"Cannot read image {image_path}: {exc}") from exc
    if image_w <= 0 or image_h <= 0:
        raise MetadataValidationError("Image dimensions must be positive")

    stats = ParseStats()
    unknown_semantic_messages: list[str] = []
    # key -> {sid, iid, bboxes}; insertion order follows source path order.
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}

    for path_index, element in enumerate(root.iter()):
        path_data = _get_attribute(element, "d")
        if path_data is None:
            continue
        stats.paths_total += 1
        semantic_text = _get_attribute(element, "semantic-id")
        if semantic_text is None:
            stats.paths_missing_semantic_id += 1
            stats.warnings.append(f"path[{path_index}] is missing semantic-id")
            continue
        try:
            semantic_id = int(semantic_text)
        except (TypeError, ValueError):
            stats.paths_invalid += 1
            stats.warnings.append(f"path[{path_index}] has invalid semantic-id {semantic_text!r}")
            continue
        if semantic_id not in SEMANTIC_ID_TO_NAME:
            stats.paths_unknown_semantic_id += 1
            message = f"path[{path_index}] has unknown semantic-id {semantic_id}"
            stats.warnings.append(message)
            unknown_semantic_messages.append(message)
            continue

        instance_text = _get_attribute(element, "instance-id")
        try:
            instance_id = int(instance_text) if instance_text not in (None, "") else -1
        except (TypeError, ValueError):
            stats.paths_invalid += 1
            stats.warnings.append(f"path[{path_index}] has invalid instance-id {instance_text!r}")
            continue

        if _get_attribute(element, "transform") is not None:
            stats.warnings.append(f"path[{path_index}] transform is unsupported and was ignored")

        try:
            bbox = parse_path_bbox(path_data)
        except Exception as exc:
            stats.paths_invalid += 1
            stats.warnings.append(f"path[{path_index}] could not be parsed: {exc}")
            continue
        if bbox is None or not all(math.isfinite(float(value)) for value in bbox):
            stats.paths_invalid += 1
            stats.warnings.append(f"path[{path_index}] has no finite bounds")
            continue
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            stats.paths_invalid += 1
            stats.warnings.append(f"path[{path_index}] has empty bounds")
            continue

        if instance_id == -1:
            if stuff_policy == "exclude":
                stats.stuff_paths_excluded += 1
                continue
            if stuff_policy == "merge_by_class":
                key = ("stuff", semantic_id)
            else:
                key = ("stuff_path", semantic_id, path_index)
        else:
            key = ("thing", semantic_id, instance_id)
        group = groups.setdefault(
            key,
            {"semantic_id": semantic_id, "instance_id": instance_id, "bboxes": []},
        )
        group["bboxes"].append(bbox)
        stats.paths_encoded += 1

    if unknown_semantic_messages and unknown_policy == "error":
        report = MetadataValidationReport(
            errors=[
                ValidationIssue("svg.paths", message, "error")
                for message in unknown_semantic_messages
            ]
        )
        raise MetadataValidationError(
            f"Unknown semantic IDs in {svg_path}: " + "; ".join(unknown_semantic_messages),
            report,
        )

    stats.groups_total = len(groups)
    instances: list[dict[str, Any]] = []
    for group in groups.values():
        semantic_id = int(group["semantic_id"])
        instance_id = int(group["instance_id"])
        svg_bbox = _merge_bboxes(group["bboxes"])
        raw_px = svg_bbox_to_pixels(svg_bbox, viewbox, (image_w, image_h))
        if not all(math.isfinite(value) for value in raw_px):
            stats.paths_invalid += len(group["bboxes"])
            stats.warnings.append(f"group semantic={semantic_id}, instance={instance_id} is non-finite")
            continue

        x0 = max(0.0, min(float(image_w), raw_px[0]))
        y0 = max(0.0, min(float(image_h), raw_px[1]))
        x1 = max(0.0, min(float(image_w), raw_px[2]))
        y1 = max(0.0, min(float(image_h), raw_px[3]))
        if (x0, y0, x1, y1) != raw_px:
            stats.instances_clipped += 1
            stats.warnings.append(
                f"group semantic={semantic_id}, instance={instance_id} exceeded image bounds and was clipped"
            )
        if x1 <= x0 or y1 <= y0:
            stats.instances_outside_image += 1
            stats.warnings.append(
                f"group semantic={semantic_id}, instance={instance_id} lies outside the image"
            )
            continue
        if (x1 - x0) < float(min_size) or (y1 - y0) < float(min_size):
            stats.instances_too_small += 1
            continue

        class_name = SEMANTIC_ID_TO_NAME[semantic_id]
        instances.append(
            {
                "class": class_name,
                "class_id": CLASS_TO_IDX[class_name],
                "semantic_id": semantic_id,
                "instance_id": instance_id,
                "bbox_px": [x0, y0, x1, y1],
            }
        )

    stats.instances_emitted = len(instances)
    settings = {
        "parser": "floorplancad_metadata_v2",
        "min_size_px": float(min_size),
        "min_size": float(min_size),
        "stuff_policy": stuff_policy,
        "unknown_policy": unknown_policy,
        "bbox_convention": BBOX_CONVENTION,
        "class_mapping_fingerprint": CLASS_MAPPING_FINGERPRINT,
    }
    image_sha256 = sha256_file(image_path)
    svg_sha256 = sha256_file(svg_path)
    fingerprint = build_fingerprint(image_sha256, svg_sha256, settings)
    metadata: dict[str, Any] = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "image": {
            "path": image_path.name,
            "width": image_w,
            "height": image_h,
            "sha256": image_sha256,
        },
        "image_size": [image_w, image_h],
        "svg_viewbox": list(viewbox),
        "source": {
            "image": _source_entry(image_path, image_sha256),
            "svg": _source_entry(svg_path, svg_sha256),
            "image_sha256": image_sha256,
            "svg_sha256": svg_sha256,
            "fingerprint": fingerprint,
        },
        # Flat aliases make provenance easy to inspect and tolerate early v2 readers.
        "source_sha256": {"image": image_sha256, "svg": svg_sha256},
        "fingerprint": fingerprint,
        "build": settings,
        "stats": stats.to_dict(),
        "num_instances": len(instances),
        "instances": instances,
    }

    validation = validate_metadata(metadata)
    if not validation.valid:
        raise MetadataValidationError(
            f"Generated invalid metadata for {svg_path}: "
            + "; ".join(f"{issue.path}: {issue.message}" for issue in validation.errors),
            validation,
        )
    if strict and (stats.warnings or validation.warnings):
        strict_report = MetadataValidationReport(
            errors=[
                ValidationIssue("stats.warnings", warning, "error") for warning in stats.warnings
            ],
            warnings=validation.warnings,
        )
        raise MetadataValidationError(
            f"Strict parsing rejected {svg_path}: " + "; ".join(stats.warnings),
            strict_report,
        )
    return metadata


def adapt_legacy_metadata(
    metadata: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
) -> dict[str, Any]:
    """Adapt legacy flat metadata to schema v2 without touching the source file."""

    if metadata.get("schema_version") == METADATA_SCHEMA_VERSION:
        return copy.deepcopy(dict(metadata))

    adapted = copy.deepcopy(dict(metadata))
    image_size = adapted.get("image_size", [0, 0])
    if not isinstance(image_size, Sequence) or isinstance(image_size, (str, bytes)) or len(image_size) != 2:
        image_size = [0, 0]
    image_w, image_h = image_size
    source_name = Path(source_path).name if source_path is not None else None
    instances: list[dict[str, Any]] = []
    for raw_instance in adapted.get("instances", []):
        if not isinstance(raw_instance, Mapping):
            instances.append(copy.deepcopy(raw_instance))
            continue
        instance = copy.deepcopy(dict(raw_instance))
        class_name = instance.get("class")
        if "class_id" not in instance and class_name in CLASS_TO_IDX:
            instance["class_id"] = CLASS_TO_IDX[class_name]
        if "semantic_id" not in instance:
            semantic_id = next(
                (sid for sid, name in SEMANTIC_ID_TO_NAME.items() if name == class_name),
                None,
            )
            instance["semantic_id"] = semantic_id
        instances.append(instance)

    settings = {
        "parser": "legacy_adapter",
        "min_size_px": None,
        "min_size": None,
        "stuff_policy": "legacy",
        "bbox_convention": BBOX_CONVENTION,
        "class_mapping_fingerprint": CLASS_MAPPING_FINGERPRINT,
    }
    content_fingerprint = _json_fingerprint(
        {
            "image_size": image_size,
            "svg_viewbox": adapted.get("svg_viewbox"),
            "instances": instances,
        }
    )
    return {
        "schema_version": METADATA_SCHEMA_VERSION,
        "image": {
            "path": None,
            "width": image_w,
            "height": image_h,
            "sha256": None,
        },
        "image_size": list(image_size),
        "svg_viewbox": copy.deepcopy(adapted.get("svg_viewbox", [0, 0, image_w, image_h])),
        "source": {
            "metadata": source_name,
            "image_sha256": None,
            "svg_sha256": None,
            "fingerprint": content_fingerprint,
        },
        "source_sha256": {"image": None, "svg": None},
        "fingerprint": content_fingerprint,
        "build": settings,
        "stats": {
            "legacy_adapted": True,
            "instances_emitted": len(instances),
            "warnings": ["metadata was adapted from legacy schema"],
        },
        "num_instances": len(instances),
        "instances": instances,
    }


def _validate_digest(
    report: MetadataValidationReport,
    value: Any,
    path: str,
    *,
    required: bool,
) -> None:
    if value is None and not required:
        report.add_warning(path, "source digest is unavailable (legacy metadata)")
        return
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        report.add_error(path, "must be a lowercase 64-character SHA-256 digest")


def validate_metadata(metadata: Mapping[str, Any]) -> MetadataValidationReport:
    """Validate schema, finite values, class ids, and half-open image bounds."""

    report = MetadataValidationReport()
    if not isinstance(metadata, Mapping):
        report.add_error("$", "metadata must be an object")
        return report

    if metadata.get("schema_version") != METADATA_SCHEMA_VERSION:
        report.add_error(
            "schema_version",
            f"must equal {METADATA_SCHEMA_VERSION}; use adapt_legacy_metadata for old files",
        )

    image_size = metadata.get("image_size")
    image_w = image_h = None
    if (
        not isinstance(image_size, Sequence)
        or isinstance(image_size, (str, bytes))
        or len(image_size) != 2
        or any(not _is_finite_number(value) for value in image_size)
    ):
        report.add_error("image_size", "must contain two finite numbers [width, height]")
    else:
        image_w, image_h = float(image_size[0]), float(image_size[1])
        if image_w <= 0 or image_h <= 0:
            report.add_error("image_size", "width and height must be positive")

    viewbox = metadata.get("svg_viewbox")
    if (
        not isinstance(viewbox, Sequence)
        or isinstance(viewbox, (str, bytes))
        or len(viewbox) != 4
        or any(not _is_finite_number(value) for value in viewbox)
    ):
        report.add_error("svg_viewbox", "must contain four finite numbers")
    elif float(viewbox[2]) <= 0 or float(viewbox[3]) <= 0:
        report.add_error("svg_viewbox", "width and height must be positive")

    build = metadata.get("build")
    if not isinstance(build, Mapping):
        report.add_error("build", "must be an object")
    else:
        if build.get("bbox_convention") != BBOX_CONVENTION:
            report.add_error("build.bbox_convention", f"must be {BBOX_CONVENTION!r}")
        if build.get("class_mapping_fingerprint") not in (None, CLASS_MAPPING_FINGERPRINT):
            report.add_error("build.class_mapping_fingerprint", "does not match canonical classes")
        parser = build.get("parser")
        stuff_policy = build.get("stuff_policy")
        if parser != "legacy_adapter" and stuff_policy not in STUFF_POLICIES:
            report.add_error("build.stuff_policy", f"must be one of {STUFF_POLICIES}")
        if parser != "legacy_adapter":
            if "unknown_policy" not in build:
                report.add_warning(
                    "build.unknown_policy",
                    "is missing from earlier schema-v2 metadata; regenerate metadata to record it",
                )
            elif build.get("unknown_policy") not in UNKNOWN_POLICIES:
                report.add_error(
                    "build.unknown_policy", f"must be one of {UNKNOWN_POLICIES}"
                )

    image = metadata.get("image")
    if not isinstance(image, Mapping):
        report.add_error("image", "must be an object")
    elif image_w is not None and image_h is not None:
        if image.get("width") != image_size[0] or image.get("height") != image_size[1]:
            report.add_error("image", "width/height must match image_size")

    source = metadata.get("source")
    if not isinstance(source, Mapping):
        report.add_error("source", "must be an object")
    else:
        is_legacy = build.get("parser") == "legacy_adapter" if isinstance(build, Mapping) else False
        image_digest = source.get("image_sha256")
        svg_digest = source.get("svg_sha256")
        source_fingerprint = source.get("fingerprint")
        _validate_digest(report, image_digest, "source.image_sha256", required=not is_legacy)
        _validate_digest(report, svg_digest, "source.svg_sha256", required=not is_legacy)
        _validate_digest(report, source_fingerprint, "source.fingerprint", required=True)
        if isinstance(image, Mapping) and image.get("sha256") != image_digest:
            report.add_error("image.sha256", "must match source.image_sha256")
        nested_image = source.get("image")
        if isinstance(nested_image, Mapping) and nested_image.get("sha256") != image_digest:
            report.add_error("source.image.sha256", "must match source.image_sha256")
        nested_svg = source.get("svg")
        if isinstance(nested_svg, Mapping) and nested_svg.get("sha256") != svg_digest:
            report.add_error("source.svg.sha256", "must match source.svg_sha256")
        if metadata.get("fingerprint") != source_fingerprint:
            report.add_error("fingerprint", "must match source.fingerprint")
        if not is_legacy and isinstance(build, Mapping) and isinstance(image_digest, str) and isinstance(svg_digest, str):
            expected_fingerprint = build_fingerprint(image_digest, svg_digest, build)
            if source_fingerprint != expected_fingerprint:
                report.add_error("source.fingerprint", "does not match source hashes and build settings")

    stats = metadata.get("stats")
    if not isinstance(stats, Mapping):
        report.add_error("stats", "must be an object")
    else:
        stored_warnings = stats.get("warnings", [])
        if not isinstance(stored_warnings, list):
            report.add_error("stats.warnings", "must be an array")
        else:
            for index, warning in enumerate(stored_warnings):
                report.add_warning(f"stats.warnings[{index}]", str(warning))

    instances = metadata.get("instances")
    if not isinstance(instances, list):
        report.add_error("instances", "must be an array")
        instances = []
    if metadata.get("num_instances") != len(instances):
        report.add_error("num_instances", "must equal len(instances)")

    for index, instance in enumerate(instances):
        prefix = f"instances[{index}]"
        if not isinstance(instance, Mapping):
            report.add_error(prefix, "must be an object")
            continue
        class_name = instance.get("class")
        if class_name not in CLASS_TO_IDX:
            report.add_error(f"{prefix}.class", f"unknown class {class_name!r}")
        elif instance.get("class_id") != CLASS_TO_IDX[class_name]:
            report.add_error(f"{prefix}.class_id", "does not match canonical class mapping")
        semantic_id = instance.get("semantic_id")
        if semantic_id is not None and SEMANTIC_ID_TO_NAME.get(semantic_id) != class_name:
            report.add_error(f"{prefix}.semantic_id", "does not match class")
        if not isinstance(instance.get("instance_id"), int):
            report.add_error(f"{prefix}.instance_id", "must be an integer")

        bbox = instance.get("bbox_px")
        if (
            not isinstance(bbox, Sequence)
            or isinstance(bbox, (str, bytes))
            or len(bbox) != 4
            or any(not _is_finite_number(value) for value in bbox)
        ):
            report.add_error(f"{prefix}.bbox_px", "must contain four finite numbers")
            continue
        x0, y0, x1, y1 = (float(value) for value in bbox)
        if x1 <= x0 or y1 <= y0:
            report.add_error(f"{prefix}.bbox_px", "must be a non-empty half-open box")
        if image_w is not None and image_h is not None:
            if x0 < 0 or y0 < 0 or x1 > image_w or y1 > image_h:
                report.add_error(
                    f"{prefix}.bbox_px",
                    f"must lie within [0, {image_w}] x [0, {image_h}]",
                )
    return report


def validate_metadata_build_settings(
    metadata: Mapping[str, Any],
    *,
    min_size: float,
    stuff_policy: StuffPolicy,
    unknown_policy: UnknownPolicy,
) -> MetadataValidationReport:
    """Compare stored target-affecting settings with a requested build.

    This is intentionally separate from schema validation: an existing metadata
    file can be internally valid while representing a different benchmark.
    """

    if not _is_finite_number(min_size) or float(min_size) < 0:
        raise ValueError("min_size must be a finite, non-negative number")
    if stuff_policy not in STUFF_POLICIES:
        raise ValueError(f"Unknown stuff_policy {stuff_policy!r}; expected one of {STUFF_POLICIES}")
    if unknown_policy not in UNKNOWN_POLICIES:
        raise ValueError(
            f"Unknown unknown_policy {unknown_policy!r}; expected one of {UNKNOWN_POLICIES}"
        )

    report = MetadataValidationReport()
    build = metadata.get("build")
    if not isinstance(build, Mapping):
        report.add_error("build", "cannot compare requested settings because build is not an object")
        return report

    min_size_path = "build.min_size_px" if "min_size_px" in build else "build.min_size"
    stored_min_size = build.get("min_size_px", build.get("min_size"))
    if not _is_finite_number(stored_min_size):
        report.add_error(
            min_size_path,
            f"does not record a finite value matching requested min_size={float(min_size)!r}",
        )
    elif float(stored_min_size) != float(min_size):
        report.add_error(
            min_size_path,
            f"stored value {float(stored_min_size)!r} does not match requested {float(min_size)!r}",
        )

    stored_stuff_policy = build.get("stuff_policy")
    if stored_stuff_policy != stuff_policy:
        report.add_error(
            "build.stuff_policy",
            f"stored value {stored_stuff_policy!r} does not match requested {stuff_policy!r}",
        )

    stored_unknown_policy = build.get("unknown_policy")
    if stored_unknown_policy != unknown_policy:
        report.add_error(
            "build.unknown_policy",
            f"stored value {stored_unknown_policy!r} does not match requested {unknown_policy!r}",
        )
    return report


def validate_metadata_sources(
    metadata: Mapping[str, Any],
    image_path: str | Path,
    svg_path: str | Path,
) -> MetadataValidationReport:
    """Validate metadata plus the current image/SVG bytes against stored hashes."""

    report = validate_metadata(metadata)
    image_path = Path(image_path)
    svg_path = Path(svg_path)
    source = metadata.get("source")
    build = metadata.get("build")
    if not image_path.is_file():
        report.add_error("source.image", f"file does not exist: {image_path}")
    if not svg_path.is_file():
        report.add_error("source.svg", f"file does not exist: {svg_path}")
    if not isinstance(source, Mapping):
        return report
    if isinstance(build, Mapping) and build.get("parser") == "legacy_adapter":
        report.add_warning("source", "legacy metadata has no source hashes to validate")
        return report
    if image_path.is_file():
        actual_image_digest = sha256_file(image_path)
        if source.get("image_sha256") != actual_image_digest:
            report.add_error("source.image_sha256", "does not match the current image file")
        try:
            with Image.open(image_path) as image:
                actual_size = list(image.size)
            if metadata.get("image_size") != actual_size:
                report.add_error("image_size", "does not match the current image file")
        except Exception as exc:
            report.add_error("source.image", f"cannot read image: {exc}")
    if svg_path.is_file():
        actual_svg_digest = sha256_file(svg_path)
        if source.get("svg_sha256") != actual_svg_digest:
            report.add_error("source.svg_sha256", "does not match the current SVG file")
    return report


def load_metadata(
    path: str | Path,
    *,
    allow_legacy: bool = True,
    strict: bool = False,
) -> dict[str, Any]:
    """Load metadata, adapt legacy files in memory, and validate the result."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("schema_version") != METADATA_SCHEMA_VERSION:
        if not allow_legacy:
            raise MetadataValidationError(f"Legacy metadata is not allowed: {path}")
        metadata = adapt_legacy_metadata(metadata, source_path=path)

    report = validate_metadata(metadata)
    if report.errors or (strict and report.warnings):
        issues = report.errors + (report.warnings if strict else [])
        raise MetadataValidationError(
            f"Invalid metadata {path}: "
            + "; ".join(f"{issue.path}: {issue.message}" for issue in issues),
            report,
        )
    return metadata


# Compatibility names used by older scripts and external callers.
parse_svg = parse_svg_metadata
validate_metadata_v2 = validate_metadata
upgrade_legacy_metadata = adapt_legacy_metadata
