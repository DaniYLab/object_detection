"""Image-level indexing and deterministic train/val/test manifests."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.data.constants import CLASS_TO_IDX
from src.data.metadata import load_metadata

DEFAULT_SEED = 1337
DEFAULT_VAL_FRACTION = 0.10
TRAIN_SOURCE_DIRS = ("train_set_1", "train_set_2")
TEST_SOURCE_DIRS = ("test_set",)
ALL_SOURCE_DIRS = TRAIN_SOURCE_DIRS + TEST_SOURCE_DIRS
SPLIT_MANIFEST_VERSION = 1


class StaleSplitManifestError(RuntimeError):
    """Raised when manifest records no longer describe their metadata files."""


@dataclass(frozen=True)
class ImageRecord:
    """One physical image and its annotation summary."""

    image_id: str
    sample_id: str
    source_dir: str
    source_split: str
    image_path: str
    metadata_path: str
    classes: tuple[str, ...]
    class_counts: dict[str, int]
    num_instances: int
    metadata_fingerprint: str | None = None

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["classes"] = list(self.classes)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ImageRecord":
        return cls(
            image_id=str(value.get("image_id") or f"{value['source_dir']}/{value['sample_id']}"),
            sample_id=str(value["sample_id"]),
            source_dir=str(value["source_dir"]),
            source_split=str(value.get("source_split") or "train"),
            image_path=str(value["image_path"]),
            metadata_path=str(value["metadata_path"]),
            classes=tuple(str(item) for item in value.get("classes", [])),
            class_counts={str(key): int(count) for key, count in value.get("class_counts", {}).items()},
            num_instances=int(value.get("num_instances", 0)),
            metadata_fingerprint=(
                str(value["metadata_fingerprint"])
                if value.get("metadata_fingerprint") is not None
                else None
            ),
        )


def _relative_posix(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _metadata_summary(
    metadata: Mapping[str, Any],
) -> tuple[tuple[str, ...], dict[str, int], int]:
    class_counts = Counter(
        instance.get("class")
        for instance in metadata.get("instances", [])
        if isinstance(instance, Mapping) and instance.get("class") in CLASS_TO_IDX
    )
    return (
        tuple(sorted(class_counts)),
        dict(sorted(class_counts.items())),
        int(metadata.get("num_instances", sum(class_counts.values()))),
    )


def index_images(
    data_root: str | Path,
    *,
    source_dirs: Sequence[str] = ALL_SOURCE_DIRS,
    strict: bool = False,
) -> list[ImageRecord]:
    """Index images once, without query/class expansion."""

    root = Path(data_root)
    records: list[ImageRecord] = []
    seen_ids: set[str] = set()
    for source_dir in source_dirs:
        directory = root / source_dir
        if not directory.exists():
            if strict:
                raise FileNotFoundError(f"Source directory does not exist: {directory}")
            continue
        source_split = "test" if source_dir in TEST_SOURCE_DIRS else "train"
        for image_path in sorted(directory.glob("*.png")):
            metadata_path = image_path.with_name(f"{image_path.stem}_meta.json")
            if not metadata_path.is_file():
                if strict:
                    raise FileNotFoundError(f"Missing metadata for {image_path}: {metadata_path}")
                continue
            metadata = load_metadata(metadata_path, allow_legacy=True, strict=strict)
            classes, class_counts, num_instances = _metadata_summary(metadata)
            image_id = f"{source_dir}/{image_path.stem}"
            if image_id in seen_ids:
                raise ValueError(f"Duplicate image id: {image_id}")
            seen_ids.add(image_id)
            records.append(
                ImageRecord(
                    image_id=image_id,
                    sample_id=image_path.stem,
                    source_dir=source_dir,
                    source_split=source_split,
                    image_path=_relative_posix(image_path, root),
                    metadata_path=_relative_posix(metadata_path, root),
                    classes=classes,
                    class_counts=class_counts,
                    num_instances=num_instances,
                    metadata_fingerprint=metadata.get("fingerprint"),
                )
            )
    return sorted(records, key=lambda record: record.image_id)


def _coerce_records(records: Iterable[ImageRecord | Mapping[str, Any]]) -> list[ImageRecord]:
    return [record if isinstance(record, ImageRecord) else ImageRecord.from_dict(record) for record in records]


def validate_image_records(
    data_root: str | Path,
    records: Iterable[ImageRecord | Mapping[str, Any]],
    *,
    strict: bool = False,
    manifest_path: str | Path | None = None,
) -> list[ImageRecord]:
    """Reject records whose fingerprint or annotation summary is stale.

    Legacy metadata remains readable through the normal in-memory adapter. The
    manifest is never modified; callers must rebuild it after metadata changes.
    """

    root = Path(data_root)
    normalized = _coerce_records(records)
    stale_count = 0
    stale_details: list[str] = []
    for record in normalized:
        metadata_path = Path(record.metadata_path)
        if not metadata_path.is_absolute():
            metadata_path = root / metadata_path
        metadata = load_metadata(metadata_path, allow_legacy=True, strict=strict)
        classes, class_counts, num_instances = _metadata_summary(metadata)
        mismatches: list[str] = []
        current_fingerprint = metadata.get("fingerprint")
        if record.metadata_fingerprint != current_fingerprint:
            mismatches.append(
                "metadata_fingerprint "
                f"manifest={record.metadata_fingerprint!r} current={current_fingerprint!r}"
            )
        if record.classes != classes:
            mismatches.append(f"classes manifest={record.classes!r} current={classes!r}")
        if record.class_counts != class_counts:
            mismatches.append(
                f"class_counts manifest={record.class_counts!r} current={class_counts!r}"
            )
        if record.num_instances != num_instances:
            mismatches.append(
                f"num_instances manifest={record.num_instances!r} current={num_instances!r}"
            )
        if mismatches:
            stale_count += 1
            if len(stale_details) < 8:
                stale_details.append(f"{record.image_id}: " + "; ".join(mismatches))

    if stale_count:
        origin = (
            f"split manifest {Path(manifest_path)}"
            if manifest_path is not None
            else "provided image records"
        )
        suffix = "" if stale_count <= len(stale_details) else f"; {stale_count - len(stale_details)} more"
        raise StaleSplitManifestError(
            f"Stale metadata in {origin}: {stale_count} record(s) no longer match current "
            "metadata. Regenerate the split manifest after rebuilding metadata; the existing "
            "manifest was not modified. "
            + " | ".join(stale_details)
            + suffix
        )
    return normalized


def split_image_index(
    records: Iterable[ImageRecord | Mapping[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
    val_fraction: float = DEFAULT_VAL_FRACTION,
) -> dict[str, list[ImageRecord]]:
    """Split only train-source images; preserve the complete test source set."""

    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not math.isfinite(val_fraction) or not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    normalized = _coerce_records(records)
    train_pool = sorted(
        (record for record in normalized if record.source_split == "train"),
        key=lambda record: record.image_id,
    )
    test = sorted(
        (record for record in normalized if record.source_split == "test"),
        key=lambda record: record.image_id,
    )
    unknown = [
        record.image_id
        for record in normalized
        if record.source_split not in {"train", "test"}
    ]
    if unknown:
        raise ValueError(f"Unknown source_split for records: {unknown[:5]}")

    if val_fraction == 0 or not train_pool:
        val_count = 0
    else:
        val_count = int(len(train_pool) * val_fraction + 0.5)
        if len(train_pool) >= 2:
            val_count = max(1, min(len(train_pool) - 1, val_count))
        else:
            val_count = 0

    class_frequency: Counter[str] = Counter()
    for record in train_pool:
        class_frequency.update(record.classes)
    class_targets = {
        class_name: (
            0
            if frequency < 2 or val_count == 0
            else min(
                frequency - 1,
                max(1, int(frequency * val_fraction + 0.5)),
            )
        )
        for class_name, frequency in class_frequency.items()
    }
    selected: list[ImageRecord] = []
    selected_ids: set[str] = set()
    selected_classes: Counter[str] = Counter()

    def stable_rank(value: str) -> str:
        return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()

    def preserves_training_coverage(record: ImageRecord) -> bool:
        # Keep at least one train image for every class with two or more images.
        return all(
            class_frequency[class_name] < 2
            or selected_classes[class_name] + 1 < class_frequency[class_name]
            for class_name in record.classes
        )

    while len(selected) < val_count:
        candidates = [
            record
            for record in train_pool
            if record.image_id not in selected_ids and preserves_training_coverage(record)
        ]
        candidate_classes = {
            class_name for record in candidates for class_name in record.classes
        }
        unmet_classes = [
            class_name
            for class_name, target in class_targets.items()
            if selected_classes[class_name] < target and class_name in candidate_classes
        ]
        if unmet_classes:
            # Iterative multilabel stratification: satisfy the rarest remaining
            # class first, then prefer images covering other unmet rare classes.
            focus_class = min(
                unmet_classes,
                key=lambda class_name: (
                    class_frequency[class_name],
                    -(class_targets[class_name] - selected_classes[class_name]),
                    stable_rank(f"class:{class_name}"),
                ),
            )
            focus_candidates = [
                record for record in candidates if focus_class in record.classes
            ]

            def candidate_score(record: ImageRecord) -> tuple[float, int, str]:
                unmet = [
                    class_name
                    for class_name in record.classes
                    if selected_classes[class_name] < class_targets.get(class_name, 0)
                ]
                rarity_gain = sum(1.0 / class_frequency[name] for name in unmet)
                return (-rarity_gain, -len(unmet), stable_rank(record.image_id))

            chosen = min(focus_candidates, key=candidate_score)
        elif candidates:
            # Targets are met; fill remaining slots without sacrificing a class
            # from train, preferring minimal per-class over-allocation.
            chosen = min(
                candidates,
                key=lambda record: (
                    sum(
                        1.0 / class_frequency[name]
                        for name in record.classes
                        if selected_classes[name] >= class_targets.get(name, 0)
                    ),
                    stable_rank(record.image_id),
                ),
            )
        else:
            # Degenerate data (for example every image has a unique class) can
            # make perfect train coverage impossible. Preserve exact split size
            # with a deterministic fallback and make the limitation reportable.
            remaining = [record for record in train_pool if record.image_id not in selected_ids]
            if not remaining:
                break
            chosen = min(remaining, key=lambda record: stable_rank(record.image_id))

        selected.append(chosen)
        selected_ids.add(chosen.image_id)
        selected_classes.update(chosen.classes)

    val_ids = selected_ids
    split = {
        "train": sorted(
            (record for record in train_pool if record.image_id not in val_ids),
            key=lambda record: record.image_id,
        ),
        "val": sorted(
            (record for record in train_pool if record.image_id in val_ids),
            key=lambda record: record.image_id,
        ),
        "test": test,
    }
    assert_no_image_leakage(split)
    return split


def assert_no_image_leakage(
    splits: Mapping[str, Sequence[ImageRecord | Mapping[str, Any]]],
) -> None:
    """Raise if a physical image appears in more than one output split."""

    owners: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        for record_value in splits.get(split_name, []):
            record = (
                record_value
                if isinstance(record_value, ImageRecord)
                else ImageRecord.from_dict(record_value)
            )
            previous = owners.setdefault(record.image_id, split_name)
            if previous != split_name:
                raise ValueError(
                    f"Image leakage: {record.image_id!r} appears in {previous!r} and {split_name!r}"
                )


def class_distribution_report(
    splits: Mapping[str, Sequence[ImageRecord | Mapping[str, Any]]],
) -> dict[str, Any]:
    """Report image, query, instance, and per-class counts for each split."""

    report: dict[str, Any] = {}
    for split_name in ("train", "val", "test"):
        records = _coerce_records(splits.get(split_name, []))
        image_counts: Counter[str] = Counter()
        instance_counts: Counter[str] = Counter()
        for record in records:
            image_counts.update(record.classes)
            instance_counts.update(record.class_counts)
        classes = {
            class_name: {
                "images": image_counts.get(class_name, 0),
                "instances": instance_counts.get(class_name, 0),
            }
            for class_name in sorted(set(image_counts) | set(instance_counts))
        }
        report[split_name] = {
            "images": len(records),
            "queries": sum(len(record.classes) for record in records),
            "instances": sum(record.num_instances for record in records),
            "classes": classes,
        }
    return report


def image_index_fingerprint(
    records: Iterable[ImageRecord | Mapping[str, Any]],
) -> str:
    """Fingerprint image ids, metadata provenance, and annotation summaries."""

    normalized = sorted(_coerce_records(records), key=lambda record: record.image_id)
    payload = [
        {
            "image_id": record.image_id,
            "metadata_fingerprint": record.metadata_fingerprint,
            "classes": list(record.classes),
            "class_counts": dict(sorted(record.class_counts.items())),
            "num_instances": record.num_instances,
        }
        for record in normalized
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def split_manifest_fingerprint(manifest: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 fingerprint excluding the fingerprint field itself."""

    payload = {key: value for key, value in manifest.items() if key != "fingerprint"}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_split_manifest(
    data_root: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    strict: bool = False,
) -> dict[str, Any]:
    """Build a deterministic, image-level manifest in memory."""

    records = index_images(data_root, strict=strict)
    split_records = split_image_index(records, seed=seed, val_fraction=val_fraction)
    serialized_splits = {
        split_name: [record.to_dict() for record in split_records[split_name]]
        for split_name in ("train", "val", "test")
    }
    manifest: dict[str, Any] = {
        "schema_version": SPLIT_MANIFEST_VERSION,
        "seed": seed,
        "val_fraction": val_fraction,
        "split_strategy": "multilabel_rare_class_greedy_v1",
        "source_dirs": {
            "train": list(TRAIN_SOURCE_DIRS),
            "test": list(TEST_SOURCE_DIRS),
        },
        "image_index": [record.to_dict() for record in records],
        "splits": serialized_splits,
        "class_distribution": class_distribution_report(split_records),
    }
    manifest["fingerprint"] = split_manifest_fingerprint(manifest)
    return manifest


def validate_split_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate manifest structure, source provenance, and split isolation."""

    if manifest.get("schema_version") != SPLIT_MANIFEST_VERSION:
        raise ValueError(f"Unsupported split manifest version: {manifest.get('schema_version')!r}")
    fingerprint = manifest.get("fingerprint")
    if fingerprint is not None and fingerprint != split_manifest_fingerprint(manifest):
        raise ValueError("Split manifest fingerprint does not match its contents")
    image_index = manifest.get("image_index")
    if image_index is not None:
        if not isinstance(image_index, list):
            raise ValueError("manifest.image_index must be an array")
        indexed = _coerce_records(image_index)
        if len({record.image_id for record in indexed}) != len(indexed):
            raise ValueError("manifest.image_index contains duplicate image ids")
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("manifest.splits must be an object")
    for name in ("train", "val", "test"):
        if not isinstance(splits.get(name), list):
            raise ValueError(f"manifest.splits.{name} must be an array")
    records = {
        name: _coerce_records(splits[name])
        for name in ("train", "val", "test")
    }
    assert_no_image_leakage(records)
    if image_index is not None:
        index_ids = {record.image_id for record in _coerce_records(image_index)}
        split_ids = {
            record.image_id
            for name in ("train", "val", "test")
            for record in records[name]
        }
        if split_ids != index_ids:
            raise ValueError("manifest splits must partition the complete image_index")
    for record in records["train"] + records["val"]:
        if record.source_split != "train":
            raise ValueError(f"{record.image_id} leaks non-train source into train/val")
    for record in records["test"]:
        if record.source_split != "test":
            raise ValueError(f"{record.image_id} is not from the untouched test source")


def write_split_manifest(manifest: Mapping[str, Any], path: str | Path) -> None:
    """Validate and atomically write a manifest as deterministic JSON."""

    validate_split_manifest(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def load_split_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_split_manifest(manifest)
    return manifest


def resolve_split_manifest_path(
    data_root: str | Path,
    manifest_path: str | Path | None = None,
) -> Path | None:
    """Resolve an explicit or conventional split manifest path."""

    if manifest_path is not None:
        return Path(manifest_path)
    root = Path(data_root)
    return next(
        (
            candidate
            for candidate in (root / "splits.json", root / "split_manifest.json")
            if candidate.is_file()
        ),
        None,
    )


def records_for_split(
    data_root: str | Path,
    split: str,
    *,
    manifest_path: str | Path | None = None,
    strict: bool = False,
) -> list[ImageRecord]:
    """Load one manifest split, or use legacy source directories if absent."""

    split = "val" if split == "validation" else split
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be 'train', 'val', or 'test'")
    root = Path(data_root)
    selected_manifest = resolve_split_manifest_path(root, manifest_path)
    if selected_manifest is not None:
        manifest = load_split_manifest(selected_manifest)
        records = _coerce_records(manifest["splits"][split])
        return validate_image_records(
            root,
            records,
            strict=strict,
            manifest_path=selected_manifest,
        )
    if split == "val":
        raise RuntimeError(
            "An image-level split manifest is required for split='val'. "
            "Run scripts/data/build_splits.py first."
        )
    source_dirs = TRAIN_SOURCE_DIRS if split == "train" else TEST_SOURCE_DIRS
    return index_images(root, source_dirs=source_dirs, strict=strict)


# Compatibility/explicit names.
build_image_index = index_images
build_class_distribution_report = class_distribution_report
