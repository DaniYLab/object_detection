"""Content-level deduplication for split manifests.

FloorPlanCAD re-uses the same PNG under different image ids (``NNNN-MMMM`` names
group a drawing family, but different ``NNNN`` can be byte-identical files). The
schema-v2 manifest splits by image id, so byte-identical content can land in two
roles and inflate validation metrics.

This module provides the grouping primitive used by
``scripts/data/dedup_splits.py``. The fix is to make the *split unit* the content
group rather than the image id: assign every member of a duplicate group to one
role, then expand back so the manifest still lists every physical image.

Grouping is by PNG SHA-256, never by filename prefix — measured on this dataset,
same-prefix images are usually *not* the same drawing (mean ink-mask IoU 0.048),
while different-prefix images can be byte-identical.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .splits import ImageRecord

_CHUNK = 1 << 20


def sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 of a file, read in chunks."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(_CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class DuplicateGroup:
    """A set of image ids whose PNG bytes are identical."""

    content_hash: str
    image_ids: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.image_ids)

    @property
    def canonical_id(self) -> str:
        """Deterministic representative: the lexicographically smallest id."""

        return min(self.image_ids)


def resolve_image_path(root: str | Path, record: ImageRecord | Mapping[str, Any]) -> Path:
    """Resolve a record's PNG path against the dataset root."""

    raw = record.image_path if isinstance(record, ImageRecord) else record["image_path"]
    path = Path(raw)
    return path if path.is_absolute() else Path(root) / path


def group_by_content(
    records: Iterable[ImageRecord],
    *,
    root: str | Path,
    on_progress: Any | None = None,
) -> dict[str, DuplicateGroup]:
    """Group records by PNG SHA-256, returning only groups of size >= 2."""

    by_hash: dict[str, list[str]] = {}
    for index, record in enumerate(records):
        path = resolve_image_path(root, record)
        if not path.is_file():
            raise FileNotFoundError(f"Missing image for record {record.image_id!r}: {path}")
        by_hash.setdefault(sha256_file(path), []).append(record.image_id)
        if on_progress is not None and (index + 1) % 1000 == 0:
            on_progress(index + 1)

    return {
        content_hash: DuplicateGroup(content_hash, tuple(sorted(ids)))
        for content_hash, ids in by_hash.items()
        if len(ids) > 1
    }


def merge_group_record(members: Sequence[ImageRecord]) -> ImageRecord:
    """Collapse a group into one canonical record describing the whole group.

    Class information is unioned rather than taken from a single member: the
    members are byte-identical images but their ``_meta.json`` can disagree
    (verified on this dataset), so the group's class profile is what the splitter
    must satisfy for rare-class coverage.
    """

    if not members:
        raise ValueError("cannot merge an empty group")
    ordered = sorted(members, key=lambda record: record.image_id)
    canonical = ordered[0]

    classes: set[str] = set()
    counts: Counter[str] = Counter()
    instances = 0
    for record in ordered:
        classes.update(record.classes)
        counts.update(record.class_counts)
        instances += record.num_instances

    return ImageRecord(
        image_id=canonical.image_id,
        sample_id=canonical.sample_id,
        source_dir=canonical.source_dir,
        source_split=canonical.source_split,
        image_path=canonical.image_path,
        metadata_path=canonical.metadata_path,
        classes=tuple(sorted(classes)),
        class_counts=dict(sorted(counts.items())),
        num_instances=instances,
        metadata_fingerprint=canonical.metadata_fingerprint,
    )


def build_split_units(
    records: Sequence[ImageRecord],
    groups: Mapping[str, DuplicateGroup],
) -> tuple[list[ImageRecord], dict[str, str]]:
    """Return (unit records, member_id -> canonical unit id).

    Every record belongs to exactly one unit. Records in a duplicate group are
    replaced by a single merged record; records with unique content pass through
    unchanged.
    """

    by_id = {record.image_id: record for record in records}
    canonical_to_members = groups_by_canonical(groups)
    member_to_unit: dict[str, str] = {}
    for group in groups.values():
        for member in group.image_ids:
            if member not in by_id:
                raise KeyError(f"Duplicate group member not in records: {member!r}")
            member_to_unit[member] = group.canonical_id

    units: list[ImageRecord] = []
    for record in records:
        unit_id = member_to_unit.get(record.image_id)
        if unit_id is None:
            units.append(record)
        elif unit_id == record.image_id:
            members = [by_id[mid] for mid in canonical_to_members[unit_id]]
            units.append(merge_group_record(members))
        # else: a non-canonical member, absorbed into its canonical unit
    return units, member_to_unit


def groups_by_canonical(
    groups: Mapping[str, DuplicateGroup],
) -> dict[str, tuple[str, ...]]:
    """Map canonical id -> its group members (including itself)."""

    return {group.canonical_id: group.image_ids for group in groups.values()}


def expand_units_to_members(
    unit_splits: Mapping[str, Sequence[ImageRecord]],
    groups: Mapping[str, DuplicateGroup],
    all_records: Mapping[str, ImageRecord],
) -> dict[str, list[ImageRecord]]:
    """Expand each role's canonical units back to every physical member image."""

    canonical = groups_by_canonical(groups)
    expanded: dict[str, list[ImageRecord]] = {}
    for role, units in unit_splits.items():
        members: list[ImageRecord] = []
        for unit in units:
            ids = canonical.get(unit.image_id, (unit.image_id,))
            for image_id in ids:
                members.append(all_records[image_id])
        expanded[role] = sorted(members, key=lambda record: record.image_id)
    return expanded


def count_cross_role_duplicates(
    role_splits: Mapping[str, Sequence[ImageRecord]],
    groups: Mapping[str, DuplicateGroup],
) -> dict[str, Any]:
    """Report how many duplicate groups straddle roles, per role pair."""

    owner: dict[str, str] = {}
    for role, records in role_splits.items():
        for record in records:
            owner[record.image_id] = role

    straddling: list[dict[str, Any]] = []
    pairs: Counter[tuple[str, str]] = Counter()
    for group in groups.values():
        roles = sorted({owner[mid] for mid in group.image_ids if mid in owner})
        if len(roles) > 1:
            straddling.append(
                {
                    "content_hash": group.content_hash,
                    "size": group.size,
                    "roles": roles,
                    "image_ids": list(group.image_ids),
                }
            )
            for i in range(len(roles)):
                for j in range(i + 1, len(roles)):
                    pairs[(roles[i], roles[j])] += 1

    return {
        "straddling_group_count": len(straddling),
        "role_pair_counts": {f"{a}|{b}": count for (a, b), count in sorted(pairs.items())},
        "straddling_groups": straddling,
    }
