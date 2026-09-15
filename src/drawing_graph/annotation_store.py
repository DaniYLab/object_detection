"""Concurrency-safe persistence for the room-gold annotation workbench."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.drawing_graph.gold_annotations import (
    summarize_room_gold_records,
    validate_room_gold_record,
)


class AnnotationConflictError(ValueError):
    """Raised when a client attempts to overwrite a newer record version."""


def _record_version(record: Mapping[str, Any]) -> str:
    payload = json.dumps(
        record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class RoomGoldStore:
    """Load, validate, and atomically update a room-gold JSONL pack."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self._lock = threading.Lock()
        self._backup_path: Path | None = None
        self._records = self._load()
        self._image_to_index = {
            str(record["image_id"]): index for index, record in enumerate(self._records)
        }
        if len(self._image_to_index) != len(self._records):
            raise ValueError("room gold pack contains duplicate image_id values")

    def _load(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number}: record must be a mapping")
            errors = validate_room_gold_record(value)
            if errors:
                raise ValueError(f"line {line_number}: {errors}")
            records.append(value)
        return records

    def __len__(self) -> int:
        return len(self._records)

    @property
    def backup_path(self) -> Path | None:
        return self._backup_path

    def contains_image(self, image_id: str) -> bool:
        return image_id in self._image_to_index

    def get(self, index: int) -> tuple[dict[str, Any], str]:
        with self._lock:
            if index < 0 or index >= len(self._records):
                raise IndexError(index)
            record = copy.deepcopy(self._records[index])
            return record, _record_version(record)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return summarize_room_gold_records(self._records)

    def update_editable_fields(
        self,
        index: int,
        payload: Mapping[str, Any],
        *,
        expected_version: str,
    ) -> tuple[dict[str, Any], str]:
        """Update reviewer/status/annotation while preserving selection and hints."""

        with self._lock:
            if index < 0 or index >= len(self._records):
                raise IndexError(index)
            current = self._records[index]
            current_version = _record_version(current)
            if expected_version != current_version:
                raise AnnotationConflictError(
                    "record changed since it was loaded; reload before saving"
                )
            updated = copy.deepcopy(current)
            for field in ("review_status", "reviewer", "annotation"):
                if field not in payload:
                    raise ValueError(f"missing editable field: {field}")
                updated[field] = copy.deepcopy(payload[field])
            errors = validate_room_gold_record(updated)
            if errors:
                raise ValueError(f"invalid room gold record: {errors}")

            self._ensure_backup()
            replacement = list(self._records)
            replacement[index] = updated
            self._write_atomic(replacement)
            self._records = replacement
            result = copy.deepcopy(updated)
            return result, _record_version(result)

    def _ensure_backup(self) -> None:
        if self._backup_path is not None:
            return
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = self.path.with_name(f"{self.path.name}.backup-{timestamp}")
        suffix = 1
        while backup.exists():
            backup = self.path.with_name(
                f"{self.path.name}.backup-{timestamp}-{suffix:02d}"
            )
            suffix += 1
        shutil.copy2(self.path, backup)
        self._backup_path = backup

    def _write_atomic(self, records: list[dict[str, Any]]) -> None:
        temporary = self.path.with_name(f".{self.path.name}.tmp-{os.getpid()}")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)
