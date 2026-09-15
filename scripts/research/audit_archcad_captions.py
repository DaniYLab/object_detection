"""Audit released ArchCAD caption QA without treating captions as gold CDG QA."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.drawing_graph.caption_audit import (  # noqa: E402
    RELATION_TERMS,
    caption_task_labels,
    normalize_caption_template,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--caption-zip",
        type=Path,
        default=Path("data/ArchCAD_raw/data/caption.zip"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/drawing_graph/archcad_caption_audit.json"),
    )
    parser.add_argument("--top-templates", type=int, default=30)
    return parser.parse_args()


def audit_caption_zip(path: Path, top_templates: int = 30) -> dict[str, Any]:
    file_count = 0
    qa_count = 0
    invalid_json_files: list[str] = []
    invalid_records: list[dict[str, Any]] = []
    qa_per_file: Counter[int] = Counter()
    type_pairs: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    relation_term_counts: Counter[str] = Counter()
    question_templates: Counter[str] = Counter()
    answer_templates: Counter[str] = Counter()

    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if member.endswith("/"):
                continue
            file_count += 1
            try:
                rows = json.loads(archive.read(member))
            except (json.JSONDecodeError, UnicodeDecodeError):
                invalid_json_files.append(member)
                continue
            if not isinstance(rows, list):
                invalid_records.append({"file": member, "error": "root must be a list"})
                continue
            qa_per_file[len(rows)] += 1
            for index, row in enumerate(rows):
                qa_count += 1
                if not isinstance(row, dict):
                    invalid_records.append(
                        {"file": member, "index": index, "error": "QA must be a mapping"}
                    )
                    continue
                question = row.get("question")
                answer = row.get("answer")
                type_pairs[f"{type(question).__name__}/{type(answer).__name__}"] += 1
                labels = caption_task_labels(question)
                label_counts.update(labels)
                if not isinstance(question, str) or not isinstance(answer, str):
                    invalid_records.append(
                        {
                            "file": member,
                            "index": index,
                            "question": question,
                            "answer": answer,
                            "error": "question and answer must both be strings",
                        }
                    )
                    continue
                question_templates[normalize_caption_template(question)] += 1
                answer_templates[normalize_caption_template(answer)] += 1
                for term in RELATION_TERMS:
                    if term in question:
                        relation_term_counts[term] += 1

    relation_count = label_counts["relation_or_room_reasoning"]
    return {
        "schema_version": "archcad-caption-audit-v0",
        "caption_zip": str(path),
        "file_count": file_count,
        "qa_count": qa_count,
        "qa_per_file_distribution": dict(sorted(qa_per_file.items())),
        "question_answer_type_counts": dict(sorted(type_pairs.items())),
        "task_label_counts_multilabel": dict(sorted(label_counts.items())),
        "relation_term_counts": dict(sorted(relation_term_counts.items())),
        "relation_or_room_reasoning_fraction": relation_count / qa_count if qa_count else 0.0,
        "unique_normalized_question_templates": len(question_templates),
        "unique_normalized_answer_templates": len(answer_templates),
        "top_question_templates": [
            {"count": count, "template": template}
            for template, count in question_templates.most_common(top_templates)
        ],
        "invalid_json_files": invalid_json_files,
        "invalid_record_count": len(invalid_records),
        "invalid_records": invalid_records,
        "research_interpretation": {
            "usable_as": "auxiliary symbol count/existence/localization QA after executable verification",
            "not_yet_usable_as": "gold room/connectivity/CDG benchmark",
            "reason": (
                "captions lack source-drawing provenance and are mostly templated; "
                "relation/room questions require independent graph validation"
            ),
        },
    }


def main() -> None:
    args = parse_args()
    report = audit_caption_zip(args.caption_zip, args.top_templates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
