"""Task-shape audit helpers for the released ArchCAD caption QA files."""

from __future__ import annotations

import re
from typing import Any


RELATION_TERMS = (
    "相邻",
    "连接",
    "连通",
    "路径",
    "同一个房间",
    "房间内",
    "在哪个房间",
    "几间",
)
LOCALIZATION_TERMS = ("坐标", "位置", "左上角", "右上角", "左下角", "右下角")
COUNT_TERMS = ("多少", "几个", "数量", "几根", "几扇", "几处", "几台", "几套", "几组", "几道")
EXISTENCE_TERMS = ("是否", "存在", "有没有", "有无")


def caption_task_labels(question: Any) -> tuple[str, ...]:
    """Return conservative, overlapping labels for one Chinese QA question."""

    if not isinstance(question, str):
        return ("invalid_question_type",)
    labels: list[str] = []
    term_groups = (
        ("relation_or_room_reasoning", RELATION_TERMS),
        ("localization", LOCALIZATION_TERMS),
        ("count", COUNT_TERMS),
        ("existence", EXISTENCE_TERMS),
    )
    for label, terms in term_groups:
        if any(term in question for term in terms):
            labels.append(label)
    return tuple(labels or ["other"])


def normalize_caption_template(value: Any) -> str:
    """Collapse coordinates and numbers to estimate caption template diversity."""

    text = str(value)
    text = re.sub(r"\([^)]*\)", "(#)", text)
    return re.sub(r"\d+(?:\.\d+)?", "#", text)
