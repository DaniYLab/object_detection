import numpy as np
import pytest

from src.data.archcad_constants import ARCHCAD_IGNORE_LABEL
from src.data.archcad_panoptic import (
    oracle_panoptic_prediction,
    parse_archcad_panoptic_annotation,
    primitive_length,
)
from src.evaluation.archcad_panoptic import evaluate_archcad_panoptic


def _annotation():
    return {
        "entities": [
            {
                "type": "LINE",
                "start": [0.0, 0.0],
                "end": [10.0, 0.0],
                "semantic": 20,
            },
            {
                "type": "LINE",
                "start": [10.0, 0.0],
                "end": [20.0, 0.0],
                "semantic": 20,
            },
            {
                "type": "LINE",
                "start": [2.0, 0.0],
                "end": [4.0, 0.0],
                "semantic": 1,
                "instance": "door-a",
            },
            {
                "type": "ARC",
                "center": [2.0, 0.0],
                "radius": 2.0,
                "start": [2.0, 0.0],
                "end": [4.0, 2.0],
                "start_angle": 0.0,
                "end_angle": 90.0,
                "semantic": 1,
                "instance": "door-a",
            },
            {
                "type": "LINE",
                "start": [12.0, 0.0],
                "end": [14.0, 0.0],
                "semantic": 1,
                "instance": "door-b",
            },
            {
                "type": "LINE",
                "start": [0.0, 1.0],
                "end": [1.0, 1.0],
                "semantic": 100,
            },
        ]
    }


def test_panoptic_parser_preserves_thing_stuff_and_ignore_contract():
    sample = parse_archcad_panoptic_annotation(_annotation(), image_id="sample")

    assert sample.semantic_labels.tolist() == [20, 20, 1, 1, 1, ARCHCAD_IGNORE_LABEL]
    assert sample.instance_ids.tolist() == [-1, -1, 0, 0, 1, -1]
    assert [(segment.semantic_id, segment.instance_key) for segment in sample.segments] == [
        (20, "stuff"),
        (1, "door-a"),
        (1, "door-b"),
    ]
    assert [int(segment.primitive_mask.sum()) for segment in sample.segments] == [2, 2, 1]
    assert sample.issues == ()


def test_primitive_lengths_cover_line_arc_and_circle():
    assert primitive_length(
        {"type": "LINE", "start": [0, 0], "end": [3, 4]}
    ) == pytest.approx(5.0)
    assert primitive_length(
        {"type": "ARC", "radius": 2, "start_angle": 0, "end_angle": 90}
    ) == pytest.approx(np.pi)
    assert primitive_length({"type": "CIRCLE", "radius": 2}) == pytest.approx(
        4 * np.pi
    )


def test_archcad_panoptic_oracle_scores_one_on_all_tracks():
    sample = parse_archcad_panoptic_annotation(_annotation(), image_id="sample")

    report = evaluate_archcad_panoptic(
        [sample], [oracle_panoptic_prediction(sample)]
    )

    assert report["semantic"]["f1_macro"] == pytest.approx(1.0)
    assert report["semantic"]["weighted_f1_macro"] == pytest.approx(1.0)
    assert report["instance"]["ap50"] == pytest.approx(1.0)
    assert report["instance"]["ap75"] == pytest.approx(1.0)
    assert report["instance"]["map_50_95"] == pytest.approx(1.0)
    assert report["panoptic"]["all"]["pq"] == pytest.approx(1.0)
    assert report["panoptic"]["thing"]["pq"] == pytest.approx(1.0)
    assert report["panoptic"]["stuff"]["pq"] == pytest.approx(1.0)


def test_archcad_panoptic_semantic_error_reduces_semantic_score():
    sample = parse_archcad_panoptic_annotation(_annotation(), image_id="sample")
    prediction = oracle_panoptic_prediction(sample)
    prediction["semantic_labels"] = prediction["semantic_labels"].copy()
    prediction["semantic_labels"][0] = 1

    report = evaluate_archcad_panoptic([sample], [prediction])

    assert report["semantic"]["f1_macro"] < 1.0
    assert report["instance"]["ap50"] == pytest.approx(1.0)


def test_unassigned_thing_primitives_follow_released_target_grouping():
    sample = parse_archcad_panoptic_annotation(
        {
            "entities": [
                {
                    "type": "LINE",
                    "start": [0, 0],
                    "end": [2, 0],
                    "semantic": 1,
                },
                {
                    "type": "LINE",
                    "start": [3, 0],
                    "end": [5, 0],
                    "semantic": 1,
                },
            ]
        },
        image_id="unassigned",
    )

    assert len(sample.issues) == 2
    assert len(sample.segments) == 1
    assert sample.segments[0].instance_key == "unassigned:-1"
    assert sample.segments[0].primitive_mask.tolist() == [True, True]
