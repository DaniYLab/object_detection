from dataclasses import replace

import pytest

from src.drawing_graph import compile_archcad_graph, evaluate_archcad_prediction


def _annotation():
    return {
        "entities": [
            {
                "type": "LINE",
                "start": [0.0, 0.0],
                "end": [20.0, 0.0],
                "semantic": 20,
            },
            {
                "type": "LINE",
                "start": [0.0, 0.0],
                "end": [0.0, 20.0],
                "semantic": 20,
            },
            {
                "type": "LINE",
                "start": [8.0, 0.0],
                "end": [12.0, 0.0],
                "semantic": 1,
                "instance": "door-1",
            },
            {
                "type": "ARC",
                "center": [8.0, 0.0],
                "radius": 4.0,
                "start": [8.0, 0.0],
                "end": [12.0, 4.0],
                "semantic": 1,
                "instance": "door-1",
            },
        ]
    }


def test_archcad_oracle_is_perfect_direct_gt_sanity_ceiling():
    ground_truth = compile_archcad_graph(
        _annotation(), drawing_id="sample", source_drawing_id="drawing-1"
    )
    prediction = compile_archcad_graph(
        _annotation(), drawing_id="sample", source_drawing_id="drawing-1"
    )

    result = evaluate_archcad_prediction(prediction, ground_truth)

    assert result["primitive_geometry_f1"] == pytest.approx(1.0)
    assert result["primitive_semantic_f1"] == pytest.approx(1.0)
    assert result["instance_f1"] == pytest.approx(1.0)
    assert result["junction_f1"] == pytest.approx(1.0)
    assert result["incident_edge_f1"] == pytest.approx(1.0)
    assert result["semantic_count_exact_fraction"] == pytest.approx(1.0)


def test_archcad_metrics_separate_geometry_from_semantic_error():
    ground_truth = compile_archcad_graph(
        _annotation(), drawing_id="sample", source_drawing_id="drawing-1"
    )
    prediction = compile_archcad_graph(
        _annotation(), drawing_id="sample", source_drawing_id="drawing-1"
    )
    prediction.nodes[0] = replace(
        prediction.nodes[0], semantic_id=24, semantic_name="steel_beam"
    )

    result = evaluate_archcad_prediction(prediction, ground_truth)

    assert result["primitive_geometry_f1"] == pytest.approx(1.0)
    assert result["primitive_semantic_f1"] < 1.0
    assert result["semantic_accuracy_on_geometry_matches"] < 1.0


def test_archcad_metrics_report_and_exclude_degenerate_instance_boxes():
    annotation = {
        "entities": [
            {
                "type": "LINE",
                "start": [5.0, 5.0],
                "end": [5.0, 10.0],
                "semantic": 21,
                "instance": "zero-width-column",
            }
        ]
    }
    ground_truth = compile_archcad_graph(
        annotation, drawing_id="sample", source_drawing_id="drawing-1"
    )
    prediction = compile_archcad_graph(
        annotation, drawing_id="sample", source_drawing_id="drawing-1"
    )

    result = evaluate_archcad_prediction(prediction, ground_truth)

    assert result["unscorable_gold_instance_count"] == 1
    assert result["unscorable_predicted_instance_count"] == 1
    assert result["instance_f1"] == pytest.approx(1.0)
