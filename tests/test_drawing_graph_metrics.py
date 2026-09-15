import pytest

from src.drawing_graph import (
    CanonicalDrawingGraph,
    DrawingEdge,
    DrawingNode,
    evaluate_structure_prediction,
    summarize_structure_metrics,
)


def _region(node_id, points, *, gold=False):
    return DrawingNode(
        id=node_id,
        kind="semantic" if gold else "geometry",
        node_type="region",
        semantic_name="room" if gold else "enclosed_wall_face_candidate",
        geometry={"type": "polygon", "points": points},
        provenance={"source": "human_room_gold"} if gold else {"rule": "test"},
    )


def test_structure_metrics_match_candidate_regions_to_human_gold():
    left = [[0, 0], [40, 0], [40, 40], [0, 40]]
    right = [[40, 0], [80, 0], [80, 40], [40, 40]]
    prediction = CanonicalDrawingGraph(
        drawing_id="sample",
        nodes=[_region("pred:left", left), _region("pred:right", right)],
        edges=[
            DrawingEdge(
                id="pred-edge",
                relation="candidate_connected_by",
                source="pred:left",
                target="pred:right",
                attributes={"door_id": "instance:door-1", "status": "candidate"},
                provenance={"rule": "test"},
            )
        ],
        metadata={"source_drawing_id": "source-1"},
    )
    gold = CanonicalDrawingGraph(
        drawing_id="sample",
        nodes=[
            _region("gold_region:left", left, gold=True),
            _region("gold_region:right", right, gold=True),
        ],
        edges=[
            DrawingEdge(
                id="gold-edge",
                relation="connected_by",
                source="gold_region:left",
                target="gold_region:right",
                attributes={"door_instance_id": "door-1"},
                provenance={"source": "human_room_gold"},
            )
        ],
        metadata={"source_drawing_id": "source-1"},
    )

    result = evaluate_structure_prediction(prediction, gold)
    summary = summarize_structure_metrics([result])

    assert result["region_f1"] == pytest.approx(1.0)
    assert result["connected_by_f1"] == pytest.approx(1.0)
    assert result["prediction_graph_valid"] is True
    assert summary["macro_region_f1"] == pytest.approx(1.0)


def test_structure_metrics_count_unmatched_region_as_false_positive():
    gold_points = [[0, 0], [40, 0], [40, 40], [0, 40]]
    far_points = [[100, 100], [140, 100], [140, 140], [100, 140]]
    prediction = CanonicalDrawingGraph(
        drawing_id="sample",
        nodes=[_region("pred:far", far_points)],
        edges=[],
        metadata={"source_drawing_id": "source-1"},
    )
    gold = CanonicalDrawingGraph(
        drawing_id="sample",
        nodes=[_region("gold_region:room", gold_points, gold=True)],
        edges=[],
        metadata={"source_drawing_id": "source-1"},
    )

    result = evaluate_structure_prediction(prediction, gold)

    assert result["matched_region_count"] == 0
    assert result["region_f1"] == 0.0
