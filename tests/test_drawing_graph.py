import json
from pathlib import Path

import pytest

from src.drawing_graph.caption_audit import caption_task_labels, normalize_caption_template
from src.drawing_graph import (
    CanonicalDrawingGraph,
    AnnotationConflictError,
    RoomGoldStore,
    add_room_gold_annotations,
    add_raster_region_candidates,
    compile_archcad_graph,
    compare_room_gold_records,
    classify_archcad_task,
    execute_query,
    infer_raster_free_space_candidates,
    summarize_room_gold_records,
    summarize_room_gold_agreement,
    validate_graph,
    validate_room_gold_record,
)


@pytest.fixture
def annotation():
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
                "start": [10.2, 0.1],
                "end": [10.0, 10.0],
                "semantic": 20,
            },
            {
                "type": "LINE",
                "start": [4.0, 0.0],
                "end": [6.0, 0.0],
                "semantic": 1,
                "instance": "door-1",
            },
            {
                "type": "ARC",
                "center": [4.0, 0.0],
                "radius": 2.0,
                "start": [4.0, 0.0],
                "end": [6.0, 2.0],
                "start_angle": 0.0,
                "end_angle": 90.0,
                "semantic": 1,
                "instance": "door-1",
            },
        ]
    }


def test_archcad_compiler_preserves_structure_and_instances(annotation):
    graph = compile_archcad_graph(
        annotation,
        drawing_id="sample",
        source_drawing_id="drawing-42",
        endpoint_tolerance=0.5,
    )

    assert execute_query(
        graph, {"op": "count_nodes", "semantic_name": "wall"}
    ).answer == 2
    assert execute_query(
        graph, {"op": "count_nodes", "node_type": "instance"}
    ).answer == 1
    assert execute_query(
        graph, {"op": "count_relations", "relation": "part_of"}
    ).answer == 2
    assert any(node.kind == "junction" for node in graph.nodes)
    assert validate_graph(graph).is_valid


def test_query_returns_evidence(annotation):
    graph = compile_archcad_graph(annotation, drawing_id="sample")
    result = execute_query(
        graph,
        {"op": "count_nodes", "node_type": "instance", "semantic_name": "single_door"},
    )
    assert result.answer == 1
    assert result.evidence_nodes == ("instance:door-1",)


def test_missing_source_drawing_id_is_a_warning(annotation):
    graph = compile_archcad_graph(annotation, drawing_id="sample")
    report = validate_graph(graph)
    assert report.is_valid
    assert "missing_source_drawing_id" in {issue.code for issue in report.warnings}


def test_graph_round_trip(annotation, tmp_path):
    graph = compile_archcad_graph(annotation, drawing_id="sample")
    output = tmp_path / "sample.cdg.json"
    graph.write_json(output)
    loaded = CanonicalDrawingGraph.read_json(output)
    assert loaded.to_dict() == graph.to_dict()
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == "cdg-v0"


def test_unknown_query_is_rejected(annotation):
    graph = compile_archcad_graph(annotation, drawing_id="sample")
    with pytest.raises(ValueError, match="unsupported query"):
        execute_query(graph, {"op": "free_form_answer"})


def test_shortest_path_returns_relation_evidence(annotation):
    graph = compile_archcad_graph(
        annotation,
        drawing_id="sample",
        endpoint_tolerance=0.5,
    )
    result = execute_query(
        graph,
        {
            "op": "shortest_path",
            "source": "primitive:000000",
            "target": "primitive:000001",
            "relations": ["incident_to"],
        },
    )
    assert result.answer[0] == "primitive:000000"
    assert result.answer[-1] == "primitive:000001"
    assert len(result.answer) == 3
    junction_id = result.answer[1]
    assert next(node for node in graph.nodes if node.id == junction_id).kind == "junction"
    assert len(result.evidence_edges) == 2


def test_topology_candidates_are_explicitly_marked():
    annotation = {
        "entities": [
            {"type": "LINE", "start": [0, 0], "end": [20, 0], "semantic": 20},
            {"type": "LINE", "start": [20, 0], "end": [20, 20], "semantic": 20},
            {"type": "LINE", "start": [20, 20], "end": [0, 20], "semantic": 20},
            {"type": "LINE", "start": [0, 20], "end": [0, 0], "semantic": 20},
            {
                "type": "LINE",
                "start": [9, 1],
                "end": [11, 1],
                "semantic": 1,
                "instance": "door-near-top-wall",
            },
        ]
    }
    graph = compile_archcad_graph(
        annotation,
        drawing_id="square",
        source_drawing_id="source-square",
        minimum_region_area=100,
        door_wall_distance=2,
    )

    regions = [node for node in graph.nodes if node.node_type == "region"]
    attachments = [
        edge for edge in graph.edges if edge.relation == "candidate_attached_to"
    ]
    assert len(regions) == 1
    assert regions[0].semantic_name == "enclosed_wall_face_candidate"
    assert regions[0].attributes["status"] == "candidate"
    assert regions[0].geometry["area"] == pytest.approx(400.0)
    assert attachments
    assert all(edge.attributes["status"] == "candidate" for edge in attachments)
    assert validate_graph(graph).is_valid


def test_door_connectivity_requires_two_bounded_regions():
    annotation = {
        "entities": [
            {"type": "LINE", "start": [0, 0], "end": [40, 0], "semantic": 20},
            {"type": "LINE", "start": [0, 20], "end": [40, 20], "semantic": 20},
            {"type": "LINE", "start": [0, 0], "end": [0, 20], "semantic": 20},
            {"type": "LINE", "start": [20, 0], "end": [20, 20], "semantic": 20},
            {"type": "LINE", "start": [40, 0], "end": [40, 20], "semantic": 20},
            {
                "type": "LINE",
                "start": [20, 8],
                "end": [20, 12],
                "semantic": 1,
                "instance": "shared-door",
            },
        ]
    }
    graph = compile_archcad_graph(
        annotation,
        drawing_id="two-rooms",
        source_drawing_id="source-two-rooms",
        minimum_region_area=100,
        door_wall_distance=1,
    )
    regions = [node for node in graph.nodes if node.node_type == "region"]
    connectivity = [
        edge for edge in graph.edges if edge.relation == "candidate_connected_by"
    ]
    assert len(regions) == 2
    assert len(connectivity) == 1
    assert {connectivity[0].source, connectivity[0].target} == {
        region.id for region in regions
    }
    assert connectivity[0].attributes["door_id"] == "instance:shared-door"


def test_archcad_task_selection_is_transparent_and_nonexclusive():
    annotation = {
        "entities": [
            *[{"semantic": 20} for _ in range(4)],
            {"semantic": 1, "instance": "door-1"},
            {"semantic": 23},
        ]
    }
    tags = classify_archcad_task(annotation)
    assert tags["wall_graph_candidate"] is True
    assert tags["door_connectivity_candidate"] is True
    assert tags["architectural_reading_candidate"] is True
    assert tags["floor_plan_reading_candidate"] is False
    assert tags["structure_engineering_candidate"] is True


def test_floor_plan_selection_requires_stronger_instance_evidence():
    annotation = {
        "entities": [
            *[{"semantic": 20} for _ in range(8)],
            {"semantic": 1, "instance": "door-1"},
            {"semantic": 1, "instance": "door-2"},
        ]
    }
    assert classify_archcad_task(annotation)["floor_plan_reading_candidate"] is True


def test_raster_regions_find_bounded_free_space():
    annotation = {
        "entities": [
            {"type": "LINE", "start": [10, 10], "end": [110, 10], "semantic": 20},
            {"type": "LINE", "start": [110, 10], "end": [110, 110], "semantic": 20},
            {"type": "LINE", "start": [110, 110], "end": [10, 110], "semantic": 20},
            {"type": "LINE", "start": [10, 110], "end": [10, 10], "semantic": 20},
        ]
    }
    regions, diagnostics = infer_raster_free_space_candidates(
        annotation,
        image_size=(128, 128),
        minimum_area=100,
        minimum_hydraulic_radius=5,
    )
    assert diagnostics["wall_line_count"] == 4
    assert len(regions) == 1
    assert regions[0].bbox == (12, 12, 109, 109)


def test_door_closure_can_close_a_known_wall_gap():
    annotation = {
        "entities": [
            {"type": "LINE", "start": [10, 10], "end": [50, 10], "semantic": 20},
            {"type": "LINE", "start": [70, 10], "end": [110, 10], "semantic": 20},
            {"type": "LINE", "start": [110, 10], "end": [110, 110], "semantic": 20},
            {"type": "LINE", "start": [110, 110], "end": [10, 110], "semantic": 20},
            {"type": "LINE", "start": [10, 110], "end": [10, 10], "semantic": 20},
            {
                "type": "LINE",
                "start": [50, 10],
                "end": [70, 10],
                "semantic": 1,
                "instance": "door-gap",
            },
        ]
    }
    without_closure, _ = infer_raster_free_space_candidates(
        annotation,
        image_size=(128, 128),
        close_doors=False,
        minimum_area=100,
        minimum_hydraulic_radius=5,
    )
    with_closure, diagnostics = infer_raster_free_space_candidates(
        annotation,
        image_size=(128, 128),
        close_doors=True,
        door_closure_width=3,
        minimum_area=100,
        minimum_hydraulic_radius=5,
    )
    assert without_closure == []
    assert len(with_closure) == 1
    assert diagnostics["door_closure_count"] == 1


def test_raster_region_typing_uses_contained_semantic_evidence():
    annotation = {
        "entities": [
            {"type": "LINE", "start": [10, 10], "end": [110, 10], "semantic": 20},
            {"type": "LINE", "start": [110, 10], "end": [110, 110], "semantic": 20},
            {"type": "LINE", "start": [110, 110], "end": [10, 110], "semantic": 20},
            {"type": "LINE", "start": [10, 110], "end": [10, 10], "semantic": 20},
            {
                "type": "CIRCLE",
                "center": [60, 60],
                "radius": 5,
                "semantic": 6,
                "instance": "stair-1",
            },
        ]
    }
    graph = compile_archcad_graph(
        annotation,
        drawing_id="typed-region",
        source_drawing_id="source-typed-region",
        infer_topology_candidates=False,
        image_size=(128, 128),
    )
    result = add_raster_region_candidates(
        graph,
        annotation,
        image_size=(128, 128),
        minimum_area=100,
        minimum_hydraulic_radius=5,
    )
    regions = [node for node in graph.nodes if node.id.startswith("raster_region:")]
    assert result["category_counts"] == {"vertical_core_candidate": 1}
    assert regions[0].semantic_name == "vertical_core_candidate"
    assert any(
        edge.relation == "candidate_inside" and edge.target == regions[0].id
        for edge in graph.edges
    )
    assert validate_graph(graph).is_valid


def test_room_gold_contract_rejects_unknown_relation_nodes():
    record = {
        "schema_version": "room-gold-v0",
        "image_id": "sample",
        "review_status": "reviewed",
        "reviewer": "reviewer-a",
        "annotation": {
            "view_type": "floor_plan",
            "regions": [
                {
                    "id": "region:001",
                    "type": "room",
                    "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
                }
            ],
            "relations": [
                {
                    "relation": "connected_by",
                    "source": "region:001",
                    "target": "region:missing",
                    "door_instance_id": "door:1",
                }
            ],
        },
    }
    errors = validate_room_gold_record(record)
    assert "relations[0] references unknown region" in errors


def test_summarize_room_gold_records_does_not_count_hints_as_gold():
    records = [
        {
            "review_status": "reviewed",
            "preannotations": {"regions": [{"candidate_id": "hint:1"}]},
            "annotation": {
                "view_type": "floor_plan",
                "regions": [
                    {
                        "id": "region:1",
                        "type": "room",
                        "polygon": [[0, 0], [1, 0], [1, 1]],
                    }
                ],
                "relations": [],
            },
        },
        {
            "review_status": "unreviewed",
            "preannotations": {"regions": [{"candidate_id": "hint:2"}]},
            "annotation": {"view_type": None, "regions": [], "relations": []},
        },
    ]

    summary = summarize_room_gold_records(records)

    assert summary["record_count"] == 2
    assert summary["reviewed_count"] == 1
    assert summary["reviewed_floor_plan_count"] == 1
    assert summary["region_count"] == 1
    assert summary["records_with_preannotations"] == 2
    assert summary["preannotations_are_ground_truth"] is False


def test_caption_task_labels_are_multilabel_and_reject_non_text():
    question = "坐标(10, 20)附近的门是否与房间内的楼梯相邻？"

    assert caption_task_labels(question) == (
        "relation_or_room_reasoning",
        "localization",
        "existence",
    )
    assert caption_task_labels(True) == ("invalid_question_type",)
    assert normalize_caption_template(question) == "坐标(#)附近的门是否与房间内的楼梯相邻？"


def test_reviewed_room_gold_compiles_truth_relations_separate_from_candidates(annotation):
    graph = compile_archcad_graph(
        annotation,
        drawing_id="sample",
        source_drawing_id="drawing-42",
    )
    record = {
        "schema_version": "room-gold-v0",
        "image_id": "sample",
        "review_status": "reviewed",
        "reviewer": "reviewer-a",
        "annotation": {
            "view_type": "floor_plan",
            "regions": [
                {
                    "id": "region:001",
                    "type": "room",
                    "polygon": [[0, 0], [5, 0], [5, 5], [0, 5]],
                },
                {
                    "id": "region:002",
                    "type": "corridor",
                    "polygon": [[5, 0], [10, 0], [10, 5], [5, 5]],
                },
            ],
            "relations": [
                {
                    "relation": "connected_by",
                    "source": "region:001",
                    "target": "region:002",
                    "door_instance_id": "door-1",
                }
            ],
            "notes": "",
        },
    }

    counts = add_room_gold_annotations(graph, record)

    assert counts == {"region_count": 2, "relation_count": 1}
    assert execute_query(
        graph, {"op": "count_relations", "relation": "connected_by"}
    ).answer == 1
    assert execute_query(
        graph, {"op": "count_nodes", "semantic_name": "room"}
    ).answer == 1
    assert all(
        edge.relation != "candidate_connected_by"
        for edge in graph.edges
        if edge.id.startswith("gold_connected_by:")
    )
    assert validate_graph(graph).is_valid


def test_unreviewed_room_gold_cannot_be_compiled(annotation):
    graph = compile_archcad_graph(annotation, drawing_id="sample")
    record = {
        "schema_version": "room-gold-v0",
        "image_id": "sample",
        "review_status": "unreviewed",
        "reviewer": None,
        "annotation": {"view_type": None, "regions": [], "relations": []},
    }

    with pytest.raises(ValueError, match="only reviewed"):
        add_room_gold_annotations(graph, record)


def test_room_gold_store_saves_atomically_and_rejects_stale_updates(tmp_path: Path):
    pack = tmp_path / "pack.jsonl"
    record = {
        "schema_version": "room-gold-v0",
        "image_id": "sample",
        "review_status": "unreviewed",
        "reviewer": None,
        "annotation": {"view_type": None, "regions": [], "relations": []},
    }
    pack.write_text(json.dumps(record) + "\n", encoding="utf-8")
    store = RoomGoldStore(pack)
    loaded, version = store.get(0)

    updated, new_version = store.update_editable_fields(
        0,
        {
            "review_status": "reviewed",
            "reviewer": "reviewer-a",
            "annotation": {
                "view_type": "floor_plan",
                "regions": [],
                "relations": [],
                "notes": "checked",
            },
        },
        expected_version=version,
    )

    assert loaded["review_status"] == "unreviewed"
    assert updated["review_status"] == "reviewed"
    assert new_version != version
    assert store.backup_path is not None and store.backup_path.exists()
    assert RoomGoldStore(pack).get(0)[0]["annotation"]["notes"] == "checked"
    with pytest.raises(AnnotationConflictError):
        store.update_editable_fields(
            0,
            {
                "review_status": "unreviewed",
                "reviewer": None,
                "annotation": {"view_type": None, "regions": [], "relations": []},
            },
            expected_version=version,
        )


@pytest.mark.parametrize(
    "polygon",
    [
        [[0, 0], [10, 10], [0, 10], [10, 0]],
        [[0, 0], [5, 0], [10, 0]],
        [[0, 0], [10, 0], [10, 0], [0, 10]],
    ],
)
def test_room_gold_contract_rejects_non_simple_polygons(polygon):
    record = {
        "schema_version": "room-gold-v0",
        "image_id": "sample",
        "review_status": "reviewed",
        "reviewer": "reviewer-a",
        "annotation": {
            "view_type": "floor_plan",
            "regions": [{"id": "region:001", "type": "room", "polygon": polygon}],
            "relations": [],
        },
    }

    assert "regions[0] has invalid polygon" in validate_room_gold_record(record)


def _reviewed_record(reviewer, region_ids=("region:a", "region:b")):
    return {
        "schema_version": "room-gold-v0",
        "image_id": "sample",
        "review_status": "reviewed",
        "reviewer": reviewer,
        "annotation": {
            "view_type": "floor_plan",
            "regions": [
                {
                    "id": region_ids[0],
                    "type": "room",
                    "polygon": [[10, 10], [50, 10], [50, 50], [10, 50]],
                },
                {
                    "id": region_ids[1],
                    "type": "corridor",
                    "polygon": [[50, 10], [90, 10], [90, 50], [50, 50]],
                },
            ],
            "relations": [
                {
                    "relation": "connected_by",
                    "source": region_ids[0],
                    "target": region_ids[1],
                    "door_instance_id": "door-1",
                }
            ],
        },
    }


def test_validation_report_serializes_invalid_graph():
    graph = CanonicalDrawingGraph(drawing_id="", nodes=[], edges=[])
    payload = validate_graph(graph).to_dict()

    assert payload["is_valid"] is False
    assert {issue["code"] for issue in payload["issues"]} >= {
        "drawing_id",
        "missing_source_drawing_id",
    }


def test_graph_validator_rejects_candidate_relation_without_candidate_status(annotation):
    graph = compile_archcad_graph(
        annotation,
        drawing_id="sample",
        source_drawing_id="source-1",
        infer_topology_candidates=False,
    )
    graph.edges[0] = type(graph.edges[0])(
        id=graph.edges[0].id,
        relation="candidate_inside",
        source=graph.edges[0].source,
        target=graph.edges[0].target,
        provenance={"rule": "bad-test-edge"},
    )

    report = validate_graph(graph)

    assert "candidate_status" in {issue.code for issue in report.errors}


def test_independent_room_gold_agreement_matches_region_ids_by_geometry():
    primary = _reviewed_record("reviewer-a")
    secondary = _reviewed_record("reviewer-b", ("second:1", "second:2"))

    comparison = compare_room_gold_records(primary, secondary)
    summary = summarize_room_gold_agreement([primary], [secondary])

    assert comparison["matched_region_count"] == 2
    assert comparison["minimum_polygon_iou"] == pytest.approx(1.0)
    assert comparison["relation_f1"] == pytest.approx(1.0)
    assert comparison["agreement_gate_passed"] is True
    assert summary["eligible_review_pair_count"] == 1
    assert summary["agreement_gate_pass_fraction"] == pytest.approx(1.0)


def test_agreement_gate_requires_independent_reviewers():
    primary = _reviewed_record("same-reviewer")
    secondary = _reviewed_record("same-reviewer", ("second:1", "second:2"))

    comparison = compare_room_gold_records(primary, secondary)

    assert comparison["independent_reviewers"] is False
    assert comparison["agreement_gate_passed"] is False


def test_room_gold_contract_rejects_self_connectivity():
    record = _reviewed_record("reviewer-a")
    record["annotation"]["relations"][0]["target"] = "region:a"

    assert (
        "relations[0] cannot connect a region to itself"
        in validate_room_gold_record(record)
    )
