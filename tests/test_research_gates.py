from scripts.research.audit_drawing_research_gates import evaluate_gates


def test_research_gates_require_gold_agreement_and_provenance():
    blocked = evaluate_gates(
        {"valid": True, "reviewed_floor_plan_count": 0},
        {
            "common_record_count": 40,
            "eligible_review_pair_count": 0,
            "agreement_gate_pass_count": 0,
        },
        {
            "source_drawing_id_available": False,
            "drawing_disjoint_split_possible": False,
            "modality_id_sets_identical": False,
            "common_modality_id_count": 0,
        },
    )
    ready = evaluate_gates(
        {"valid": True, "reviewed_floor_plan_count": 100},
        {
            "common_record_count": 40,
            "eligible_review_pair_count": 40,
            "agreement_gate_pass_count": 40,
        },
        {
            "source_drawing_id_available": True,
            "drawing_disjoint_split_possible": True,
            "modality_id_sets_identical": True,
            "common_modality_id_count": 41097,
        },
    )

    assert blocked["structure_bakeoff_ready"] is False
    assert blocked["publication_evaluation_ready"] is False
    assert ready["structure_bakeoff_ready"] is True
    assert ready["publication_evaluation_ready"] is True


def test_direct_archcad_bakeoff_does_not_wait_for_optional_room_gold():
    result = evaluate_gates(
        {"valid": True, "reviewed_floor_plan_count": 0},
        {
            "common_record_count": 40,
            "eligible_review_pair_count": 0,
            "agreement_gate_pass_count": 0,
        },
        {
            "source_drawing_id_available": False,
            "drawing_disjoint_split_possible": False,
            "modality_id_sets_identical": True,
            "common_modality_id_count": 41097,
        },
    )

    assert result["structure_bakeoff_ready"] is True
    assert result["room_connectivity_extension_ready"] is False
    assert result["publication_evaluation_ready"] is False
    assert result["next_action"] == "run_direct_archcad_structure_bakeoff"
