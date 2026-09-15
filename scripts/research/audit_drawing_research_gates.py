"""Combine room-gold, agreement, and provenance evidence into decision gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def evaluate_gates(
    room_validation: Mapping[str, Any],
    agreement: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    primary_ready = bool(room_validation.get("valid")) and int(
        room_validation.get("reviewed_floor_plan_count", 0)
    ) >= 100
    secondary_ready = (
        int(agreement.get("common_record_count", 0)) >= 40
        and int(agreement.get("eligible_review_pair_count", 0)) >= 40
        and int(agreement.get("agreement_gate_pass_count", 0))
        == int(agreement.get("eligible_review_pair_count", 0))
    )
    provenance_ready = bool(provenance.get("source_drawing_id_available")) and bool(
        provenance.get("drawing_disjoint_split_possible")
    )
    direct_gt_ready = bool(provenance.get("modality_id_sets_identical")) and int(
        provenance.get("common_modality_id_count", 0)
    ) > 0
    return {
        "schema_version": "drawing-research-gates-v0",
        "gates": {
            "primary_room_gold": {
                "passed": primary_ready,
                "reviewed_floor_plans": int(
                    room_validation.get("reviewed_floor_plan_count", 0)
                ),
                "target": 100,
            },
            "independent_double_annotation": {
                "passed": secondary_ready,
                "eligible_pairs": int(
                    agreement.get("eligible_review_pair_count", 0)
                ),
                "passing_pairs": int(
                    agreement.get("agreement_gate_pass_count", 0)
                ),
                "target": 40,
            },
            "source_drawing_provenance": {
                "passed": provenance_ready,
                "source_drawing_id_available": bool(
                    provenance.get("source_drawing_id_available")
                ),
            },
            "direct_archcad_ground_truth": {
                "passed": direct_gt_ready,
                "aligned_sample_count": int(
                    provenance.get("common_modality_id_count", 0)
                ),
                "contract": "primitive geometry + semantic + instance",
            },
        },
        "structure_bakeoff_ready": direct_gt_ready,
        "room_connectivity_extension_ready": primary_ready and secondary_ready,
        "publication_evaluation_ready": direct_gt_ready and provenance_ready,
        "next_action": (
            "run_direct_archcad_structure_bakeoff"
            if direct_gt_ready
            else "restore_aligned_archcad_modalities"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--room-validation",
        type=Path,
        default=Path("outputs/drawing_graph/room_gold_pack_v0.validation.json"),
    )
    parser.add_argument(
        "--agreement",
        type=Path,
        default=Path("outputs/drawing_graph/room_gold_agreement_v0.json"),
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path("outputs/drawing_graph/archcad_provenance_audit.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/drawing_graph/research_gates_v0.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_gates(
        json.loads(args.room_validation.read_text(encoding="utf-8")),
        json.loads(args.agreement.read_text(encoding="utf-8")),
        json.loads(args.provenance.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
