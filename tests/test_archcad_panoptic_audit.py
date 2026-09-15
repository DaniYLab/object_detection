from scripts.research.audit_archcad_panoptic_dataset import audit_annotation


def test_panoptic_audit_counts_contract_anomalies():
    result = audit_annotation(
        {
            "entities": [
                {
                    "type": "LINE",
                    "start": [0, 0],
                    "end": [10, 0],
                    "semantic": 20,
                },
                {
                    "type": "LINE",
                    "start": [0, 0],
                    "end": [0, 0],
                    "semantic": 1,
                },
                {
                    "type": "LINE",
                    "start": [0, 0],
                    "end": [2, 0],
                    "semantic": 1,
                    "instance": "door-a",
                },
            ]
        },
        image_id="sample",
    )

    assert result["primitive_count"] == 3
    assert result["segment_count"] == 3
    assert result["thing_without_instance"] == 1
    assert result["zero_length_primitives"] == 1
    assert result["semantic_counts"] == {20: 1, 1: 2}
