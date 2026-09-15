import copy

from scripts.research.build_room_gold_double_pack import build_double_review_records


def _record(index: int):
    return {
        "schema_version": "room-gold-v0",
        "image_id": f"image-{index:03d}",
        "image_path": f"image-{index:03d}.png",
        "annotation_path": f"image-{index:03d}.json",
        "selection": {"index": index},
        "preannotations": {"regions": [{"candidate_id": f"hint-{index}"}]},
        "review_status": "reviewed",
        "reviewer": "primary-reviewer",
        "annotation": {
            "view_type": "floor_plan",
            "regions": [
                {
                    "id": "secret-primary-region",
                    "type": "room",
                    "polygon": [[0, 0], [10, 0], [10, 10]],
                }
            ],
            "relations": [],
        },
    }


def test_double_pack_is_deterministic_and_blinds_primary_annotations():
    records = [_record(index) for index in range(10)]
    original = copy.deepcopy(records)

    first = build_double_review_records(records, fraction=0.2, seed=1337)
    second = build_double_review_records(records, fraction=0.2, seed=1337)

    assert first == second
    assert len(first) == 2
    assert records == original
    assert all(record["review_status"] == "unreviewed" for record in first)
    assert all(record["reviewer"] is None for record in first)
    assert all(record["annotation"]["regions"] == [] for record in first)
    assert all(record["preannotations"]["regions"] for record in first)


def test_double_pack_rejects_invalid_fraction():
    try:
        build_double_review_records([_record(0)], fraction=0.0, seed=1)
    except ValueError as exc:
        assert "fraction" in str(exc)
    else:
        raise AssertionError("expected invalid fraction to fail")
