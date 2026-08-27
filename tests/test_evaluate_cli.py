from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

import evaluate
from scripts.dev import calibrate_thresholds, viz_heatmap
from src.data.constants import CLASS_TO_IDX
from src.data.splits import build_split_manifest_v2, write_split_manifest
from src.evaluation import write_predictions
from src.training.checkpoint import CheckpointError


def test_external_prediction_cli_computes_perfect_ap(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    test_dir = data_root / "test_set"
    test_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), "white").save(test_dir / "sample.png")
    (test_dir / "sample_meta.json").write_text(
        json.dumps(
            {
                "image_size": [8, 8],
                "svg_viewbox": [0, 0, 8, 8],
                "instances": [
                    {
                        "class": "chair",
                        "class_id": CLASS_TO_IDX["chair"],
                        "instance_id": 1,
                        "bbox_px": [1.0, 1.0, 5.0, 5.0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    prediction_path = tmp_path / "predictions.json"
    write_predictions(
        prediction_path,
        [
            {
                "image_id": "test_set/sample",
                "boxes": [[1.0, 1.0, 5.0, 5.0]],
                "scores": [0.9],
                "labels": [CLASS_TO_IDX["chair"]],
            }
        ],
    )
    report_path = tmp_path / "report.json"

    result = evaluate.main(
        [
            "--data-root",
            str(data_root),
            "--split",
            "test",
            "--predictions-json",
            str(prediction_path),
            "--report",
            str(report_path),
            "--image-size",
            "8",
            "--num-workers",
            "0",
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result == 0
    assert report["metrics"]["AP50"] == 1.0
    assert report["metrics"]["AP50:95"] == 1.0
    assert report["data"]["split"] == "test"



def test_v2_evaluation_uses_explicit_role_and_records_provenance(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    train_dir = data_root / "train_set_1"
    train_dir.mkdir(parents=True)
    for index in range(12):
        Image.new("RGB", (8, 8), "white").save(train_dir / f"sample{index}.png")
        (train_dir / f"sample{index}_meta.json").write_text(
            json.dumps(
                {
                    "image_size": [8, 8],
                    "svg_viewbox": [0, 0, 8, 8],
                    "instances": [
                        {
                            "class": "chair",
                            "class_id": CLASS_TO_IDX["chair"],
                            "instance_id": 1,
                            "bbox_px": [1.0, 1.0, 5.0, 5.0],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
    manifest_path = data_root / "splits_v2.json"
    write_split_manifest(build_split_manifest_v2(data_root), manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = manifest["splits"]["val_select"][0]
    image_id = selected["image_id"]
    prediction_path = tmp_path / "predictions.json"
    write_predictions(
        prediction_path,
        [
            {
                "image_id": image_id,
                "boxes": [[1.0, 1.0, 5.0, 5.0]],
                "scores": [0.9],
                "labels": [CLASS_TO_IDX["chair"]],
            }
        ],
    )
    report_path = tmp_path / "report.json"
    evaluate.main(
        [
            "--data-root", str(data_root),
            "--manifest", str(manifest_path),
            "--split", "val_select",
            "--predictions-json", str(prediction_path),
            "--report", str(report_path),
            "--image-size", "8",
        ]
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["data"]["requested_split"] == "val_select"
    assert report["data"]["split"] == "val_select"
    assert report["data"]["manifest_schema_version"] == 2
    assert report["data"]["manifest_fingerprint"] == manifest["fingerprint"]
    assert report["data"]["role_fingerprint"] == manifest["role_fingerprints"]["val_select"]


def test_evaluate_and_viz_reject_pre_schema_checkpoints_early(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "pre_schema.pt"
    torch.save(
        {
            "model_state": {"historical_encoder.weight": torch.ones(1)},
            "args": {"model_dim": 16},
        },
        checkpoint_path,
    )

    with pytest.raises(
        CheckpointError,
        match="pre-schema.*cannot be migrated automatically",
    ):
        evaluate._load_model(checkpoint_path, device=torch.device("cpu"), image_size=32)

    with pytest.raises(
        CheckpointError,
        match="pre-schema.*cannot be migrated automatically",
    ):
        viz_heatmap._checkpoint_model(str(checkpoint_path), image_size=32)


def test_calibration_rejects_test_roles_before_reading_predictions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot use the test_set role"):
        calibrate_thresholds.main(
            [
                "--predictions", str(tmp_path / "missing.json"),
                "--split", "test_set",
            ]
        )


def test_manifest_provenance_requires_an_exact_checkpoint_match(
    capsys: pytest.CaptureFixture[str],
) -> None:
    matching = {"split_manifest_fingerprint": "manifest-a"}
    evaluate._validate_manifest_provenance(
        matching,
        "manifest-a",
        allow_mismatch=False,
    )

    for checkpoint in (
        {"split_manifest_fingerprint": "manifest-b"},
        {},
    ):
        with pytest.raises(CheckpointError, match="manifest fingerprint mismatch"):
            evaluate._validate_manifest_provenance(
                checkpoint,
                "manifest-a",
                allow_mismatch=False,
            )

    evaluate._validate_manifest_provenance(
        {},
        "manifest-a",
        allow_mismatch=True,
    )
    assert "WARNING" in capsys.readouterr().out
