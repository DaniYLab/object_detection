from __future__ import annotations

import pytest
import torch

from src.training.distillation import distillation_loss


def _batch():
    return {
        "center_logits": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]]]),
        "size_map": torch.ones(1, 2, 2, 2),
        "offset_map": torch.full((1, 2, 2, 2), 0.5),
    }


def test_distillation_loss_matches_when_identical() -> None:
    student = _batch()
    teacher = _batch()
    targets = {
        "mask_map": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]]),
    }

    result = distillation_loss(student, teacher, targets)

    # Identical size/offset → zero regression loss at the single masked cell.
    # Center BCE against a soft target has an irreducible entropy floor, so it is
    # strictly positive even when identical — assert finite and zero regression.
    assert result["kd_size"].item() == pytest.approx(0.0, abs=1e-6)
    assert result["kd_offset"].item() == pytest.approx(0.0, abs=1e-6)
    assert torch.isfinite(result["kd_center"])
    assert torch.isfinite(result["kd_total"])
    assert set(result) == {
        "kd_center",
        "kd_size",
        "kd_offset",
        "kd_total",
    }


def test_distillation_center_gated_by_positive_cells() -> None:
    student = _batch()
    teacher = _batch()
    teacher["center_logits"] = -student["center_logits"]
    # Zero mask → center KD must be zero even when logits diverge.
    targets = {
        "mask_map": torch.zeros(1, 1, 2, 2),
    }

    result = distillation_loss(student, teacher, targets)

    assert result["kd_center"].item() == pytest.approx(0.0, abs=1e-6)
    assert result["kd_size"].item() == pytest.approx(0.0, abs=1e-6)
    assert result["kd_offset"].item() == pytest.approx(0.0, abs=1e-6)
    assert result["kd_total"].item() == pytest.approx(0.0, abs=1e-6)


def test_distillation_loss_penalises_divergence() -> None:
    student = _batch()
    teacher = _batch()
    teacher["center_logits"] = -student["center_logits"]
    teacher["size_map"] = student["size_map"] + 1.0
    targets = {
        "mask_map": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]]),
    }

    result = distillation_loss(student, teacher, targets)

    assert result["kd_center"].item() > 0.0
    assert result["kd_size"].item() == pytest.approx(1.0)
    assert result["kd_total"].item() > 0.0


def test_distillation_loss_validates_shapes_and_args() -> None:
    student = _batch()
    teacher = _batch()
    teacher["size_map"] = torch.ones(1, 3, 2, 2)
    targets = {"mask_map": torch.tensor([[[[1.0]]]])}

    with pytest.raises(ValueError, match="shape mismatch"):
        distillation_loss(student, teacher, targets)

    with pytest.raises(ValueError, match="positive"):
        distillation_loss(student, _batch(), {"mask_map": targets["mask_map"]}, temperature=0.0)
