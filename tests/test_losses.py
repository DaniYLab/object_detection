from __future__ import annotations

import pytest
import torch

from src.training.losses import centernet_loss, focal_loss, l1_loss_masked


def test_focal_loss_logits_has_finite_gradient() -> None:
    logits = torch.tensor([[[[-20.0, 20.0]]]], requires_grad=True)
    target = torch.tensor([[[[1.0, 0.0]]]])

    loss = focal_loss(None, target, logits=logits)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_masked_l1_only_uses_centers() -> None:
    pred = torch.tensor([[[[2.0, 100.0]], [[4.0, 100.0]]]])
    target = torch.tensor([[[[1.0, 0.0]], [[2.0, 0.0]]]])
    mask = torch.tensor([[[[1.0, 0.0]]]])

    loss = l1_loss_masked(pred, target, mask)

    assert loss.item() == pytest.approx(1.0)


def test_centernet_loss_prefers_logits_and_validates_resolution() -> None:
    predictions = {
        "center_logits": torch.zeros(1, 1, 2, 2),
        "center_heatmap": torch.full((1, 1, 2, 2), 0.5),
        "size_map": torch.ones(1, 2, 2, 2),
        "offset_map": torch.full((1, 2, 2, 2), 0.5),
    }
    targets = {
        "center_heatmap": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]]),
        "size_map": torch.ones(1, 2, 2, 2),
        "offset_map": torch.full((1, 2, 2, 2), 0.5),
        "mask_map": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]]),
    }

    result = centernet_loss(predictions, targets)
    assert set(result) == {"total", "focal", "size_l1", "offset_l1", "num_pos"}
    assert result["size_l1"].item() == 0.0
    assert result["offset_l1"].item() == 0.0

    targets["center_heatmap"] = torch.zeros(1, 1, 1, 1)
    with pytest.raises(ValueError, match="prediction resolution"):
        centernet_loss(predictions, targets)
