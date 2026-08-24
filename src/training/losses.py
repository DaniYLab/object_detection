"""CenterNet losses with explicit shape and numerical checks."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_loss(
    pred: torch.Tensor | None,
    target: torch.Tensor,
    *,
    logits: torch.Tensor | None = None,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    """Penalty-reduced CenterNet focal loss.

    Pass ``logits`` when available for stable log-sigmoid calculations. ``pred``
    is retained for compatibility with models that only expose probabilities.
    Targets must contain exact 1.0 peaks at the prediction resolution.
    """
    if logits is None and pred is None:
        raise ValueError("Either pred or logits must be provided")

    reference = logits if logits is not None else pred
    assert reference is not None
    if reference.shape != target.shape:
        raise ValueError(
            f"Heatmap shape mismatch: prediction={reference.shape}, target={target.shape}"
        )

    pos_inds = target.eq(1).to(reference.dtype)
    neg_inds = target.lt(1).to(reference.dtype)
    neg_weights = torch.pow(1 - target, beta)

    if logits is not None:
        probability = torch.sigmoid(logits)
        pos_loss = F.logsigmoid(logits) * torch.pow(1 - probability, alpha) * pos_inds
        neg_loss = (
            F.logsigmoid(-logits)
            * torch.pow(probability, alpha)
            * neg_weights
            * neg_inds
        )
    else:
        assert pred is not None
        probability = pred.clamp(1e-6, 1 - 1e-6)
        pos_loss = (
            torch.log(probability)
            * torch.pow(1 - probability, alpha)
            * pos_inds
        )
        neg_loss = (
            torch.log1p(-probability)
            * torch.pow(probability, alpha)
            * neg_weights
            * neg_inds
        )

    num_pos = pos_inds.sum()
    if num_pos.item() == 0:
        return -neg_loss.mean()
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos


def l1_loss_masked(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Smooth L1 regression loss at supervised center cells only."""
    if pred.shape != target.shape:
        raise ValueError(
            f"Regression shape mismatch: pred={pred.shape}, target={target.shape}"
        )
    if mask.ndim != pred.ndim or mask.shape[0] != pred.shape[0]:
        raise ValueError(f"Mask batch/rank mismatch: mask={mask.shape}, pred={pred.shape}")
    if mask.shape[-2:] != pred.shape[-2:]:
        raise ValueError(f"Mask spatial mismatch: mask={mask.shape}, pred={pred.shape}")
    if mask.shape[1] not in {1, pred.shape[1]}:
        raise ValueError(f"Mask channel mismatch: mask={mask.shape}, pred={pred.shape}")

    expanded_mask = mask.expand_as(pred).to(pred.dtype)
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    denominator = expanded_mask.sum().clamp_min(1.0)
    return (loss * expanded_mask).sum() / denominator


def centernet_loss(
    preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    focal_w: float = 10.0,
    size_w: float = 1.0,
    offset_w: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Combined CenterNet center, size, and fractional-offset losses."""
    required_preds = {"center_heatmap", "size_map", "offset_map"}
    required_targets = {
        "center_heatmap",
        "size_map",
        "offset_map",
        "mask_map",
    }
    missing_preds = required_preds.difference(preds)
    missing_targets = required_targets.difference(targets)
    if missing_preds or missing_targets:
        raise KeyError(
            f"Missing prediction keys={sorted(missing_preds)}, "
            f"target keys={sorted(missing_targets)}"
        )

    pred_hm = preds["center_heatmap"]
    target_hm = targets["center_heatmap"]
    if pred_hm.shape[-2:] != target_hm.shape[-2:]:
        raise ValueError(
            "CenterNet targets must match prediction resolution. "
            f"pred={pred_hm.shape[-2:]}, target={target_hm.shape[-2:]}"
        )

    heatmap_logits = preds.get("center_logits")
    center = focal_loss(pred_hm, target_hm, logits=heatmap_logits)
    size = l1_loss_masked(preds["size_map"], targets["size_map"], targets["mask_map"])
    offset = l1_loss_masked(
        preds["offset_map"], targets["offset_map"], targets["mask_map"]
    )
    total = focal_w * center + size_w * size + offset_w * offset
    return {
        "total": total,
        "focal": center,
        "size_l1": size,
        "offset_l1": offset,
        "num_pos": targets["mask_map"].sum(),
    }
