"""Knowledge-distillation losses for one-pass student training.

The teacher (an early-conditioned dual-pathway detector) is run frozen and
selected-query, so its ``center_logits``/``size_map``/``offset_map`` align 1:1
with the student's, even though the student (``dual_pathway_late``) applies class
conditioning only at a compact late head. Distillation is response-based only:
soft targets on center logits and masked regression targets on size/offset. The
student's one-pass inference path is untouched by these losses.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_mse(
    student: torch.Tensor,
    teacher: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if student.shape != teacher.shape:
        raise ValueError(
            f"KD regression shape mismatch: student={student.shape}, teacher={teacher.shape}"
        )
    expanded = mask.expand_as(student).to(student.dtype)
    loss = F.mse_loss(student, teacher, reduction="none")
    denominator = expanded.sum().clamp_min(1.0)
    return (loss * expanded).sum() / denominator


def distillation_loss(
    student_preds: dict[str, torch.Tensor],
    teacher_preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    temperature: float = 2.0,
    center_weight: float = 1.0,
    size_weight: float = 0.5,
    offset_weight: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Response-based KD from a frozen teacher to a student.

    ``student_preds`` and ``teacher_preds`` carry ``center_logits``, ``size_map``
    and ``offset_map``; ``targets`` carries ``mask_map`` to gate both the
    center and the size/offset distillation to supervised center cells. This
    keeps the hard focal loss (which already suppresses background) from being
    fought by temperature-smeared soft background targets.
    """
    required = {"center_logits", "size_map", "offset_map"}
    missing_student = required.difference(student_preds)
    missing_teacher = required.difference(teacher_preds)
    if missing_student or missing_teacher:
        raise KeyError(
            f"KD missing student keys={sorted(missing_student)}, "
            f"teacher keys={sorted(missing_teacher)}"
        )
    if "mask_map" not in targets:
        raise KeyError("KD requires targets['mask_map'] to gate center/size/offset")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    mask = targets["mask_map"]
    teacher_logits = teacher_preds["center_logits"]
    student_logits = student_preds["center_logits"]
    if teacher_logits.shape != student_logits.shape:
        raise ValueError(
            f"KD center shape mismatch: student={student_logits.shape}, "
            f"teacher={teacher_logits.shape}"
        )
    # Soft targets from the teacher at temperature T, restricted to positive
    # center cells so background suppression stays the focal loss's job.
    soft_target = torch.sigmoid(teacher_logits / temperature).detach()
    center_cells = mask.expand_as(student_logits).to(student_logits.dtype)
    center_bce = F.binary_cross_entropy_with_logits(
        student_logits / temperature,
        soft_target,
        reduction="none",
    )
    center = (center_bce * center_cells).sum() / center_cells.sum().clamp_min(1.0)

    size = _masked_mse(student_preds["size_map"], teacher_preds["size_map"], mask)
    offset = _masked_mse(student_preds["offset_map"], teacher_preds["offset_map"], mask)

    return {
        "kd_center": center,
        "kd_size": size,
        "kd_offset": offset,
        "kd_total": center_weight * center + size_weight * size + offset_weight * offset,
    }