"""Reasoning selector for hybrid inference mode.

Determines when to use fast parallel predictions vs slower reasoning refinement.
Based on confidence thresholds and spatial ambiguity detection.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class ReasoningSelector(nn.Module):
    """Selective reasoning module for hybrid inference.

    Implements confidence-based and spatial ambiguity-based selection
    criteria from LocateAnything paper.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        spatial_threshold_px: int = 8,
    ):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.spatial_threshold_px = spatial_threshold_px

    def forward(
        self,
        parallel_output: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Select which predictions need reasoning refinement.

        Args:
            parallel_output: Dict with keys:
                - 'boxes': [B, num_classes, 4] (x, y, w, h)
                - 'confidence': [B, num_classes]

        Returns:
            needs_reasoning: [B, num_classes] boolean mask
            stats: Dict with selection statistics
        """
        confidence = parallel_output['confidence']  # [B, num_classes]
        boxes = parallel_output['boxes']  # [B, num_classes, 4]

        # Criterion 1: Low confidence
        low_confidence = confidence < self.confidence_threshold

        # Criterion 2: Spatial ambiguity
        # Check if multiple classes predict similar locations
        spatial_ambiguity = self._detect_spatial_ambiguity(boxes, confidence)

        # Combine criteria
        needs_reasoning = low_confidence | spatial_ambiguity

        # Statistics
        stats = {
            'total_predictions': confidence.numel(),
            'low_confidence_count': low_confidence.sum().item(),
            'spatial_ambiguity_count': spatial_ambiguity.sum().item(),
            'needs_reasoning_count': needs_reasoning.sum().item(),
            'needs_reasoning_ratio': needs_reasoning.float().mean().item(),
        }

        return needs_reasoning, stats

    def _detect_spatial_ambiguity(
        self,
        boxes: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        """Detect spatial ambiguity: multiple high-conf predictions at similar locations.

        Args:
            boxes: [B, num_classes, 4]
            confidence: [B, num_classes]

        Returns:
            ambiguous: [B, num_classes] boolean mask
        """
        B, num_classes, _ = boxes.shape
        device = boxes.device

        ambiguous = torch.zeros(B, num_classes, dtype=torch.bool, device=device)

        # Only check confident predictions
        confident_mask = confidence > self.confidence_threshold

        for b in range(B):
            conf_indices = torch.where(confident_mask[b])[0]
            if len(conf_indices) < 2:
                continue

            # Get centers for confident predictions
            centers = boxes[b, conf_indices, :2]  # [N, 2] (x, y)

            # Compute pairwise distances
            dists = torch.cdist(centers, centers)  # [N, N]

            # Find predictions with nearby neighbors
            close_pairs = dists < self.spatial_threshold_px
            close_pairs.fill_diagonal_(False)  # Exclude self

            # Mark as ambiguous if has close neighbor
            has_close_neighbor = close_pairs.any(dim=1)
            ambiguous[b, conf_indices[has_close_neighbor]] = True

        return ambiguous


def build_reasoning_selector(
    confidence_threshold: float = 0.7,
    spatial_threshold_px: int = 8,
) -> ReasoningSelector:
    """Build reasoning selector with specified thresholds.

    Args:
        confidence_threshold: Predictions below this need reasoning (default: 0.7)
        spatial_threshold_px: Distance threshold for spatial ambiguity (default: 8px)

    Returns:
        ReasoningSelector instance
    """
    return ReasoningSelector(
        confidence_threshold=confidence_threshold,
        spatial_threshold_px=spatial_threshold_px,
    )

