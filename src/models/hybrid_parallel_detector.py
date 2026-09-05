"""Hybrid Parallel Detector: Fast parallel decoding + selective reasoning refinement.

Architecture:
- Parallel branch: Decode all 35 classes in ONE forward pass
- Confidence gating: Detect ambiguous predictions (conf < 0.7)
- Selective reasoning: Only refine ambiguous cases with teacher features
- Expected: 15-20× speedup + 2-5 AP50:95 improvement

Key insight: 90% predictions are confident, only 10% need reasoning refinement.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ParallelDecoder(nn.Module):
    """Parallel decoder for all classes at once."""

    def __init__(
        self,
        feature_dim: int = 256,
        num_classes: int = 35,
        grid_size: int = 64,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size

        # Shared feature extraction
        self.shared_conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.GroupNorm(32, feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.GroupNorm(32, feature_dim),
            nn.ReLU(inplace=True),
        )

        # Parallel prediction heads
        # Output: [B, num_classes, 5, H, W] where 5 = (center, w, h, offset_x, offset_y)
        self.parallel_head = nn.Conv2d(
            feature_dim,
            num_classes * 5,  # All classes × 5 channels
            kernel_size=1,
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            features: [B, feature_dim, H, W]

        Returns:
            dict with:
                'heatmaps': [B, num_classes, H, W] - center logits
                'sizes': [B, num_classes, 2, H, W] - width, height
                'offsets': [B, num_classes, 2, H, W] - offset x, y
                'confidence': [B, num_classes, H, W] - max confidence per class
        """
        B, _, H, W = features.shape

        # Shared features
        x = self.shared_conv(features)

        # Parallel predictions: [B, num_classes*5, H, W]
        preds = self.parallel_head(x)

        # Reshape to [B, num_classes, 5, H, W]
        preds = preds.view(B, self.num_classes, 5, H, W)

        # Split channels
        heatmaps = preds[:, :, 0, :, :]  # [B, num_classes, H, W]
        sizes = preds[:, :, 1:3, :, :]   # [B, num_classes, 2, H, W]
        offsets = preds[:, :, 3:5, :, :] # [B, num_classes, 2, H, W]

        # Compute confidence from heatmap
        confidence = torch.sigmoid(heatmaps)  # [B, num_classes, H, W]

        return {
            'heatmaps': heatmaps,
            'sizes': sizes,
            'offsets': offsets,
            'confidence': confidence,
        }


class SelectiveReasoning(nn.Module):
    """Selective reasoning refinement using teacher features."""

    def __init__(
        self,
        feature_dim: int = 256,
        reasoning_dim: int = 256,
        num_reasoning_steps: int = 4,
        grid_size: int = 64,
    ):
        super().__init__()
        self.num_reasoning_steps = num_reasoning_steps

        # Reasoning encoder: project teacher features to spatial domain
        self.reasoning_projection = nn.Sequential(
            nn.Linear(reasoning_dim, feature_dim),
            nn.ReLU(inplace=True),
        )

        # Cross-attention: use reasoning to refine predictions
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True,
        )

        # Refinement head
        self.refinement = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.GroupNorm(32, feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, 5, 1),  # Output 5 channels
        )

    def forward(
        self,
        features: torch.Tensor,
        reasoning_embedding: torch.Tensor,
        ambiguous_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Refine predictions for ambiguous cases.

        Args:
            features: [B, feature_dim, H, W]
            reasoning_embedding: [B, reasoning_dim] - teacher features
            ambiguous_mask: [B, H, W] - binary mask for ambiguous locations

        Returns:
            refinement: [B, 5, H, W] - refined predictions
        """
        B, C, H, W = features.shape

        # Project reasoning features
        reasoning = self.reasoning_projection(reasoning_embedding)  # [B, feature_dim]
        reasoning = reasoning.unsqueeze(1)  # [B, 1, feature_dim]

        # Reshape features to sequence
        features_seq = features.flatten(2).transpose(1, 2)  # [B, H*W, feature_dim]

        # Cross-attention: let reasoning guide feature refinement
        refined_seq, _ = self.cross_attention(
            query=features_seq,
            key=reasoning,
            value=reasoning,
        )  # [B, H*W, feature_dim]

        # Reshape back
        refined = refined_seq.transpose(1, 2).view(B, C, H, W)

        # Apply refinement only to ambiguous locations
        refinement = self.refinement(refined)  # [B, 5, H, W]

        # Mask: only refine where ambiguous_mask is True
        ambiguous_mask = ambiguous_mask.unsqueeze(1)  # [B, 1, H, W]
        refinement = refinement * ambiguous_mask.float()

        return refinement


class HybridParallelDetector(nn.Module):
    """Hybrid detector: Parallel + Selective Reasoning.

    Strategy:
    1. Encode image once (shared backbone)
    2. Parallel decode all 35 classes
    3. Detect ambiguous predictions (confidence < 0.7)
    4. Selectively refine only ambiguous cases with reasoning

    Expected:
    - Latency: ~12ms (vs 203ms sequential, ~17× speedup)
    - AP50:95: 0.33-0.35 (vs 0.306, +2-5 points)
    - Reasoning overhead: ~10% cases only
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int = 256,
        num_classes: int = 35,
        grid_size: int = 64,
        confidence_threshold: float = 0.7,
        use_reasoning: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.use_reasoning = use_reasoning

        # Parallel decoder
        self.parallel_decoder = ParallelDecoder(
            feature_dim=feature_dim,
            num_classes=num_classes,
            grid_size=grid_size,
        )

        # Selective reasoning (optional)
        if use_reasoning:
            self.selective_reasoning = SelectiveReasoning(
                feature_dim=feature_dim,
                reasoning_dim=256,
                grid_size=grid_size,
            )

    def forward(
        self,
        image: torch.Tensor,
        class_id: Optional[int] = None,
        reasoning_embedding: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            image: [B, 3, H, W]
            class_id: Optional class ID for single-class query (legacy support)
            reasoning_embedding: Optional [B, 256] teacher features

        Returns:
            dict with keys 'heatmap', 'size', 'offset', 'confidence', 'ambiguous_rate'
        """
        # Encode image once
        features = self.backbone(image)  # [B, feature_dim, H, W]

        # Parallel decode all classes
        parallel_output = self.parallel_decoder(features)

        heatmaps = parallel_output['heatmaps']  # [B, num_classes, H, W]
        sizes = parallel_output['sizes']        # [B, num_classes, 2, H, W]
        offsets = parallel_output['offsets']    # [B, num_classes, 2, H, W]
        confidence = parallel_output['confidence']  # [B, num_classes, H, W]

        # Detect ambiguous predictions
        ambiguous_mask = confidence < self.confidence_threshold  # [B, num_classes, H, W]
        ambiguous_rate = ambiguous_mask.float().mean().item()

        # Selective reasoning refinement
        if self.use_reasoning and reasoning_embedding is not None and ambiguous_rate > 0:
            # For each class, refine if any location is ambiguous
            for c in range(self.num_classes):
                class_ambiguous = ambiguous_mask[:, c, :, :]  # [B, H, W]

                if class_ambiguous.any():
                    # Refine this class
                    refinement = self.selective_reasoning(
                        features=features,
                        reasoning_embedding=reasoning_embedding,
                        ambiguous_mask=class_ambiguous,
                    )  # [B, 5, H, W]

                    # Apply refinement
                    heatmaps[:, c, :, :] += refinement[:, 0, :, :]
                    sizes[:, c, :, :] += refinement[:, 1:3, :, :]
                    offsets[:, c, :, :] += refinement[:, 3:5, :, :]

        # Legacy support: extract single class if class_id given
        if class_id is not None:
            return {
                'heatmap': heatmaps[:, class_id:class_id+1, :, :],  # [B, 1, H, W]
                'size': sizes[:, class_id, :, :],                    # [B, 2, H, W]
                'offset': offsets[:, class_id, :, :],                # [B, 2, H, W]
                'confidence': confidence[:, class_id, :, :],         # [B, H, W]
                'ambiguous_rate': ambiguous_rate,
            }

        return {
            'heatmap': heatmaps,      # [B, num_classes, H, W]
            'size': sizes,            # [B, num_classes, 2, H, W]
            'offset': offsets,        # [B, num_classes, 2, H, W]
            'confidence': confidence, # [B, num_classes, H, W]
            'ambiguous_rate': ambiguous_rate,
        }


def build_hybrid_parallel_detector(
    backbone_name: str = 'resnet50',
    num_classes: int = 35,
    use_reasoning: bool = True,
) -> HybridParallelDetector:
    """Build hybrid parallel detector with simple ConvNet backbone."""

    # Simple ConvNet backbone: 512×512 → 64×64 (stride-8)
    class SimpleBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                # 512 → 256
                nn.Conv2d(3, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),

                # 256 → 128
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),

                # 128 → 64
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),

                # Keep 64×64
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.encoder(x)  # [B, 256, 64, 64]

    backbone = SimpleBackbone()

    model = HybridParallelDetector(
        backbone=backbone,
        feature_dim=256,
        num_classes=num_classes,
        grid_size=64,
        confidence_threshold=0.7,
        use_reasoning=use_reasoning,
    )

    return model

