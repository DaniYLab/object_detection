"""FloorPlanNet v2.1 - Complete Unified Model

Integrates:
1. Dual-pathway encoder (ResNet18 + Transformer)
2. Cross-attention fusion
3. FPN multi-scale features
4. Hybrid parallel decoder
5. Stride-4 refinement for small objects
6. Adaptive class routing
7. Confidence monitoring

Usage:
    model = FloorPlanNetV2(num_classes=35)
    outputs = model(images, strokes, stroke_mask)

    # Training mode: Returns all detection outputs
    # Inference mode: Includes confidence monitoring for hybrid fallback
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import UnifiedEncoder
from .decoder import HybridMultiScaleDecoder


class FloorPlanNetV2(nn.Module):
    """Complete FloorPlanNet v2.1 architecture.

    Expected performance:
    - AP50: 0.52-0.56
    - AP50:95: 0.32-0.36
    - Speed: 2-3× faster than thin-head baseline
    - Training time: 70 epochs, ~3-4 days

    Args:
        num_classes: Number of detection classes
        image_size: Input image size (default 512)
        vector_node_dim: SVG stroke feature dimension
        vector_embed_dim: Vector transformer hidden dimension
        vector_depth: Vector transformer layers
        fusion_dim: Cross-attention output dimension
        fpn_channels: FPN output channels
        enable_stride4: Enable stride-4 head for small objects
        confidence_threshold: Threshold for hybrid inference fallback
        small_object_classes: List of class indices for stride-4 refinement
    """

    def __init__(
        self,
        num_classes: int = 35,
        image_size: int = 512,
        vector_node_dim: int = 12,
        vector_embed_dim: int = 128,
        vector_depth: int = 4,
        fusion_dim: int = 256,
        fpn_channels: int = 256,
        enable_stride4: bool = True,
        confidence_threshold: float = 0.7,
        small_object_classes: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size

        # Unified encoder (dual-pathway + cross-fusion)
        self.encoder = UnifiedEncoder(
            vector_node_dim=vector_node_dim,
            vector_embed_dim=vector_embed_dim,
            vector_depth=vector_depth,
            fusion_dim=fusion_dim,
            num_fusion_heads=8,
            dropout=dropout,
        )

        # Hybrid multi-scale decoder
        self.decoder = HybridMultiScaleDecoder(
            num_classes=num_classes,
            fpn_channels=fpn_channels,
            enable_stride4=enable_stride4,
            confidence_threshold=confidence_threshold,
            small_object_classes=small_object_classes,
        )

    def forward(
        self,
        images: torch.Tensor,
        strokes: torch.Tensor,
        stroke_mask: torch.Tensor | None = None,
        return_routing: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            images: [B, 3, H, W] input images
            strokes: [B, N, 12] SVG stroke coordinates
            stroke_mask: [B, N] bool mask (True = valid stroke)
            return_routing: Return adaptive routing probabilities

        Returns:
            Dict with keys:
            - 'heatmap': [B, num_classes, H/8, W/8] class heatmaps
            - 'size': [B, 2, H/8, W/8] object sizes (w, h)
            - 'offset': [B, 2, H/8, W/8] sub-pixel offsets (dx, dy)
            - 'stride': int, output stride (8)
            - 'max_confidence': float (inference only)
            - 'low_confidence': bool (inference only)
            - 'stride4_used': bool (if stride-4 refinement applied)
            - 'class_routing': [B, num_classes] (if return_routing=True)
        """
        # Encode: dual-pathway + cross-fusion
        encoder_features = self.encoder(images, strokes, stroke_mask)

        # Decode: multi-scale hybrid
        outputs = self.decoder(encoder_features, return_routing=return_routing)

        return outputs

    def get_num_params(self) -> dict[str, int]:
        """Count parameters by component."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        total_params = encoder_params + decoder_params

        return {
            'encoder': encoder_params,
            'decoder': decoder_params,
            'total': total_params,
        }

    @torch.no_grad()
    def estimate_flops(self, batch_size: int = 1) -> dict[str, float]:
        """Estimate FLOPs (rough approximation).

        Returns:
            Dict with FLOPs in GFLOPs for different components
        """
        # Rough estimates based on layer types
        # ResNet18: ~1.8 GFLOPs for 224x224, scale to 512x512
        resnet_flops = 1.8 * (512 / 224) ** 2  # ~9.5 GFLOPs

        # Transformer: 4 layers × (self-attention + FFN)
        # Self-attention: 4 * N * D^2 (N=max strokes, D=128)
        # FFN: 8 * N * D^2
        N_strokes = 1000
        D = 128
        transformer_flops = 4 * (4 * N_strokes * D**2 + 8 * N_strokes * D**2) / 1e9  # ~1.0 GFLOPs

        # Cross-attention: Q @ K^T + softmax + @ V, per scale
        # 3 scales × (2 * H*W * N * D)
        HW_total = (64*64 + 32*32 + 16*16)  # Sum of spatial sizes
        cross_attn_flops = 3 * 2 * HW_total * N_strokes * 256 / 1e9  # ~2.5 GFLOPs

        # FPN: Top-down pathway with upsampling + convs
        fpn_flops = 0.5  # ~0.5 GFLOPs

        # Decoder heads: 3 conv layers + 3 output heads
        decoder_flops = 1.0  # ~1.0 GFLOPs

        total_flops = resnet_flops + transformer_flops + cross_attn_flops + fpn_flops + decoder_flops

        return {
            'resnet18': resnet_flops,
            'vector_transformer': transformer_flops,
            'cross_attention': cross_attn_flops,
            'fpn': fpn_flops,
            'decoder': decoder_flops,
            'total': total_flops,
        }


def create_floorplannet_v2(
    num_classes: int = 35,
    preset: str = 'default',
) -> FloorPlanNetV2:
    """Factory function for creating FloorPlanNet v2.1 models.

    Presets:
    - 'default': Full model with all features (12M params)
    - 'fast': Lighter model for speed (8M params)
    - 'small': For resource-constrained deployment (6M params)

    Args:
        num_classes: Number of detection classes
        preset: Model preset name

    Returns:
        FloorPlanNetV2 model instance
    """
    if preset == 'default':
        return FloorPlanNetV2(
            num_classes=num_classes,
            vector_embed_dim=128,
            vector_depth=4,
            fusion_dim=256,
            enable_stride4=True,
        )
    elif preset == 'fast':
        return FloorPlanNetV2(
            num_classes=num_classes,
            vector_embed_dim=96,  # Smaller
            vector_depth=3,  # Fewer layers
            fusion_dim=192,  # Smaller fusion
            enable_stride4=False,  # Skip stride-4 for speed
        )
    elif preset == 'small':
        return FloorPlanNetV2(
            num_classes=num_classes,
            vector_embed_dim=64,  # Minimal
            vector_depth=2,  # Minimal layers
            fusion_dim=128,  # Minimal fusion
            enable_stride4=False,
            dropout=0.05,  # Less dropout for small model
        )
    else:
        raise ValueError(f"Unknown preset: {preset}. Choose from 'default', 'fast', 'small'")


# Convenience exports
__all__ = [
    'FloorPlanNetV2',
    'create_floorplannet_v2',
]
