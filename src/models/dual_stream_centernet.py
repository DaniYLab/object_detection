"""DualStreamCenterNet: Hybrid architecture combining dual-stream encoder with proven CenterNet decoder.

Architecture:
- Encoder: DualStreamEncoder (ViT + Graph Transformer) from floorplan_llm.py
- Decoder: CenterNet heatmap-based head (proven with thin-head)
- Advantage: Novel dual-stream + reliable multi-instance handling

Expected: AP50:95 ≥ 0.28 (compare vs thin-head 0.3063)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.floorplan_llm import DualStreamEncoder
from src.models.detector import HeatmapHead


class SpatialProjection(nn.Module):
    """Project ViT tokens [B, T, D] back to spatial feature map [B, C, H, W]."""

    def __init__(
        self,
        embed_dim: int = 768,
        out_channels: int = 256,
        grid_size: int = 32,  # 32x32 patches from 512x512 image at patch_size=16
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.grid_size = grid_size

        # Project from embed_dim to out_channels
        self.projection = nn.Conv2d(embed_dim, out_channels, kernel_size=1)

        # Normalization
        self.norm = nn.GroupNorm(num_groups=32, num_channels=out_channels)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert tokens to spatial feature map.

        Args:
            tokens: [B, T, D] where T = grid_size^2

        Returns:
            [B, out_channels, grid_size, grid_size]
        """
        B, T, D = tokens.shape

        # Reshape to spatial grid
        H = W = self.grid_size
        assert T == H * W, f"Expected T={H*W}, got {T}"

        # [B, T, D] → [B, H, W, D] → [B, D, H, W]
        spatial = tokens.reshape(B, H, W, D).permute(0, 3, 1, 2)

        # Project and normalize
        features = self.projection(spatial)  # [B, out_channels, H, W]
        features = self.norm(features)

        return features


class DualStreamCenterNet(nn.Module):
    """Hybrid: Dual-stream encoder + CenterNet decoder.

    Combines:
    - Novel dual-stream encoder (image + SVG)
    - Proven CenterNet decoder (multi-instance handling)

    This gives us:
    - Innovation: dual-stream fusion
    - Reliability: proven decoder from thin-head
    """

    def __init__(
        self,
        image_size: int = 512,
        patch_size: int = 16,
        num_classes: int = 35,
        embed_dim: int = 768,
        vit_depth: int = 12,
        graph_depth: int = 6,
        num_heads: int = 12,
        decoder_channels: int = 256,
        output_stride: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        self.output_stride = output_stride

        # Encoder (from FloorPlanLLM)
        self.encoder = DualStreamEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim,
            vit_depth=vit_depth,
            graph_depth=graph_depth,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Spatial projection
        grid_size = image_size // patch_size  # 512 / 16 = 32
        self.spatial_projection = SpatialProjection(
            embed_dim=embed_dim,
            out_channels=decoder_channels,
            grid_size=grid_size,
        )

        # Upsample to match output_stride (32 → 64 for stride-8)
        # 32x32 → 64x64 (factor of 2)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(
                decoder_channels,
                decoder_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(num_groups=32, num_channels=decoder_channels),
            nn.ReLU(inplace=True),
        )

        # CenterNet heads (one per class)
        self.heads = nn.ModuleList([
            HeatmapHead(
                in_dim=decoder_channels,
                out_channels=5,  # (center, w, h, offset_x, offset_y)
                hidden_dim=128,
            )
            for _ in range(num_classes)
        ])

    def forward(
        self,
        image: torch.Tensor,
        stroke_tokens: torch.Tensor,
        stroke_mask: torch.Tensor,
        class_id: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            image: [B, 3, 512, 512]
            stroke_tokens: [B, N, 12]
            stroke_mask: [B, N]
            class_id: Optional single class to predict (for training with QueryDataset)
                     If None, predict all classes (for evaluation)

        Returns:
            If class_id is given:
                {
                    'heatmap': [B, 1, H, W],
                    'size': [B, 2, H, W],
                    'offset': [B, 2, H, W],
                }
            If class_id is None:
                {
                    'heatmap': [B, num_classes, H, W],
                    'size': [B, num_classes, 2, H, W],
                    'offset': [B, num_classes, 2, H, W],
                }
        """
        # Encode with dual-stream
        img_tokens, svg_tokens = self.encoder(image, stroke_tokens, stroke_mask)
        # img_tokens: [B, 1024, 768] (32x32 patches)

        # Project to spatial feature map
        features = self.spatial_projection(img_tokens)  # [B, 256, 32, 32]

        # Upsample to 64x64 (stride-8 resolution)
        features = self.upsample(features)  # [B, 256, 64, 64]

        # Single-class mode (training with QueryDataset)
        if class_id is not None:
            output = self.heads[class_id](features)  # [B, 5, H, W]
            return {
                'heatmap': output[:, 0:1],  # [B, 1, H, W]
                'size': output[:, 1:3],      # [B, 2, H, W]
                'offset': output[:, 3:5],    # [B, 2, H, W]
            }

        # All-class mode (evaluation)
        B, _, H, W = features.shape

        heatmaps = []
        sizes = []
        offsets = []

        for class_idx, head in enumerate(self.heads):
            output = head(features)  # [B, 5, H, W]

            # Split channels: center (1), size (2), offset (2)
            center_logits = output[:, 0:1]  # [B, 1, H, W]
            size_pred = output[:, 1:3]      # [B, 2, H, W]
            offset_pred = output[:, 3:5]    # [B, 2, H, W]

            heatmaps.append(center_logits)
            sizes.append(size_pred)
            offsets.append(offset_pred)

        # Stack per-class predictions
        heatmap = torch.cat(heatmaps, dim=1)  # [B, num_classes, H, W]
        size = torch.stack(sizes, dim=1)      # [B, num_classes, 2, H, W]
        offset = torch.stack(offsets, dim=1)  # [B, num_classes, 2, H, W]

        return {
            'heatmap': heatmap,
            'size': size,
            'offset': offset,
        }


def build_dual_stream_centernet(
    preset: str = 'small',
    num_classes: int = 35,
    output_stride: int = 8,
) -> DualStreamCenterNet:
    """Factory function.

    Presets:
    - 'tiny': Fast (6 layers ViT, 3 layers Graph, 384 dim)
    - 'small': Balanced (8 layers, 4 layers, 512 dim) ← recommended
    - 'base': Full (12 layers, 6 layers, 768 dim)
    """
    configs = {
        'tiny': {
            'embed_dim': 384,
            'vit_depth': 6,
            'graph_depth': 3,
            'num_heads': 6,
            'decoder_channels': 128,
        },
        'small': {
            'embed_dim': 512,
            'vit_depth': 8,
            'graph_depth': 4,
            'num_heads': 8,
            'decoder_channels': 256,
        },
        'base': {
            'embed_dim': 768,
            'vit_depth': 12,
            'graph_depth': 6,
            'num_heads': 12,
            'decoder_channels': 256,
        },
    }

    if preset not in configs:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(configs.keys())}")

    config = configs[preset]
    return DualStreamCenterNet(
        num_classes=num_classes,
        output_stride=output_stride,
        **config,
    )
