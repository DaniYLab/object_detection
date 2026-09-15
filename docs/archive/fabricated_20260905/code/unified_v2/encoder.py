"""Unified Encoder - Dual-pathway with cross-attention fusion

Components:
1. Image encoder: ResNet18 (fast, proven)
2. Vector encoder: Lightweight Transformer (4L, 128D)
3. Cross-attention fusion: Multi-head cross-attention
4. Multi-scale outputs: FPN-style pyramid features

Design philosophy:
- Speed: ResNet18 over ViT (3× faster)
- Quality: Cross-fusion captures image-vector interactions
- Robustness: Multi-scale features for small/large objects
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class VectorTransformer(nn.Module):
    """Lightweight transformer for SVG stroke encoding.

    Args:
        node_dim: Input feature dimension per stroke (x0,y0,x1,y1,...)
        embed_dim: Hidden dimension
        depth: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        node_dim: int = 12,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.embed_dim = embed_dim

        # Input projection
        self.input_proj = nn.Linear(node_dim, embed_dim)

        # Positional encoding (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1000, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Output normalization
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        strokes: torch.Tensor,
        stroke_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode SVG strokes.

        Args:
            strokes: [B, N, node_dim] where N is max strokes
            stroke_mask: [B, N] bool mask (True = valid stroke)

        Returns:
            [B, N, embed_dim] encoded stroke features
        """
        B, N, D = strokes.shape

        # Project to embedding space
        x = self.input_proj(strokes)  # [B, N, embed_dim]

        # Add positional encoding
        x = x + self.pos_embed[:, :N, :]

        # Create attention mask (True = ignore)
        if stroke_mask is not None:
            # Invert mask for transformer (True = attend)
            attn_mask = ~stroke_mask  # [B, N]
        else:
            attn_mask = None

        # Transformer encoding
        if attn_mask is not None:
            # Expand for multi-head attention [B*num_heads, N, N]
            src_key_padding_mask = ~stroke_mask  # True = ignore
            x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        else:
            x = self.transformer(x)

        # Output normalization
        x = self.norm(x)

        return x


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion between image and vector features.

    Strategy:
    - Query: Image features (what to look for)
    - Key/Value: Vector features (geometry information)
    - Output: Enriched image features with geometry awareness
    """

    def __init__(
        self,
        image_dim: int = 512,
        vector_dim: int = 128,
        output_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.image_dim = image_dim
        self.vector_dim = vector_dim
        self.output_dim = output_dim

        # Project image features to common dimension
        self.image_proj = nn.Linear(image_dim, output_dim)

        # Project vector features to common dimension
        self.vector_proj = nn.Linear(vector_dim, output_dim)

        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 4, output_dim),
            nn.Dropout(dropout),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(output_dim)
        self.norm2 = nn.LayerNorm(output_dim)

    def forward(
        self,
        image_features: torch.Tensor,
        vector_features: torch.Tensor,
        vector_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse image and vector features via cross-attention.

        Args:
            image_features: [B, H*W, image_dim] spatial image features
            vector_features: [B, N, vector_dim] stroke features
            vector_mask: [B, N] bool mask (True = valid)

        Returns:
            [B, H*W, output_dim] fused features
        """
        # Project to common dimension
        query = self.image_proj(image_features)  # [B, H*W, output_dim]
        key = self.vector_proj(vector_features)   # [B, N, output_dim]
        value = key  # Same as key

        # Key padding mask (True = ignore)
        if vector_mask is not None:
            key_padding_mask = ~vector_mask
        else:
            key_padding_mask = None

        # Cross-attention: image queries attend to vector keys/values
        attn_out, _ = self.cross_attention(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
        )

        # Residual + LayerNorm
        query = self.norm1(query + attn_out)

        # Feed-forward + Residual
        ffn_out = self.ffn(query)
        output = self.norm2(query + ffn_out)

        return output


class UnifiedEncoder(nn.Module):
    """Dual-pathway encoder with cross-fusion.

    Architecture:
    1. Image branch: ResNet18 → multi-scale features (res2, res3, res4)
    2. Vector branch: Transformer → stroke embeddings
    3. Cross-fusion: Enrich image features with vector geometry
    4. Output: Multi-scale fused features for FPN decoder
    """

    def __init__(
        self,
        vector_node_dim: int = 12,
        vector_embed_dim: int = 128,
        vector_depth: int = 4,
        fusion_dim: int = 256,
        num_fusion_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Image encoder: ResNet18 (pretrained)
        resnet = resnet18(weights='IMAGENET1K_V1')
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 64 channels, stride 4
        self.layer2 = resnet.layer2  # 128 channels, stride 8
        self.layer3 = resnet.layer3  # 256 channels, stride 16
        self.layer4 = resnet.layer4  # 512 channels, stride 32

        # Vector encoder
        self.vector_encoder = VectorTransformer(
            node_dim=vector_node_dim,
            embed_dim=vector_embed_dim,
            depth=vector_depth,
            num_heads=4,
            dropout=dropout,
        )

        # Cross-fusion modules (one per scale)
        self.fusion_res2 = CrossAttentionFusion(
            image_dim=128,
            vector_dim=vector_embed_dim,
            output_dim=fusion_dim,
            num_heads=num_fusion_heads,
            dropout=dropout,
        )

        self.fusion_res3 = CrossAttentionFusion(
            image_dim=256,
            vector_dim=vector_embed_dim,
            output_dim=fusion_dim,
            num_heads=num_fusion_heads,
            dropout=dropout,
        )

        self.fusion_res4 = CrossAttentionFusion(
            image_dim=512,
            vector_dim=vector_embed_dim,
            output_dim=fusion_dim,
            num_heads=num_fusion_heads,
            dropout=dropout,
        )

    def forward(
        self,
        images: torch.Tensor,
        strokes: torch.Tensor,
        stroke_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode image + vector, return multi-scale fused features.

        Args:
            images: [B, 3, 512, 512]
            strokes: [B, N, 12] SVG stroke coordinates
            stroke_mask: [B, N] bool mask (True = valid)

        Returns:
            Dictionary with keys:
            - 'res2': [B, 256, H/8, W/8] fused features at stride 8
            - 'res3': [B, 256, H/16, W/16] fused features at stride 16
            - 'res4': [B, 256, H/32, W/32] fused features at stride 32
        """
        B = images.shape[0]

        # ===== Image Branch (ResNet18) =====
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        res1 = self.layer1(x)  # [B, 64, H/4, W/4]
        res2 = self.layer2(res1)  # [B, 128, H/8, W/8]
        res3 = self.layer3(res2)  # [B, 256, H/16, W/16]
        res4 = self.layer4(res3)  # [B, 512, H/32, W/32]

        # ===== Vector Branch (Transformer) =====
        vector_feats = self.vector_encoder(strokes, stroke_mask)  # [B, N, 128]

        # ===== Cross-Fusion (Per Scale) =====
        # Flatten spatial dimensions for attention
        _, C2, H2, W2 = res2.shape
        res2_flat = res2.flatten(2).transpose(1, 2)  # [B, H2*W2, 128]
        fused2 = self.fusion_res2(res2_flat, vector_feats, stroke_mask)  # [B, H2*W2, 256]
        fused2 = fused2.transpose(1, 2).reshape(B, -1, H2, W2)  # [B, 256, H2, W2]

        _, C3, H3, W3 = res3.shape
        res3_flat = res3.flatten(2).transpose(1, 2)  # [B, H3*W3, 256]
        fused3 = self.fusion_res3(res3_flat, vector_feats, stroke_mask)  # [B, H3*W3, 256]
        fused3 = fused3.transpose(1, 2).reshape(B, -1, H3, W3)  # [B, 256, H3, W3]

        _, C4, H4, W4 = res4.shape
        res4_flat = res4.flatten(2).transpose(1, 2)  # [B, H4*W4, 512]
        fused4 = self.fusion_res4(res4_flat, vector_feats, stroke_mask)  # [B, H4*W4, 256]
        fused4 = fused4.transpose(1, 2).reshape(B, -1, H4, W4)  # [B, 256, H4, W4]

        return {
            'res2': fused2,  # Stride 8
            'res3': fused3,  # Stride 16
            'res4': fused4,  # Stride 32
        }
