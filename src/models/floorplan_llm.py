"""FloorPlanLLM: Structured Understanding Model for Floor Plan Reading.

Big architecture change:
- Dual-stream encoder (ViT for image + Graph Transformer for SVG)
- Cross-attention fusion between raster and vector
- Parallel box decoder (LocateAnything-inspired)
- Autoregressive scene graph decoder
- Hybrid inference mode (confidence-based fallback)

Output: Structured scene graph (nodes + edges + relationships)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import SelfAttention, FFN
from .vector_encoder import VectorEncoder


class PatchEmbedding(nn.Module):
    """Convert image into patch tokens (ViT-style)."""

    def __init__(
        self,
        image_size: int = 512,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert [B, C, H, W] → [B, num_patches, embed_dim]."""
        # x: [B, 3, 512, 512]
        x = self.projection(x)  # [B, embed_dim, 32, 32]
        x = x.flatten(2).transpose(1, 2)  # [B, 1024, embed_dim]
        return x


class TransformerBlock(nn.Module):
    """Standard Transformer block: pre-norm self-attention + FFN."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply self-attention + FFN with residual connections."""
        x = x + self.attn(self.norm1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """ViT encoder for raster image."""

    def __init__(
        self,
        image_size: int = 512,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        # Learnable position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image into tokens.

        Args:
            x: [B, 3, 512, 512]

        Returns:
            [B, num_patches, embed_dim]
        """
        x = self.patch_embed(x)  # [B, 1024, embed_dim]
        x = x + self.pos_embed  # Add positional encoding

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x


class GraphTransformer(nn.Module):
    """Transformer encoder for SVG graph (reuse VectorEncoder structure)."""

    def __init__(
        self,
        node_dim: int = 12,
        embed_dim: int = 768,
        depth: int = 6,
        num_heads: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.embed_dim = embed_dim

        # Project SVG features to shared embedding space
        self.input_projection = nn.Linear(node_dim, embed_dim)

        # Type embedding (line vs arc)
        self.type_embedding = nn.Embedding(2, embed_dim)
        nn.init.trunc_normal_(self.type_embedding.weight, std=0.02)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, 4.0, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        stroke_tokens: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode SVG primitives.

        Args:
            stroke_tokens: [B, N, 12] SVG features
            valid_mask: [B, N] True for valid primitives

        Returns:
            [B, N, embed_dim]
        """
        batch, length, _ = stroke_tokens.shape

        # Type detection: arc slots have non-zero radius (dim 6)
        type_ids = (stroke_tokens[..., 6] != 0).to(dtype=torch.long)

        # Project and add type embedding
        x = self.input_projection(stroke_tokens) + self.type_embedding(type_ids)

        for block in self.blocks:
            x = block(x, attention_mask=valid_mask)

        x = self.norm(x)

        # Zero out padding positions
        x = x * valid_mask.to(x.dtype).unsqueeze(-1)
        return x


class CrossAttentionFusion(nn.Module):
    """Cross-attention between image tokens and SVG tokens."""

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim

        # Image queries SVG (image attends to vector)
        self.norm_img = nn.LayerNorm(dim)
        self.norm_svg = nn.LayerNorm(dim)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        img_tokens: torch.Tensor,
        svg_tokens: torch.Tensor,
        svg_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cross-attention: image queries SVG.

        Args:
            img_tokens: [B, T_img, D]
            svg_tokens: [B, T_svg, D]
            svg_mask: [B, T_svg] True for valid SVG tokens

        Returns:
            [B, T_img, D] fused image tokens
        """
        B, T_img, D = img_tokens.shape
        T_svg = svg_tokens.shape[1]

        # Normalize
        img = self.norm_img(img_tokens)
        svg = self.norm_svg(svg_tokens)

        # Project
        Q = self.q_proj(img).reshape(B, T_img, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(svg).reshape(B, T_svg, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(svg).reshape(B, T_svg, self.num_heads, self.head_dim).transpose(1, 2)
        # Q, K, V: [B, num_heads, T, head_dim]

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # scores: [B, num_heads, T_img, T_svg]

        # Apply mask if provided
        if svg_mask is not None:
            # svg_mask: [B, T_svg] → [B, 1, 1, T_svg]
            mask_expanded = svg_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask_expanded, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, V)  # [B, num_heads, T_img, head_dim]
        out = out.transpose(1, 2).reshape(B, T_img, D)
        out = self.out_proj(out)

        # Residual connection
        return img_tokens + out


class DualStreamEncoder(nn.Module):
    """Dual-stream encoder: ViT (image) + GraphTransformer (SVG) + cross-attention fusion."""

    def __init__(
        self,
        image_size: int = 512,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        vit_depth: int = 12,
        graph_depth: int = 6,
        num_heads: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Image encoder (ViT)
        self.image_encoder = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=vit_depth,
            num_heads=num_heads,
            dropout=dropout,
        )

        # SVG encoder (Graph Transformer)
        self.svg_encoder = GraphTransformer(
            node_dim=12,
            embed_dim=embed_dim,
            depth=graph_depth,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Cross-attention fusion
        self.fusion = CrossAttentionFusion(
            dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(
        self,
        image: torch.Tensor,
        stroke_tokens: torch.Tensor,
        stroke_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode image + SVG → fused representation.

        Args:
            image: [B, 3, 512, 512]
            stroke_tokens: [B, N, 12]
            stroke_mask: [B, N]

        Returns:
            img_tokens: [B, num_patches, embed_dim] fused image tokens
            svg_tokens: [B, N, embed_dim] SVG tokens (for reference)
        """
        # Encode both modalities
        img_tokens = self.image_encoder(image)  # [B, 1024, 768]
        svg_tokens = self.svg_encoder(stroke_tokens, stroke_mask)  # [B, N, 768]

        # Cross-attention: image queries SVG
        img_tokens_fused = self.fusion(img_tokens, svg_tokens, stroke_mask)

        return img_tokens_fused, svg_tokens


class ParallelBoxDecoder(nn.Module):
    """Parallel box decoder: decode all classes in single forward pass.

    Inspired by LocateAnything's Parallel Box Decoding (PBD).
    Output: [B, num_classes, 5] → (x, y, w, h, confidence)
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 35,
        num_patches: int = 1024,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Class queries (learnable)
        self.class_queries = nn.Parameter(torch.randn(1, num_classes, embed_dim))
        nn.init.trunc_normal_(self.class_queries, std=0.02)

        # Cross-attention: class queries attend to image tokens
        # num_heads must divide embed_dim
        num_heads = 12 if embed_dim % 12 == 0 else 8
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )

        # MLP heads
        self.box_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 4),  # (x, y, w, h)
        )

        self.confidence_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),  # confidence
        )

    def forward(self, encoder_output: torch.Tensor) -> dict[str, torch.Tensor]:
        """Parallel decoding.

        Args:
            encoder_output: [B, T, D] image tokens from encoder

        Returns:
            {
                'boxes': [B, num_classes, 4],  # (x, y, w, h) normalized [0, 1]
                'confidence': [B, num_classes],  # logits
            }
        """
        B, T, D = encoder_output.shape

        # Expand class queries
        queries = self.class_queries.expand(B, -1, -1)  # [B, num_classes, D]

        # Cross-attention: queries attend to encoder output
        attended, _ = self.cross_attn(
            query=queries,
            key=encoder_output,
            value=encoder_output,
        )  # [B, num_classes, D]

        # Decode boxes and confidence
        boxes = self.box_head(attended)  # [B, num_classes, 4]
        boxes = torch.sigmoid(boxes)  # Normalize to [0, 1]

        confidence = self.confidence_head(attended).squeeze(-1)  # [B, num_classes]

        return {
            'boxes': boxes,
            'confidence': confidence,
        }


class FloorPlanLLM(nn.Module):
    """Main model: FloorPlanLLM for structured understanding.

    Architecture:
    1. Dual-stream encoder (ViT + Graph Transformer)
    2. Cross-attention fusion
    3. Parallel box decoder

    Future extensions:
    - Autoregressive scene graph decoder
    - Relationship head
    - Hybrid inference mode
    """

    def __init__(
        self,
        image_size: int = 512,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        vit_depth: int = 12,
        graph_depth: int = 6,
        num_heads: int = 12,
        num_classes: int = 35,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_classes = num_classes

        # Encoder
        self.encoder = DualStreamEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            vit_depth=vit_depth,
            graph_depth=graph_depth,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Decoder
        num_patches = (image_size // patch_size) ** 2
        self.decoder = ParallelBoxDecoder(
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_patches=num_patches,
        )

    def forward(
        self,
        image: torch.Tensor,
        stroke_tokens: torch.Tensor,
        stroke_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            image: [B, 3, 512, 512]
            stroke_tokens: [B, N, 12]
            stroke_mask: [B, N]

        Returns:
            {
                'boxes': [B, num_classes, 4],
                'confidence': [B, num_classes],
            }
        """
        # Encode
        img_tokens, svg_tokens = self.encoder(image, stroke_tokens, stroke_mask)

        # Decode (parallel)
        output = self.decoder(img_tokens)

        return output

    def get_confidence_stats(self, output: dict[str, torch.Tensor]) -> dict[str, float]:
        """Compute confidence statistics for hybrid inference."""
        confidence = output['confidence']  # [B, num_classes]

        # Apply softmax to get probabilities
        probs = torch.softmax(confidence, dim=-1)

        # Max confidence
        max_conf = probs.max(dim=-1)[0].mean().item()

        # Top-5 spread (for spatial ambiguity detection)
        top5_vals = torch.topk(probs, k=min(5, probs.shape[-1]), dim=-1)[0]
        spread = (top5_vals.max(dim=-1)[0] - top5_vals.min(dim=-1)[0]).mean().item()

        return {
            'max_confidence': max_conf,
            'top5_spread': spread,
        }


def build_floorplan_llm(
    preset: str = 'base',
    num_classes: int = 35,
) -> FloorPlanLLM:
    """Factory function to build FloorPlanLLM.

    Presets:
    - 'tiny': Fast, low resource (6 layers ViT, 3 layers Graph, 384 dim)
    - 'small': Balanced (8 layers ViT, 4 layers Graph, 512 dim)
    - 'base': Default (12 layers ViT, 6 layers Graph, 768 dim)
    - 'large': High quality (16 layers ViT, 8 layers Graph, 1024 dim)
    """
    configs = {
        'tiny': {
            'embed_dim': 384,
            'vit_depth': 6,
            'graph_depth': 3,
            'num_heads': 6,
        },
        'small': {
            'embed_dim': 512,
            'vit_depth': 8,
            'graph_depth': 4,
            'num_heads': 8,
        },
        'base': {
            'embed_dim': 768,
            'vit_depth': 12,
            'graph_depth': 6,
            'num_heads': 12,
        },
        'large': {
            'embed_dim': 1024,
            'vit_depth': 16,
            'graph_depth': 8,
            'num_heads': 16,
        },
    }

    if preset not in configs:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(configs.keys())}")

    config = configs[preset]
    return FloorPlanLLM(
        num_classes=num_classes,
        **config,
    )
