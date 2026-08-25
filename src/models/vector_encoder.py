"""Vector (SVG stroke) encoder for the dual-pathway architecture.

Consumes the 12-dim stroke tokens produced by ``src/data/strokes.py`` and
encodes them with the same ``SelfAttention`` and ``FFN`` primitives used by
the image pathway (``object_learning_block.py``), treating each SVG primitive
as one token. A learned TypeEmbedding distinguishes lines from arcs; a
``[B, N]`` padding mask keeps padded positions out of attention. The 2-D
``GatedSpatialMixer2D`` is deliberately not used — strokes form a 1-D
sequence with no spatial grid.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import FFN, SelfAttention


class VectorEncoderBlock(nn.Module):
    """Pre-norm self-attention + FFN over 1-D stroke tokens."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        attention_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads, dropout, chunk_size=attention_chunk_size)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FFN(dim, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attention_mask=attention_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class VectorEncoder(nn.Module):
    """Transformer encoder over SVG primitive tokens."""

    def __init__(
        self,
        feature_dim: int = 12,
        model_dim: int = 256,
        depth: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        attention_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        if model_dim <= 0 or model_dim % num_heads != 0:
            raise ValueError("model_dim must be positive and divisible by num_heads")
        if depth < 0:
            raise ValueError("depth cannot be negative")
        self.feature_dim = feature_dim
        self.model_dim = model_dim
        self.input_projection = nn.Linear(feature_dim, model_dim)
        self.type_embedding = nn.Embedding(2, model_dim)
        nn.init.trunc_normal_(self.type_embedding.weight, std=0.02)
        self.blocks = nn.ModuleList(
            VectorEncoderBlock(
                dim=model_dim,
                num_heads=num_heads,
                dropout=dropout,
                attention_chunk_size=attention_chunk_size,
            )
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        stroke_tokens: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode ``[B, N, 12]`` tokens with ``[B, N]`` validity into ``[B, N, D]``.

        ``valid_mask`` uses ``True`` for real primitives (the ``[B, L]``
        padding-mask convention of :class:`SelfAttention`). Pad positions and
        all-pad rows return zeros so downstream fusion stays well-defined.
        """

        if stroke_tokens.ndim != 3:
            raise ValueError(
                f"stroke_tokens must have shape [B,N,{self.feature_dim}], got {tuple(stroke_tokens.shape)}"
            )
        if stroke_tokens.shape[-1] != self.feature_dim:
            raise ValueError(
                f"stroke_tokens must have {self.feature_dim} features, got {stroke_tokens.shape[-1]}"
            )
        batch, length, _ = stroke_tokens.shape
        if valid_mask.shape != (batch, length):
            raise ValueError(
                f"valid_mask must have shape [{batch},{length}], got {tuple(valid_mask.shape)}"
            )
        # Arc slots are zero for lines by construction (radius in dim 6).
        type_ids = (stroke_tokens[..., 6] != 0).to(dtype=torch.long)
        features = self.input_projection(stroke_tokens) + self.type_embedding(type_ids)
        for block in self.blocks:
            features = block(features, attention_mask=valid_mask)
        features = self.norm(features)
        return features * valid_mask.to(features.dtype).unsqueeze(-1)
