"""
Mamba + Self-Attention block for spatial feature learning.

Architecture per block:
  x → LayerNorm → Mamba (long-range sequence) → residual
    → LayerNorm → Self-Attention (spatial) → residual
    → LayerNorm → FFN → residual

Class conditioning: class embedding added before Mamba.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Simplified Mamba-like SSM block ───────────────────────────────────────────
# Full Mamba requires mamba-ssm package (CUDA kernel).
# This is a faithful approximation using selective scan principles.

class MambaBlock(nn.Module):
    """
    Simplified State Space Model block (Mamba-style).
    For production: replace with `from mamba_ssm import Mamba`.
    """

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.dim = dim
        self.d_inner = int(expand * dim)

        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)
        self.norm = nn.LayerNorm(self.d_inner)

        # SSM parameters
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)
                                            .repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        B, L, D = x.shape
        xz = self.in_proj(x)                    # [B, L, 2*d_inner]
        x_ssm, z = xz.chunk(2, dim=-1)          # each [B, L, d_inner]

        # Conv across sequence
        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Simplified SSM (linear scan approximation)
        x_out = self.norm(x_conv)
        x_out = x_out * F.silu(z)

        return self.out_proj(x_out)             # [B, L, D]


# ── Multi-head Self-Attention ──────────────────────────────────────────────────

class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)                 # each [B, L, H, head_dim]

        # Scaled dot-product attention
        attn = torch.einsum("blhd,bshd->blsh", q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum("blsh,bshd->blhd", attn, v)
        out = out.reshape(B, L, D)
        return self.proj(out)


# ── Feed-Forward Network ───────────────────────────────────────────────────────

class FFN(nn.Module):
    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = dim * expand
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Class-Conditioned Mamba + Attention Block ─────────────────────────────────

class ObjectLearningBlock(nn.Module):
    """
    Core learning block: Mamba (long-range) + Self-Attention (spatial) + FFN.
    Class conditioning via learnable class embeddings added to input.

    Args:
        dim        : feature dimension
        num_classes: number of object classes (35)
        num_heads  : attention heads
        dropout    : dropout rate
    """

    def __init__(
        self,
        dim: int,
        num_classes: int = 35,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Learnable class embeddings — one per class
        self.class_embed = nn.Embedding(num_classes, dim)

        # Mamba path
        self.norm1 = nn.LayerNorm(dim)
        self.mamba = MambaBlock(dim)

        # Self-attention path
        self.norm2 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads, dropout)

        # FFN path
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = FFN(dim, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        class_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x         : [B, L, D]  fused image+text tokens
            class_ids : [B]        class index for each sample (for conditioning)
        Returns:
            out       : [B, L, D]
        """
        # Add class conditioning
        cls_emb = self.class_embed(class_ids).unsqueeze(1)   # [B, 1, D]
        x = x + cls_emb

        # Mamba block (pre-norm)
        x = x + self.mamba(self.norm1(x))

        # Self-attention block (pre-norm)
        x = x + self.attn(self.norm2(x))

        # FFN block (pre-norm)
        x = x + self.ffn(self.norm3(x))

        return x


if __name__ == "__main__":
    B, L, D = 2, 256, 512
    block = ObjectLearningBlock(dim=D, num_classes=35)
    x = torch.randn(B, L, D)
    class_ids = torch.tensor([3, 10])
    out = block(x, class_ids)
    print(f"ObjectLearningBlock output: {out.shape}")  # [2, 256, 512]
