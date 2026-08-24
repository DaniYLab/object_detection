"""Spatial mixing, attention, and object-learning blocks."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedSpatialMixer2D(nn.Module):
    """Gated depthwise 2D convolutional mixer for flattened spatial tokens.

    This is deliberately named for what it implements. It is not a Mamba or
    state-space model. Every learned parameter participates in the forward path.
    """

    def __init__(
        self,
        dim: int,
        *,
        expand: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim <= 0 or expand <= 0:
            raise ValueError("dim and expand must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        inner_dim = dim * expand
        self.dim = dim
        self.inner_dim = inner_dim
        self.in_proj = nn.Linear(dim, inner_dim * 2, bias=False)
        self.depthwise = nn.Conv2d(
            inner_dim,
            inner_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=inner_dim,
        )
        self.norm = nn.LayerNorm(inner_dim)
        self.out_proj = nn.Linear(inner_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _resolve_spatial_shape(length: int, spatial_shape: tuple[int, int] | None) -> tuple[int, int]:
        if spatial_shape is None:
            side = math.isqrt(length)
            if side * side != length:
                raise ValueError(
                    "spatial_shape is required when token length is not a perfect square"
                )
            return side, side
        height, width = spatial_shape
        if height <= 0 or width <= 0 or height * width != length:
            raise ValueError(
                f"spatial_shape {spatial_shape} does not match token length {length}"
            )
        return height, width

    def forward(
        self,
        x: torch.Tensor,
        spatial_shape: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must have shape [B,L,D], got {tuple(x.shape)}")
        batch, length, _ = x.shape
        height, width = self._resolve_spatial_shape(length, spatial_shape)
        content, gate = self.in_proj(x).chunk(2, dim=-1)
        content_2d = content.transpose(1, 2).reshape(batch, self.inner_dim, height, width)
        content_2d = self.depthwise(content_2d)
        content = content_2d.flatten(2).transpose(1, 2)
        mixed = self.norm(content) * F.silu(gate)
        return self.dropout(self.out_proj(mixed))


# Concise public name used by architecture configs and ablation descriptions.
GatedSpatialMixer = GatedSpatialMixer2D


class SelfAttention(nn.Module):
    """Multi-head self-attention using canonical ``[B,H,L,D]`` SDPA layout.

    ``attention_mask`` uses ``True`` for positions that may be attended to. A
    ``[B,L]`` mask is interpreted as a valid-key padding mask. Supplying
    ``chunk_size`` forces the memory-bounded fallback; otherwise PyTorch SDPA is
    used and the same chunked implementation is used if SDPA is unavailable.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        if dim <= 0 or num_heads <= 0 or dim % num_heads != 0:
            raise ValueError("dim must be positive and divisible by num_heads")
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("chunk_size must be positive when provided")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.chunk_size = chunk_size
        self.fallback_chunk_size = chunk_size or 256

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _normalise_mask(
        attention_mask: torch.Tensor | None,
        *,
        batch: int,
        heads: int,
        query_length: int,
        key_length: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        mask = attention_mask.to(device=device)
        if mask.ndim == 2:
            if mask.shape == (batch, key_length):
                # Tokenizer-style [B,L] masks use 1/True for valid keys.
                if mask.dtype != torch.bool:
                    mask = mask != 0
                mask = mask[:, None, None, :]
            elif mask.shape == (query_length, key_length):
                mask = mask[None, None, :, :]
            else:
                raise ValueError(
                    "2D attention_mask must have shape [B,L] or [Q,K], got "
                    f"{tuple(mask.shape)}"
                )
        elif mask.ndim == 3:
            if mask.shape != (batch, query_length, key_length):
                raise ValueError(
                    "3D attention_mask must have shape [B,Q,K], got "
                    f"{tuple(mask.shape)}"
                )
            mask = mask[:, None, :, :]
        elif mask.ndim == 4:
            if mask.shape[0] not in {1, batch} or mask.shape[1] not in {1, heads}:
                raise ValueError(f"invalid 4D attention_mask shape: {tuple(mask.shape)}")
            if mask.shape[2] not in {1, query_length} or mask.shape[3] != key_length:
                raise ValueError(f"invalid 4D attention_mask shape: {tuple(mask.shape)}")
        else:
            raise ValueError("attention_mask must have 2, 3, or 4 dimensions")
        return mask

    @staticmethod
    def _combine_causal_mask(
        mask: torch.Tensor | None,
        *,
        query_length: int,
        key_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        causal = torch.arange(key_length, device=device).view(1, 1, 1, key_length) <= torch.arange(
            query_length, device=device
        ).view(1, 1, query_length, 1)
        if mask is None:
            return causal
        if mask.dtype == torch.bool:
            return mask & causal
        causal_bias = torch.zeros(
            1, 1, query_length, key_length, device=device, dtype=mask.dtype
        ).masked_fill(~causal, float("-inf"))
        return mask + causal_bias

    def _chunked_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        is_causal: bool,
    ) -> torch.Tensor:
        batch, heads, query_length, _ = q.shape
        key_length = k.shape[-2]
        outputs = []
        chunk_size = min(self.fallback_chunk_size, query_length)
        key_positions = torch.arange(key_length, device=q.device).view(1, 1, 1, key_length)

        for start in range(0, query_length, chunk_size):
            end = min(start + chunk_size, query_length)
            scores = torch.matmul(q[:, :, start:end], k.transpose(-2, -1)) * self.scale
            chunk_mask = None
            if attention_mask is not None:
                chunk_mask = attention_mask
                if attention_mask.shape[-2] != 1:
                    chunk_mask = attention_mask[..., start:end, :]
            if is_causal:
                query_positions = torch.arange(start, end, device=q.device).view(1, 1, -1, 1)
                causal = key_positions <= query_positions
                if chunk_mask is None:
                    chunk_mask = causal
                elif chunk_mask.dtype == torch.bool:
                    chunk_mask = chunk_mask & causal
                else:
                    causal_bias = torch.zeros_like(scores).masked_fill(~causal, float("-inf"))
                    chunk_mask = chunk_mask + causal_bias
            if chunk_mask is not None:
                if chunk_mask.dtype == torch.bool:
                    scores = scores.masked_fill(~chunk_mask, float("-inf"))
                else:
                    scores = scores + chunk_mask

            weights = torch.softmax(scores, dim=-1)
            # An entirely masked query has no valid distribution and should
            # contribute zero rather than NaN.
            weights = torch.nan_to_num(weights, nan=0.0)
            weights = F.dropout(weights, p=self.dropout.p, training=self.training)
            outputs.append(torch.matmul(weights, v))
        return torch.cat(outputs, dim=-2).reshape(batch, heads, query_length, self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must have shape [B,L,D], got {tuple(x.shape)}")
        batch, length, dim = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.num_heads, self.head_dim)
        # [B,L,3,H,D] -> three tensors in canonical [B,H,L,D] layout.
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        mask = self._normalise_mask(
            attention_mask,
            batch=batch,
            heads=self.num_heads,
            query_length=length,
            key_length=length,
            device=x.device,
        )

        if self.chunk_size is not None:
            out = self._chunked_attention(q, k, v, mask, is_causal=is_causal)
        else:
            sdpa_mask = mask
            sdpa_is_causal = is_causal
            if is_causal and mask is not None:
                sdpa_mask = self._combine_causal_mask(
                    mask,
                    query_length=length,
                    key_length=length,
                    device=x.device,
                )
                sdpa_is_causal = False
            sdpa = getattr(F, "scaled_dot_product_attention", None)
            if sdpa is None:
                out = self._chunked_attention(q, k, v, mask, is_causal=is_causal)
            else:
                try:
                    out = sdpa(
                        q,
                        k,
                        v,
                        attn_mask=sdpa_mask,
                        dropout_p=self.dropout.p if self.training else 0.0,
                        is_causal=sdpa_is_causal,
                    )
                except (RuntimeError, NotImplementedError):
                    out = self._chunked_attention(q, k, v, mask, is_causal=is_causal)

        out = out.transpose(1, 2).contiguous().reshape(batch, length, dim)
        return self.proj(out)


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


class ObjectLearningBlock(nn.Module):
    """Pre-norm spatial mixer, self-attention, and feed-forward block."""

    def __init__(
        self,
        dim: int,
        num_classes: int | None = 35,
        num_heads: int = 8,
        dropout: float = 0.1,
        *,
        use_class_embedding: bool = True,
        attention_chunk_size: int | None = None,
        mixer_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if use_class_embedding and (num_classes is None or num_classes <= 0):
            raise ValueError("positive num_classes is required when class embedding is enabled")
        self.class_embed = (
            nn.Embedding(num_classes, dim) if use_class_embedding and num_classes is not None else None
        )
        self.norm1 = nn.LayerNorm(dim)
        self.spatial_mixer = GatedSpatialMixer2D(
            dim,
            kernel_size=mixer_kernel_size,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads, dropout, chunk_size=attention_chunk_size)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = FFN(dim, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        class_ids: torch.Tensor | None = None,
        *,
        spatial_shape: tuple[int, int] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.class_embed is not None:
            if class_ids is None:
                raise ValueError("class_ids are required when class embedding is enabled")
            if class_ids.ndim != 1 or class_ids.shape[0] != x.shape[0]:
                raise ValueError("class_ids must have shape [B]")
            x = x + self.class_embed(class_ids).unsqueeze(1)
        x = x + self.spatial_mixer(self.norm1(x), spatial_shape=spatial_shape)
        x = x + self.attn(self.norm2(x), attention_mask=attention_mask)
        x = x + self.ffn(self.norm3(x))
        return x
