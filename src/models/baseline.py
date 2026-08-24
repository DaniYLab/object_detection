"""Project-native shared CenterNet baseline."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ObjectLearningBlock
from .config import ModelConfig, VAEConfig
from .detector import ConvImageEncoder, HeatmapHead, _group_count


class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
        )
        self.activation = nn.SiLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.activation(features + self.block(features))


class SharedCenterNetBaseline(nn.Module):
    """A compact shared-backbone, shared-head CenterNet reference model.

    The output contract matches :class:`FloorPlanDetector`: selected-class mode
    returns 1/2/2 channels and all-class mode returns C/2C/2C channels.
    """

    def __init__(
        self,
        image_size: int = 512,
        model_dim: int = 256,
        num_classes: int = 35,
        depth: int = 2,
        *,
        output_stride: int = 8,
        head_channels: int = 128,
        image_cfg: VAEConfig | None = None,
        center_prior: float = 0.01,
    ) -> None:
        super().__init__()
        if output_stride != 8:
            raise ValueError("SharedCenterNetBaseline currently supports output_stride=8 only")
        if image_size <= 0 or image_size % output_stride != 0:
            raise ValueError("image_size must be positive and divisible by output_stride")
        if num_classes <= 0 or model_dim <= 0 or depth < 0:
            raise ValueError("num_classes/model_dim must be positive and depth cannot be negative")
        self.image_size = image_size
        self.output_stride = output_stride
        self.num_classes = num_classes
        self.model_dim = model_dim
        self.image_cfg = image_cfg or VAEConfig()
        self.image_encoder = ConvImageEncoder(self.image_cfg)
        self.feature_projection = nn.Sequential(
            nn.Conv2d(self.image_cfg.latent_channels, model_dim, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(model_dim), model_dim),
            nn.SiLU(),
        )
        self.backbone = nn.Sequential(*(_ResidualConvBlock(model_dim) for _ in range(depth)))
        self.head = nn.Sequential(
            nn.Conv2d(model_dim, head_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(head_channels), head_channels),
            nn.SiLU(),
            nn.Conv2d(head_channels, num_classes * 5, kernel_size=1),
        )
        # P1-D: CenterNet-style negative prior bias on center channel.
        final_conv: nn.Conv2d = self.head[-1]  # type: ignore[assignment]
        if final_conv.bias is not None:
            with torch.no_grad():
                bias_val = math.log(center_prior / (1.0 - center_prior))
                for class_id in range(num_classes):
                    final_conv.bias[class_id * 5].fill_(bias_val)

    def _validate_image(self, image: torch.Tensor) -> None:
        if image.ndim != 4 or image.shape[0] <= 0:
            raise ValueError("image must have non-empty shape [B,C,H,W]")
        if image.shape[1] != self.image_cfg.in_channels:
            raise ValueError(f"expected {self.image_cfg.in_channels} image channels")
        if image.shape[-2] % self.output_stride or image.shape[-1] % self.output_stride:
            raise ValueError("image height and width must be divisible by output_stride=8")

    def _decode(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, _, height, width = raw.shape
        raw = raw.reshape(batch, self.num_classes, 5, height, width)
        center_logits = raw[:, :, 0]
        heatmap = torch.sigmoid(center_logits)
        sizes = F.softplus(raw[:, :, 1:3])
        offsets = torch.sigmoid(raw[:, :, 3:5])
        return heatmap, center_logits, sizes, offsets

    def forward(
        self,
        image: torch.Tensor,
        class_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        self._validate_image(image)
        features = self.backbone(self.feature_projection(self.image_encoder(image)))
        expected = (image.shape[-2] // self.output_stride, image.shape[-1] // self.output_stride)
        if features.shape[-2:] != expected:
            raise RuntimeError(
                f"image encoder violated stride-8 contract: expected {expected}, got {features.shape[-2:]}"
            )
        heatmap, center_logits, sizes, offsets = self._decode(self.head(features))

        if class_ids is None:
            batch, _, _, height, width = sizes.shape
            return {
                "center_heatmap": heatmap,
                "center_logits": center_logits,
                "size_map": sizes.reshape(batch, self.num_classes * 2, height, width),
                "offset_map": offsets.reshape(batch, self.num_classes * 2, height, width),
            }

        ids = torch.as_tensor(class_ids, device=image.device, dtype=torch.long)
        if ids.ndim != 1 or ids.shape[0] != image.shape[0]:
            raise ValueError(f"class_ids must have shape [{image.shape[0]}]")
        if torch.any(ids < 0) or torch.any(ids >= self.num_classes):
            raise ValueError(f"class_ids must be in [0, {self.num_classes})")
        batch_indices = torch.arange(image.shape[0], device=image.device)
        return {
            "center_heatmap": heatmap[batch_indices, ids].unsqueeze(1),
            "center_logits": center_logits[batch_indices, ids].unsqueeze(1),
            "size_map": sizes[batch_indices, ids],
            "offset_map": offsets[batch_indices, ids],
        }


class SharedPathwayCenterNet(nn.Module):
    """Unconditioned multi-class control with the FloorPlan pathway architecture.

    This model shares the image encoder, learned 2-D positional embedding,
    GatedSpatialMixer/SelfAttention/FFN blocks, normalization, and CenterNet head
    design with :class:`FloorPlanDetector`, but has no class/text conditioner or
    fusion. A 5C head gives each class a distinct output, avoiding the degenerate
    class-agnostic one-channel query control.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.architecture != "floorplan_unconditioned":
            raise ValueError(
                "SharedPathwayCenterNet requires architecture='floorplan_unconditioned'"
            )
        if config.pathway_mode != "shared":
            raise ValueError("SharedPathwayCenterNet requires pathway_mode='shared'")
        if config.conditioner.kind != "none" or config.fusion_mode != "none":
            raise ValueError("SharedPathwayCenterNet cannot use conditioning or fusion")

        self.config = ModelConfig.from_dict(config.to_dict())
        self.image_size = config.image_size
        self.output_stride = config.output_stride
        self.num_classes = config.num_classes
        self.model_dim = config.model_dim
        self.image_cfg = config.vae
        self.latent_size = config.latent_size

        self.image_encoder = ConvImageEncoder(config.vae)
        self.image_projection = nn.Linear(config.vae.latent_channels, config.model_dim)
        self.image_pos_embed = nn.Parameter(
            torch.zeros(1, self.latent_size * self.latent_size, config.model_dim)
        )
        nn.init.trunc_normal_(self.image_pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            ObjectLearningBlock(
                dim=config.model_dim,
                num_classes=None,
                num_heads=config.num_heads,
                dropout=config.dropout,
                use_class_embedding=False,
                attention_chunk_size=config.attention_chunk_size,
            )
            for _ in range(config.depth_per_class)
        )
        self.out_norm = nn.LayerNorm(config.model_dim)
        self.head = HeatmapHead(
            config.model_dim,
            out_channels=config.num_classes * 5,
            hidden_dim=config.head_channels,
        )
        final_conv: nn.Conv2d = self.head.head[-1]  # type: ignore[assignment]
        if final_conv.bias is not None:
            with torch.no_grad():
                prior_bias = math.log(0.01 / 0.99)
                final_conv.bias[0::5].fill_(prior_bias)

    def _validate_image(self, image: torch.Tensor) -> None:
        if image.ndim != 4 or image.shape[0] <= 0:
            raise ValueError("image must have non-empty shape [B,C,H,W]")
        if image.shape[1] != self.image_cfg.in_channels:
            raise ValueError(f"expected {self.image_cfg.in_channels} image channels")
        if image.shape[-2] % self.output_stride or image.shape[-1] % self.output_stride:
            raise ValueError("image height and width must be divisible by output_stride")

    def _encode(self, image: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        self._validate_image(image)
        latent = self.image_encoder(image)
        batch, _, height, width = latent.shape
        expected = (
            image.shape[-2] // self.output_stride,
            image.shape[-1] // self.output_stride,
        )
        if (height, width) != expected:
            raise RuntimeError(
                f"image encoder violated stride-{self.output_stride} contract: "
                f"expected {expected}, got {(height, width)}"
            )
        tokens = self.image_projection(latent.flatten(2).transpose(1, 2))
        if (height, width) == (self.latent_size, self.latent_size):
            position = self.image_pos_embed
        else:
            position = self.image_pos_embed.transpose(1, 2).reshape(
                1,
                self.model_dim,
                self.latent_size,
                self.latent_size,
            )
            position = F.interpolate(
                position,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).flatten(2).transpose(1, 2)
        return tokens + position, height, width

    def _decode(
        self, raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, _, height, width = raw.shape
        raw = raw.reshape(batch, self.num_classes, 5, height, width)
        center_logits = raw[:, :, 0]
        heatmap = torch.sigmoid(center_logits)
        sizes = F.softplus(raw[:, :, 1:3])
        offsets = torch.sigmoid(raw[:, :, 3:5])
        return heatmap, center_logits, sizes, offsets

    def forward(
        self,
        image: torch.Tensor,
        class_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        tokens, height, width = self._encode(image)
        for block in self.blocks:
            tokens = block(tokens, spatial_shape=(height, width))
        features = self.out_norm(tokens).transpose(1, 2).reshape(
            image.shape[0], self.model_dim, height, width
        )
        heatmap, center_logits, sizes, offsets = self._decode(self.head(features))

        if class_ids is None:
            return {
                "center_heatmap": heatmap,
                "center_logits": center_logits,
                "size_map": sizes.reshape(image.shape[0], self.num_classes * 2, height, width),
                "offset_map": offsets.reshape(
                    image.shape[0], self.num_classes * 2, height, width
                ),
            }

        ids = torch.as_tensor(class_ids, device=image.device, dtype=torch.long)
        if ids.ndim != 1 or ids.shape[0] != image.shape[0]:
            raise ValueError(f"class_ids must have shape [{image.shape[0]}]")
        if torch.any(ids < 0) or torch.any(ids >= self.num_classes):
            raise ValueError(f"class_ids must be in [0, {self.num_classes})")
        batch_indices = torch.arange(image.shape[0], device=image.device)
        return {
            "center_heatmap": heatmap[batch_indices, ids].unsqueeze(1),
            "center_logits": center_logits[batch_indices, ids].unsqueeze(1),
            "size_map": sizes[batch_indices, ids],
            "offset_map": offsets[batch_indices, ids],
        }


CenterNetBaseline = SharedCenterNetBaseline
