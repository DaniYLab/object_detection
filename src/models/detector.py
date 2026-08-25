"""Conditioned stride-8 floor-plan detector and reusable CenterNet heads."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ObjectLearningBlock
from .conditioning import ConditioningOutput, build_conditioner, masked_mean
from .config import ConditionerConfig, ModelConfig, TextEncoderConfig, VAEConfig, VectorBranchConfig
from .vector_encoder import VectorEncoder

# Alphabetical FloorPlanCAD names mirror src.data.constants without importing the
# data package (whose optional image dependencies should not be required by models).
_PROJECT_CLASS_NAMES = [
    "annotation_text",
    "bathtub",
    "bed",
    "cabinet",
    "chair",
    "column",
    "counter",
    "dimension_line",
    "door_double",
    "door_revolving",
    "door_single",
    "door_sliding",
    "elevator",
    "escalator",
    "escalator_stair",
    "floor_plan_area",
    "oven",
    "parking",
    "plant",
    "ramp",
    "refrigerator",
    "room_label",
    "shower",
    "sink",
    "sofa",
    "stair",
    "symbol_misc",
    "table",
    "toilet",
    "tv",
    "wall",
    "washing_machine",
    "window",
    "window_bay",
    "window_blind",
]
_PROJECT_TEXT_TEMPLATE = "Find {cls} in this floor plan drawing"


def _default_class_texts(num_classes: int) -> list[str]:
    names = list(_PROJECT_CLASS_NAMES[:num_classes])
    names.extend(f"class_{index}" for index in range(len(names), num_classes))
    return [_PROJECT_TEXT_TEMPLATE.format(cls=name) for name in names]


CLASS_TEXTS = _default_class_texts(max(35, len(_PROJECT_CLASS_NAMES)))


def _group_count(channels: int, max_groups: int = 32) -> int:
    """Choose a GroupNorm count with at least two channels per group."""
    upper = min(max_groups, max(1, channels // 2))
    for groups in range(upper, 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvImageEncoder(nn.Module):
    """Project-native convolutional image encoder with a fixed stride of eight."""

    output_stride = 8

    def __init__(self, cfg: VAEConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or VAEConfig()
        if cfg.downsample_factor != self.output_stride:
            raise ValueError("ConvImageEncoder supports downsample_factor=8 only")
        layers: list[nn.Module] = []
        in_channels = cfg.in_channels
        for index, out_channels in enumerate(cfg.block_out_channels):
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2 if index < 3 else 1,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(_group_count(out_channels, cfg.norm_num_groups), out_channels),
                    nn.SiLU(),
                ]
            )
            in_channels = out_channels
        layers.append(nn.Conv2d(in_channels, cfg.latent_channels, kernel_size=1))
        self.encoder = nn.Sequential(*layers)
        self.config = cfg

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image)


# Historical name retained for compatibility; the implementation is explicitly
# a convolutional encoder and does not claim to be a pretrained VAE.
VAEEncoderStub = ConvImageEncoder


class TextEncoderStub(nn.Module):
    """Compatibility token-ID encoder without the former multi-gigabyte stub."""

    def __init__(self, cfg: TextEncoderConfig | None = None, model_dim: int = 256) -> None:
        super().__init__()
        cfg = cfg or TextEncoderConfig()
        hidden_dim = cfg.embedding_dim
        self.max_length = cfg.max_length
        self.embedding = nn.Embedding(cfg.vocab_size, hidden_dim, padding_idx=0)
        self.position = nn.Embedding(cfg.max_length, hidden_dim)
        self.projection = nn.Linear(hidden_dim, model_dim)
        self.norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[1] > self.max_length:
            raise ValueError("input_ids must have shape [B,L] with L <= max_length")
        if attention_mask is None:
            attention_mask = input_ids.ne(0)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        hidden = self.embedding(input_ids) + self.position(positions)
        tokens = self.norm(self.projection(hidden))
        return tokens * attention_mask.unsqueeze(-1).to(tokens.dtype)


def _tokenize_fixed(texts: list[str], max_len: int = 32, vocab_size: int = 32000) -> torch.Tensor:
    """Stable compatibility tokenizer for callers using ``TextEncoderStub``."""
    tokens = []
    for text in texts:
        ids = []
        for word in text.lower().split():
            digest = hashlib.md5(word.encode("utf-8")).hexdigest()
            ids.append(int(digest, 16) % (vocab_size - 1) + 1)
        ids = ids[:max_len] + [0] * max(0, max_len - len(ids))
        tokens.append(ids)
    return torch.tensor(tokens, dtype=torch.long)


class EarlyFusion(nn.Module):
    """Fuse image tokens with pooled and token-level conditioning.

    Supported modes are ``none``, ``add``, ``film``, ``cross_attention``,
    ``film_cross_attention``, and the legacy ``current`` mode. Legacy short
    spellings such as ``film_cross_attn`` remain accepted.
    """

    _ALIASES = {
        "identity": "none",
        "additive": "add",
        "cross_attn": "cross_attention",
        "film_cross_attn": "film_cross_attention",
    }

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mode: str = "film",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        mode = self._ALIASES.get(mode, mode)
        valid_modes = {"none", "add", "film", "cross_attention", "film_cross_attention", "current"}
        if mode not in valid_modes:
            raise ValueError(f"unknown fusion mode '{mode}'. Expected one of {sorted(valid_modes)}")
        if dim % num_heads != 0:
            raise ValueError("fusion dim must be divisible by num_heads")
        self.mode = mode
        self.dim = dim

        if mode in {"add"}:
            self.condition_projection = nn.Linear(dim, dim)
        if mode in {"film", "film_cross_attention"}:
            self.film = nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.SiLU(),
                nn.Linear(dim * 2, dim * 2),
            )
        if mode in {"cross_attention", "film_cross_attention", "current"}:
            self.image_norm = nn.LayerNorm(dim)
            self.condition_norm = nn.LayerNorm(dim)
            self.cross_attention = nn.MultiheadAttention(
                dim,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.attention_projection = nn.Linear(dim, dim)
        if mode != "none":
            self.output_norm = nn.LayerNorm(dim)

    @staticmethod
    def _validate_conditioning(
        image_tokens: torch.Tensor,
        conditioning: ConditioningOutput,
    ) -> None:
        if conditioning.tokens.ndim != 3 or conditioning.pooled.ndim != 2:
            raise ValueError("conditioning tokens/pooled tensors must have shapes [B,L,D] and [B,D]")
        if conditioning.tokens.shape[0] != image_tokens.shape[0]:
            raise ValueError("conditioning and image batch sizes must match")
        if conditioning.tokens.shape[-1] != image_tokens.shape[-1]:
            raise ValueError("conditioning and image dimensions must match")
        if conditioning.attention_mask.shape != conditioning.tokens.shape[:2]:
            raise ValueError("conditioning attention_mask must have shape [B,L]")

    @staticmethod
    def _safe_condition_mask(
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid_rows = attention_mask.any(dim=1)
        safe_mask = attention_mask.clone()
        safe_tokens = tokens
        if not torch.all(valid_rows):
            safe_mask[~valid_rows, 0] = True
            safe_tokens = tokens.clone()
            safe_tokens[~valid_rows, 0] = 0
        return safe_tokens, safe_mask, valid_rows

    def forward(
        self,
        image_tokens: torch.Tensor,
        conditioning: ConditioningOutput | torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Tensor input preserves the former EarlyFusion(img, txt_tokens) API.
        if isinstance(conditioning, torch.Tensor):
            if attention_mask is None:
                attention_mask = torch.ones(
                    conditioning.shape[:2], device=conditioning.device, dtype=torch.bool
                )
            conditioning = ConditioningOutput(
                tokens=conditioning,
                attention_mask=attention_mask.to(dtype=torch.bool),
                pooled=masked_mean(conditioning, attention_mask.to(dtype=torch.bool)),
            )
        self._validate_conditioning(image_tokens, conditioning)
        if self.mode == "none":
            return image_tokens

        valid_rows = conditioning.attention_mask.any(dim=1).to(image_tokens.dtype)
        if self.mode == "add":
            delta = self.condition_projection(conditioning.pooled) * valid_rows.unsqueeze(-1)
            fused = image_tokens + delta.unsqueeze(1)
            return self.output_norm(fused)

        if self.mode in {"film", "film_cross_attention"}:
            gamma, beta = self.film(conditioning.pooled).chunk(2, dim=-1)
            gamma = gamma * valid_rows.unsqueeze(-1)
            beta = beta * valid_rows.unsqueeze(-1)
            fused = image_tokens * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        else:
            fused = image_tokens

        if self.mode in {"cross_attention", "film_cross_attention"}:
            condition_tokens, safe_mask, valid_rows = self._safe_condition_mask(
                self.condition_norm(conditioning.tokens),
                conditioning.attention_mask,
            )
            attended, _ = self.cross_attention(
                query=self.image_norm(fused),
                key=condition_tokens,
                value=condition_tokens,
                key_padding_mask=~safe_mask,
                need_weights=False,
            )
            attended = attended * valid_rows[:, None, None].to(attended.dtype)
            fused = fused + self.attention_projection(attended)
        elif self.mode == "current":
            attended, _ = self.cross_attention(
                query=self.condition_norm(conditioning.tokens),
                key=self.image_norm(image_tokens),
                value=self.image_norm(image_tokens),
                need_weights=False,
            )
            summary = masked_mean(attended, conditioning.attention_mask)
            fused = image_tokens + self.attention_projection(summary).unsqueeze(1)

        return self.output_norm(fused)


class HeatmapHead(nn.Module):
    """GroupNorm/SiLU class-conditioned CenterNet head producing five channels.

    The final Conv2d has no activation; raw center logits are returned alongside
    the sigmoid probability so the loss can use the numerically stable
    ``logsigmoid`` path instead of ``log(p.clamp(...))``.
    """

    def __init__(
        self,
        in_dim: int,
        out_channels: int = 5,
        hidden_dim: int = 128,
        center_prior: float = 0.01,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
        )
        # P1-D: CenterNet-style negative prior bias. Initialise the center
        # channel bias so sigmoid(bias) ≈ center_prior to avoid large focal
        # losses at the start of training with a large 64×64 heatmap.
        import math as _math
        final_conv: nn.Conv2d = self.head[-1]  # type: ignore[assignment]
        if final_conv.bias is not None:
            with torch.no_grad():
                bias_val = _math.log(center_prior / (1.0 - center_prior))
                final_conv.bias[0].fill_(bias_val)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)


class FloorPlanDetector(nn.Module):
    """Conditioned CenterNet detector with shared or per-class pathways.

    ``class_ids`` selects one output per image and is routed in class groups.
    Omitting it computes all classes in bounded class chunks. Runtime ``texts``
    override the built-in class prompts; fixed prompts are the fallback.
    """

    def __init__(
        self,
        image_size: int | ModelConfig = 512,
        model_dim: int = 256,
        num_classes: int = 35,
        depth_per_class: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        fusion_mode: str = "film",
        vae_cfg: VAEConfig | None = None,
        text_cfg: TextEncoderConfig | None = None,
        *,
        output_stride: int = 8,
        pathway_mode: str = "shared",
        conditioner: str | ConditionerConfig | nn.Module | None = None,
        conditioner_cfg: ConditionerConfig | None = None,
        class_texts: Sequence[str] | None = None,
        class_chunk_size: int = 8,
        attention_chunk_size: int | None = None,
        head_channels: int = 128,
        config: ModelConfig | None = None,
    ) -> None:
        super().__init__()
        if isinstance(image_size, ModelConfig):
            if config is not None:
                raise ValueError("pass ModelConfig either positionally or with config=, not both")
            config = image_size

        custom_conditioner = conditioner if isinstance(conditioner, nn.Module) else None
        if conditioner_cfg is not None and conditioner is not None:
            raise ValueError("pass conditioner or conditioner_cfg, not both")
        if config is None:
            resolved_text_config = text_cfg or TextEncoderConfig()
            if conditioner_cfg is not None:
                cond_config = conditioner_cfg
            elif isinstance(conditioner, ConditionerConfig):
                cond_config = conditioner
            elif isinstance(conditioner, str):
                cond_config = ConditionerConfig(
                    kind=conditioner,
                    max_length=resolved_text_config.max_length,
                    embedding_dim=resolved_text_config.embedding_dim,
                )
            else:
                cond_config = ConditionerConfig(kind="class_embedding")
            config = ModelConfig(
                image_size=int(image_size),
                output_stride=output_stride,
                model_dim=model_dim,
                num_classes=num_classes,
                class_texts=tuple(class_texts) if class_texts is not None else None,
                depth_per_class=depth_per_class,
                num_heads=num_heads,
                dropout=dropout,
                fusion_mode=fusion_mode,
                pathway_mode=pathway_mode,
                class_chunk_size=class_chunk_size,
                attention_chunk_size=attention_chunk_size,
                head_channels=head_channels,
                vae=vae_cfg or VAEConfig(),
                text_encoder=resolved_text_config,
                conditioner=cond_config,
            )
        else:
            # Round-trip to detach mutable caller-owned nested configs and validate.
            config = ModelConfig.from_dict(config.to_dict())
        if conditioner_cfg is not None:
            config = config.with_overrides(conditioner=conditioner_cfg)
        elif isinstance(conditioner, ConditionerConfig):
            config = config.with_overrides(conditioner=conditioner)
        elif isinstance(conditioner, str):
            config = config.with_overrides(
                conditioner=ConditionerConfig(
                    kind=conditioner,
                    max_length=config.text_encoder.max_length,
                    embedding_dim=config.text_encoder.embedding_dim,
                )
            )
        if class_texts is not None and tuple(class_texts) != config.class_texts:
            config = config.with_overrides(class_texts=tuple(class_texts))

        if config.architecture != "floorplan_detector":
            raise ValueError(
                "FloorPlanDetector requires architecture='floorplan_detector'; use build_model() "
                "for architecture-dispatched construction"
            )
        fixed_texts = (
            tuple(config.class_texts)
            if config.class_texts is not None
            else tuple(_default_class_texts(config.num_classes))
        )
        if len(fixed_texts) != config.num_classes or not all(
            isinstance(text, str) for text in fixed_texts
        ):
            raise ValueError(f"class_texts must contain exactly {config.num_classes} strings")
        if config.class_texts is None:
            config = config.with_overrides(class_texts=fixed_texts)

        self.config = config
        self.image_size = config.image_size
        self.output_stride = config.output_stride
        self.latent_size = config.latent_size
        self.model_dim = config.model_dim
        self.num_classes = config.num_classes
        self.depth_per_class = config.depth_per_class
        self.fusion_mode = EarlyFusion._ALIASES.get(config.fusion_mode, config.fusion_mode)
        self.pathway_mode = config.pathway_mode
        self.class_chunk_size = config.class_chunk_size
        self.class_texts = fixed_texts

        self.image_encoder = ConvImageEncoder(config.vae)
        self.image_projection = nn.Linear(config.vae.latent_channels, self.model_dim)
        self.image_pos_embed = nn.Parameter(
            torch.zeros(1, self.latent_size * self.latent_size, self.model_dim)
        )
        nn.init.trunc_normal_(self.image_pos_embed, std=0.02)

        self.conditioner = custom_conditioner or build_conditioner(
            config.conditioner,
            model_dim=self.model_dim,
            num_classes=self.num_classes,
        )

        block_kwargs = {
            "dim": self.model_dim,
            "num_classes": None,
            "num_heads": config.num_heads,
            "dropout": config.dropout,
            "use_class_embedding": False,
            "attention_chunk_size": config.attention_chunk_size,
        }
        self.shared_fusion: EarlyFusion | None = None
        self.shared_blocks = nn.ModuleList()
        self.early_fusions = nn.ModuleList()
        self.class_blocks = nn.ModuleList()
        if self.pathway_mode == "shared":
            self.shared_fusion = EarlyFusion(
                self.model_dim,
                config.num_heads,
                self.fusion_mode,
                config.dropout,
            )
            self.shared_blocks = nn.ModuleList(
                ObjectLearningBlock(**block_kwargs) for _ in range(self.depth_per_class)
            )
        else:
            self.early_fusions = nn.ModuleList(
                EarlyFusion(
                    self.model_dim,
                    config.num_heads,
                    self.fusion_mode,
                    config.dropout,
                )
                for _ in range(self.num_classes)
            )
            self.class_blocks = nn.ModuleList(
                nn.ModuleList(
                    ObjectLearningBlock(**block_kwargs) for _ in range(self.depth_per_class)
                )
                for _ in range(self.num_classes)
            )

        self.out_norm = nn.LayerNorm(self.model_dim)
        self.heatmap_head = HeatmapHead(
            self.model_dim,
            out_channels=5,
            hidden_dim=config.head_channels,
        )

        # Dual-pathway vector (SVG stroke) branch. Cross-attention fusion uses
        # image tokens as Query and encoded strokes as Key/Value so the stride-8
        # spatial layout is preserved; "add" pools strokes per image and adds.
        self.vector_enabled = bool(config.vector.enabled)
        self.vector_n_max = int(config.vector.n_max)
        self.vector_fusion_mode = config.vector.fusion
        self.vector_encoder: VectorEncoder | None = None
        self.vector_cross_attention = None
        if self.vector_enabled:
            self.vector_encoder = VectorEncoder(
                feature_dim=config.vector.feature_dim,
                model_dim=self.model_dim,
                depth=config.vector.depth,
                num_heads=config.vector.num_heads,
                dropout=config.dropout,
                attention_chunk_size=config.attention_chunk_size,
            )
            if self.vector_fusion_mode == "cross_attention":
                self.vector_norm = nn.LayerNorm(self.model_dim)
                self.vector_cross_attention = nn.MultiheadAttention(
                    self.model_dim,
                    config.vector.num_heads,
                    dropout=config.dropout,
                    batch_first=True,
                )
                self.vector_projection = nn.Linear(self.model_dim, self.model_dim)
            else:
                self.vector_projection = nn.Linear(self.model_dim, self.model_dim)

    @property
    def vae_encoder(self) -> nn.Module:
        """Compatibility view of the project-native image encoder."""
        return self.image_encoder

    @property
    def img_proj(self) -> nn.Module:
        """Compatibility view of the image token projection."""
        return self.image_projection

    @property
    def text_encoder(self) -> nn.Module:
        """Compatibility view of the active conditioner."""
        return self.conditioner

    @classmethod
    def from_config(cls, config: ModelConfig, **kwargs) -> FloorPlanDetector:
        return cls(config=config, **kwargs)

    def _validate_image(self, image: torch.Tensor) -> None:
        if image.ndim != 4:
            raise ValueError(f"image must have shape [B,C,H,W], got {tuple(image.shape)}")
        if image.shape[0] <= 0:
            raise ValueError("image batch must be non-empty")
        if image.shape[1] != self.config.vae.in_channels:
            raise ValueError(f"expected {self.config.vae.in_channels} image channels")
        height, width = image.shape[-2:]
        if height % self.output_stride != 0 or width % self.output_stride != 0:
            raise ValueError(
                f"image height and width must be divisible by output_stride={self.output_stride}"
            )

    def encode_image(self, image: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        self._validate_image(image)
        latent = self.image_encoder(image)
        batch, channels, height, width = latent.shape
        expected = (image.shape[-2] // self.output_stride, image.shape[-1] // self.output_stride)
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

    def encode_vector(
        self,
        stroke_tokens: torch.Tensor | None,
        valid_mask: torch.Tensor | None,
        image_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse the vector branch into image tokens (identity when disabled).

        ``stroke_tokens`` is ``[B, N, 12]`` with ``[B, N]`` validity. In
        cross-attention mode image tokens query the encoded strokes; in add
        mode the masked-mean stroke feature is added to every image token.
        Rows with no valid stroke pass through unchanged.
        """

        if not self.vector_enabled or self.vector_encoder is None:
            return image_tokens
        if stroke_tokens is None or valid_mask is None:
            return image_tokens
        if stroke_tokens.ndim != 3 or stroke_tokens.shape[-1] != self.vector_encoder.feature_dim:
            raise ValueError(
                f"stroke_tokens must have shape [B,N,{self.vector_encoder.feature_dim}], "
                f"got {tuple(stroke_tokens.shape)}"
            )
        if valid_mask.shape != stroke_tokens.shape[:2]:
            raise ValueError(
                f"valid_mask must have shape {tuple(stroke_tokens.shape[:2])}, "
                f"got {tuple(valid_mask.shape)}"
            )
        if stroke_tokens.shape[0] != image_tokens.shape[0]:
            raise ValueError("stroke batch size must match the image batch size")

        encoded = self.vector_encoder(stroke_tokens, valid_mask)
        valid_rows = valid_mask.any(dim=1).to(encoded.dtype)

        if self.vector_fusion_mode == "cross_attention":
            key_padding = ~valid_mask.to(torch.bool)
            # Guard all-pad rows so key_padding_mask never masks everything.
            if not bool(valid_rows.all()):
                key_padding = key_padding.clone()
                key_padding[~valid_rows.bool(), 0] = False
            attended, _ = self.vector_cross_attention(
                query=self.vector_norm(image_tokens),
                key=encoded,
                value=encoded,
                key_padding_mask=key_padding,
                need_weights=False,
            )
            delta = self.vector_projection(attended)
            return image_tokens + delta * valid_rows[:, None, None].to(delta.dtype)

        pooled = masked_mean(encoded, valid_mask.to(torch.bool)) * valid_rows[:, None].to(
            encoded.dtype
        )
        delta = self.vector_projection(pooled)
        return image_tokens + delta.unsqueeze(1)

    @staticmethod
    def _normalise_conditioning_output(output) -> ConditioningOutput:
        if isinstance(output, ConditioningOutput):
            return output
        if isinstance(output, dict):
            tokens = output.get("tokens", output.get("token_embeddings"))
            mask = output.get("attention_mask", output.get("mask"))
            pooled = output.get("pooled", output.get("pooled_embedding"))
            if tokens is not None and mask is not None and pooled is not None:
                return ConditioningOutput(tokens=tokens, attention_mask=mask, pooled=pooled)
        if isinstance(output, (tuple, list)) and len(output) == 3:
            return ConditioningOutput(tokens=output[0], attention_mask=output[1], pooled=output[2])
        raise TypeError("conditioner must return ConditioningOutput, a matching dict, or a 3-tuple")

    def _condition(
        self,
        class_ids: torch.Tensor,
        texts: Sequence[str],
        reference: torch.Tensor,
    ) -> ConditioningOutput:
        output = self.conditioner(
            texts=list(texts),
            class_ids=class_ids,
            batch_size=class_ids.shape[0],
            device=reference.device,
            dtype=reference.dtype,
        )
        output = self._normalise_conditioning_output(output)
        return ConditioningOutput(
            tokens=output.tokens.to(device=reference.device, dtype=reference.dtype),
            attention_mask=output.attention_mask.to(device=reference.device, dtype=torch.bool),
            pooled=output.pooled.to(device=reference.device, dtype=reference.dtype),
        )

    def _run_pathway(
        self,
        image_tokens: torch.Tensor,
        height: int,
        width: int,
        class_ids: torch.Tensor,
        texts: Sequence[str],
        *,
        pathway_class_id: int | None,
    ) -> dict[str, torch.Tensor]:
        conditioning = self._condition(class_ids, texts, image_tokens)
        if self.pathway_mode == "shared":
            assert self.shared_fusion is not None
            fusion = self.shared_fusion
            blocks = self.shared_blocks
        else:
            if pathway_class_id is None:
                raise ValueError("per-class pathway requires pathway_class_id")
            fusion = self.early_fusions[pathway_class_id]
            blocks = self.class_blocks[pathway_class_id]

        features = fusion(image_tokens, conditioning)
        for block in blocks:
            features = block(features, spatial_shape=(height, width))
        batch = features.shape[0]
        features = self.out_norm(features).transpose(1, 2).reshape(
            batch,
            self.model_dim,
            height,
            width,
        )
        raw = self.heatmap_head(features)
        center_logits = raw[:, 0:1]
        return {
            "center_heatmap": torch.sigmoid(center_logits),
            "center_logits": center_logits,
            "size_map": F.softplus(raw[:, 1:3]),
            "offset_map": torch.sigmoid(raw[:, 3:5]),
        }

    def _process_class(
        self,
        class_id: int,
        image_tokens: torch.Tensor,
        height: int,
        width: int,
        texts: Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        if class_id < 0 or class_id >= self.num_classes:
            raise ValueError(f"class_id must be in [0, {self.num_classes})")
        batch = image_tokens.shape[0]
        class_ids = torch.full(
            (batch,), class_id, device=image_tokens.device, dtype=torch.long
        )
        resolved_texts = list(texts) if texts is not None else [self.class_texts[class_id]] * batch
        if len(resolved_texts) != batch:
            raise ValueError("texts must match the image batch size")
        return self._run_pathway(
            image_tokens,
            height,
            width,
            class_ids,
            resolved_texts,
            pathway_class_id=class_id if self.pathway_mode == "per_class" else None,
        )

    def _validate_class_ids(self, class_ids: torch.Tensor | Sequence[int], batch: int, device) -> torch.Tensor:
        class_ids = torch.as_tensor(class_ids, device=device, dtype=torch.long)
        if class_ids.ndim != 1 or class_ids.shape[0] != batch:
            raise ValueError(f"class_ids must have shape [{batch}]")
        if torch.any(class_ids < 0) or torch.any(class_ids >= self.num_classes):
            raise ValueError(f"class_ids must be in [0, {self.num_classes})")
        return class_ids

    def _selected_texts(
        self,
        texts: str | Sequence[str] | None,
        class_ids: torch.Tensor,
    ) -> list[str]:
        batch = class_ids.shape[0]
        if texts is None:
            return [self.class_texts[index] for index in class_ids.tolist()]
        if isinstance(texts, str):
            return [texts] * batch
        values = list(texts)
        if not all(isinstance(text, str) for text in values):
            raise TypeError("texts must contain strings")
        if len(values) == batch:
            return values
        if len(values) == self.num_classes:
            return [values[index] for index in class_ids.tolist()]
        raise ValueError(
            f"selected-class texts must contain {batch} sample texts or {self.num_classes} class texts"
        )

    def _all_class_texts(self, texts: str | Sequence[str] | None) -> list[str]:
        if texts is None:
            return list(self.class_texts)
        if isinstance(texts, str):
            return [texts] * self.num_classes
        values = list(texts)
        if len(values) != self.num_classes or not all(isinstance(text, str) for text in values):
            raise ValueError(f"all-class texts must contain exactly {self.num_classes} strings")
        return values

    def _forward_selected(
        self,
        image_tokens: torch.Tensor,
        height: int,
        width: int,
        class_ids: torch.Tensor,
        texts: list[str],
    ) -> dict[str, torch.Tensor]:
        grouped_indices = []
        grouped_outputs: dict[str, list[torch.Tensor]] = {
            "center_heatmap": [],
            "center_logits": [],
            "size_map": [],
            "offset_map": [],
        }
        for class_id_tensor in torch.unique(class_ids, sorted=True):
            class_id = int(class_id_tensor.item())
            indices = torch.nonzero(class_ids == class_id_tensor, as_tuple=False).flatten()
            grouped_indices.append(indices)
            group_texts = [texts[index] for index in indices.tolist()]
            output = self._run_pathway(
                image_tokens.index_select(0, indices),
                height,
                width,
                class_ids.index_select(0, indices),
                group_texts,
                pathway_class_id=class_id if self.pathway_mode == "per_class" else None,
            )
            for key in grouped_outputs:
                grouped_outputs[key].append(output[key])

        grouped_order = torch.cat(grouped_indices)
        restore_order = torch.argsort(grouped_order)
        return {
            key: torch.cat(values, dim=0).index_select(0, restore_order)
            for key, values in grouped_outputs.items()
        }

    def _forward_all_shared(
        self,
        image_tokens: torch.Tensor,
        height: int,
        width: int,
        class_texts: list[str],
        chunk_size: int,
    ) -> dict[str, torch.Tensor]:
        batch, length, dim = image_tokens.shape
        all_heatmaps, all_logits, all_sizes, all_offsets = [], [], [], []
        for start in range(0, self.num_classes, chunk_size):
            end = min(start + chunk_size, self.num_classes)
            classes_in_chunk = end - start
            expanded_tokens = image_tokens[:, None].expand(
                batch, classes_in_chunk, length, dim
            ).reshape(batch * classes_in_chunk, length, dim)
            chunk_ids = torch.arange(start, end, device=image_tokens.device).view(1, -1).expand(
                batch, -1
            ).reshape(-1)
            chunk_texts = [
                class_texts[class_id]
                for _ in range(batch)
                for class_id in range(start, end)
            ]
            output = self._run_pathway(
                expanded_tokens,
                height,
                width,
                chunk_ids,
                chunk_texts,
                pathway_class_id=None,
            )
            all_heatmaps.append(
                output["center_heatmap"].reshape(batch, classes_in_chunk, height, width)
            )
            all_logits.append(
                output["center_logits"].reshape(batch, classes_in_chunk, height, width)
            )
            all_sizes.append(
                output["size_map"].reshape(batch, classes_in_chunk * 2, height, width)
            )
            all_offsets.append(
                output["offset_map"].reshape(batch, classes_in_chunk * 2, height, width)
            )
        return {
            "center_heatmap": torch.cat(all_heatmaps, dim=1),
            "center_logits": torch.cat(all_logits, dim=1),
            "size_map": torch.cat(all_sizes, dim=1),
            "offset_map": torch.cat(all_offsets, dim=1),
        }

    def _forward_all_per_class(
        self,
        image_tokens: torch.Tensor,
        height: int,
        width: int,
        class_texts: list[str],
        chunk_size: int,
    ) -> dict[str, torch.Tensor]:
        all_heatmaps, all_logits, all_sizes, all_offsets = [], [], [], []
        for start in range(0, self.num_classes, chunk_size):
            end = min(start + chunk_size, self.num_classes)
            chunk_heatmaps, chunk_logits, chunk_sizes, chunk_offsets = [], [], [], []
            for class_id in range(start, end):
                output = self._process_class(
                    class_id,
                    image_tokens,
                    height,
                    width,
                    texts=[class_texts[class_id]] * image_tokens.shape[0],
                )
                chunk_heatmaps.append(output["center_heatmap"])
                chunk_logits.append(output["center_logits"])
                chunk_sizes.append(output["size_map"])
                chunk_offsets.append(output["offset_map"])
            all_heatmaps.append(torch.cat(chunk_heatmaps, dim=1))
            all_logits.append(torch.cat(chunk_logits, dim=1))
            all_sizes.append(torch.cat(chunk_sizes, dim=1))
            all_offsets.append(torch.cat(chunk_offsets, dim=1))
        return {
            "center_heatmap": torch.cat(all_heatmaps, dim=1),
            "center_logits": torch.cat(all_logits, dim=1),
            "size_map": torch.cat(all_sizes, dim=1),
            "offset_map": torch.cat(all_offsets, dim=1),
        }

    def forward(
        self,
        image: torch.Tensor,
        class_ids: torch.Tensor | Sequence[int] | None = None,
        texts: str | Sequence[str] | None = None,
        *,
        class_chunk_size: int | None = None,
        stroke_tokens: torch.Tensor | None = None,
        stroke_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        image_tokens, height, width = self.encode_image(image)
        # Vector (SVG stroke) fusion runs once per image, before conditioning,
        # so every class chunk benefits from the same fused image tokens.
        image_tokens = self.encode_vector(stroke_tokens, stroke_mask, image_tokens)
        if class_ids is not None:
            selected_ids = self._validate_class_ids(class_ids, image.shape[0], image.device)
            selected_texts = self._selected_texts(texts, selected_ids)
            return self._forward_selected(
                image_tokens,
                height,
                width,
                selected_ids,
                selected_texts,
            )

        chunk_size = self.class_chunk_size if class_chunk_size is None else class_chunk_size
        if chunk_size <= 0:
            raise ValueError("class_chunk_size must be positive")
        all_texts = self._all_class_texts(texts)
        if self.pathway_mode == "shared":
            return self._forward_all_shared(
                image_tokens,
                height,
                width,
                all_texts,
                chunk_size,
            )
        return self._forward_all_per_class(
            image_tokens,
            height,
            width,
            all_texts,
            chunk_size,
        )
