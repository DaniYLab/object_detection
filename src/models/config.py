"""Serializable model configuration and named model presets."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from typing import Any


@dataclass
class VAEConfig:
    """Configuration for the project-native stride-8 image encoder.

    The historical name is retained because existing checkpoints and launch code
    use ``vae_cfg``. The default implementation is a small convolutional encoder;
    it does not claim to contain pretrained VAE weights.
    """

    _class_name: str = "ConvImageEncoder"
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 64
    block_out_channels: tuple[int, ...] = (64, 128, 256, 256)
    down_block_types: tuple[str, ...] = ()
    up_block_types: tuple[str, ...] = ()
    layers_per_block: int = 1
    act_fn: str = "silu"
    norm_num_groups: int = 32
    mid_block_add_attention: bool = False
    sample_size: int = 512
    scaling_factor: float = 1.0
    shift_factor: float = 0.0
    use_quant_conv: bool = False
    use_post_quant_conv: bool = False
    force_upcast: bool = False
    downsample_factor: int = 8

    def __post_init__(self) -> None:
        self.block_out_channels = tuple(self.block_out_channels)
        self.down_block_types = tuple(self.down_block_types)
        self.up_block_types = tuple(self.up_block_types)
        if len(self.block_out_channels) < 3:
            raise ValueError("block_out_channels must contain at least three stages")
        if self.downsample_factor != 8:
            raise ValueError("the project-native image encoder has a fixed stride of 8")
        if self.in_channels <= 0 or self.latent_channels <= 0:
            raise ValueError("encoder channel counts must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> VAEConfig:
        return cls(**values)


# A clearer name for new code while preserving the public VAEConfig API.
ImageEncoderConfig = VAEConfig


@dataclass
class TextEncoderConfig:
    """Compatibility settings for text encoders.

    Lightweight byte conditioning uses ``max_length`` and ``embedding_dim``.
    ``model_name`` and the Hugging Face metadata are used only by the optional
    lazy pretrained conditioner.
    """

    model_name: str = "google/t5-v1_1-xxl"
    vocab_size: int = 32128
    d_model: int = 4096
    d_ff: int = 10240
    d_kv: int = 64
    num_heads: int = 64
    num_layers: int = 24
    feed_forward_proj: str = "gated-gelu"
    dropout_rate: float = 0.1
    max_length: int = 64
    embedding_dim: int = 128

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> TextEncoderConfig:
        return cls(**values)


@dataclass
class ConditionerConfig:
    """Serializable conditioning configuration.

    Kinds:
      - ``none``: no class or text signal.
      - ``class_embedding``: one learned embedding per class.
      - ``lightweight_text``: lightweight UTF-8 byte token encoder.
      - ``pretrained_text``: lazily loaded Hugging Face tokenizer and encoder.
    """

    kind: str = "byte"
    max_length: int = 64
    embedding_dim: int = 128
    model_name: str = "distilbert-base-uncased"
    freeze_pretrained: bool = True
    local_files_only: bool = False
    revision: str | None = None

    def __post_init__(self) -> None:
        aliases = {
            "class": "class_embedding",
            "byte": "lightweight_text",
            "lightweight": "lightweight_text",
            "hf": "pretrained_text",
            "hf_pretrained": "pretrained_text",
            "pretrained": "pretrained_text",
        }
        self.kind = aliases.get(self.kind, self.kind)
        if self.kind not in {"none", "class_embedding", "lightweight_text", "pretrained_text"}:
            raise ValueError(f"unknown conditioner kind: {self.kind}")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ConditionerConfig:
        return cls(**values)


@dataclass
class VectorBranchConfig:
    """Configuration for the dual-pathway SVG stroke (vector) branch.

    ``enabled`` gates the whole branch. ``n_max`` bounds the token count per
    drawing (drawings are subsampled when larger — see ``src/data/strokes.py``).
    ``fusion`` selects how encoded vector tokens meet image tokens:

    - ``cross_attention``: image tokens query the vector tokens (K/V);
    - ``add``: mean-pooled vector features are added to every image token.
    """

    enabled: bool = False
    n_max: int = 1024
    depth: int = 1
    num_heads: int = 8
    fusion: str = "cross_attention"
    feature_dim: int = 12

    def __post_init__(self) -> None:
        if self.n_max <= 0:
            raise ValueError("vector n_max must be positive")
        if self.depth < 0:
            raise ValueError("vector depth cannot be negative")
        if self.num_heads <= 0:
            raise ValueError("vector num_heads must be positive")
        if self.fusion not in {"cross_attention", "add"}:
            raise ValueError(f"unknown vector fusion mode: {self.fusion!r}")
        if self.feature_dim <= 0:
            raise ValueError("vector feature_dim must be positive")


@dataclass
class ModelConfig:
    """Full serializable configuration for :class:`FloorPlanDetector`."""

    architecture: str = "floorplan_detector"
    image_size: int = 512
    output_stride: int = 8
    model_dim: int = 256
    num_classes: int = 35
    class_texts: tuple[str, ...] | None = None
    depth_per_class: int = 2
    num_heads: int = 8
    dropout: float = 0.1
    fusion_mode: str = "film"
    pathway_mode: str = "shared"
    class_chunk_size: int = 8
    attention_chunk_size: int | None = None
    head_channels: int = 128
    vae: VAEConfig = field(default_factory=VAEConfig)
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    conditioner: ConditionerConfig = field(
        default_factory=lambda: ConditionerConfig(kind="class_embedding")
    )
    vector: VectorBranchConfig = field(default_factory=VectorBranchConfig)

    def __post_init__(self) -> None:
        if isinstance(self.vae, dict):
            self.vae = VAEConfig.from_dict(self.vae)
        if isinstance(self.text_encoder, dict):
            self.text_encoder = TextEncoderConfig.from_dict(self.text_encoder)
        if isinstance(self.conditioner, dict):
            self.conditioner = ConditionerConfig.from_dict(self.conditioner)
        if isinstance(self.vector, dict):
            self.vector = VectorBranchConfig(**self.vector)
        elif not isinstance(self.vector, VectorBranchConfig):
            raise TypeError("vector must be a VectorBranchConfig or a dict")
        if self.class_texts is not None:
            self.class_texts = tuple(self.class_texts)
            if len(self.class_texts) != self.num_classes or not all(
                isinstance(text, str) for text in self.class_texts
            ):
                raise ValueError("class_texts must contain one string per class")

        architecture_aliases = {
            "detector": "floorplan_detector",
            "floorplan": "floorplan_detector",
            "baseline": "centernet_baseline",
            "shared_centernet": "centernet_baseline",
        }
        self.architecture = architecture_aliases.get(self.architecture, self.architecture)
        if self.architecture not in {
            "floorplan_detector",
            "floorplan_unconditioned",
            "centernet_baseline",
        }:
            raise ValueError(f"unknown model architecture: {self.architecture}")

        aliases = {"shared_pathway": "shared", "class": "per_class", "per-class": "per_class"}
        self.pathway_mode = aliases.get(self.pathway_mode, self.pathway_mode)
        if self.pathway_mode not in {"shared", "per_class"}:
            raise ValueError(f"unknown pathway_mode: {self.pathway_mode}")
        if self.output_stride != 8:
            raise ValueError("FloorPlanDetector currently supports output_stride=8 only")
        if self.image_size <= 0 or self.image_size % self.output_stride != 0:
            raise ValueError("image_size must be positive and divisible by output_stride")
        if self.model_dim <= 0 or self.model_dim % self.num_heads != 0:
            raise ValueError("model_dim must be positive and divisible by num_heads")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if self.depth_per_class < 0:
            raise ValueError("depth_per_class cannot be negative")
        if self.class_chunk_size <= 0:
            raise ValueError("class_chunk_size must be positive")
        if self.attention_chunk_size is not None and self.attention_chunk_size <= 0:
            raise ValueError("attention_chunk_size must be positive when provided")
        if self.head_channels <= 0:
            raise ValueError("head_channels must be positive")

    @property
    def latent_size(self) -> int:
        return self.image_size // self.output_stride

    @property
    def latent_channels(self) -> int:
        return self.vae.latent_channels

    @property
    def text_output_dim(self) -> int:
        return self.text_encoder.d_model

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable nested dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ModelConfig:
        data = deepcopy(values)
        if "model_type" in data and "architecture" not in data:
            data["architecture"] = data.pop("model_type")
        if "vae" in data and isinstance(data["vae"], dict):
            data["vae"] = VAEConfig.from_dict(data["vae"])
        if "text_encoder" in data and isinstance(data["text_encoder"], dict):
            data["text_encoder"] = TextEncoderConfig.from_dict(data["text_encoder"])
        if "conditioner" in data and isinstance(data["conditioner"], dict):
            data["conditioner"] = ConditionerConfig.from_dict(data["conditioner"])
        return cls(**data)

    def with_overrides(self, **overrides: Any) -> ModelConfig:
        """Return a validated copy with top-level fields replaced."""
        return replace(self, **overrides)


MODEL_PRESETS: dict[str, ModelConfig] = {}
# Public alias for callers that prefer an explicit registry name.
PRESET_REGISTRY = MODEL_PRESETS


def register_model_preset(name: str, config: ModelConfig, *, overwrite: bool = False) -> None:
    """Register a named configuration without exposing mutable shared state."""
    if not name or not isinstance(name, str):
        raise ValueError("preset name must be a non-empty string")
    if name in MODEL_PRESETS and not overwrite:
        raise KeyError(f"model preset already registered: {name}")
    MODEL_PRESETS[name] = deepcopy(config)


def get_model_preset(name: str, **overrides: Any) -> ModelConfig:
    """Return an independent copy of a registered model configuration."""
    try:
        config = deepcopy(MODEL_PRESETS[name])
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_PRESETS))
        raise KeyError(f"unknown model preset '{name}'. Available: {choices}") from exc
    return config.with_overrides(**overrides) if overrides else config


def list_model_presets() -> tuple[str, ...]:
    return tuple(sorted(MODEL_PRESETS))


# Short aliases are convenient in scripts and preserve a small public API.
get_preset = get_model_preset
register_preset = register_model_preset


register_model_preset("floorplan_base", ModelConfig())
register_model_preset(
    "floorplan_shared",
    ModelConfig(pathway_mode="shared", conditioner=ConditionerConfig(kind="class_embedding")),
)
register_model_preset(
    "shared_class_embedding_film",
    ModelConfig(
        pathway_mode="shared",
        fusion_mode="film",
        conditioner=ConditionerConfig(kind="class_embedding"),
    ),
)
register_model_preset(
    "floorplan_tiny",
    ModelConfig(
        image_size=64,
        model_dim=64,
        num_classes=4,
        depth_per_class=1,
        num_heads=4,
        dropout=0.0,
        pathway_mode="shared",
        class_chunk_size=2,
        head_channels=64,
        vae=VAEConfig(latent_channels=32, block_out_channels=(16, 24, 32)),
        conditioner=ConditionerConfig(kind="lightweight_text", max_length=24, embedding_dim=32),
    ),
)
register_model_preset(
    "per_class_no_text",
    ModelConfig(
        pathway_mode="per_class",
        fusion_mode="none",
        conditioner=ConditionerConfig(kind="none"),
    ),
)
register_model_preset(
    "shared_fixed_byte_text_film",
    ModelConfig(
        pathway_mode="shared",
        fusion_mode="film",
        conditioner=ConditionerConfig(kind="lightweight_text"),
    ),
)
register_model_preset(
    "per_class_fixed_byte_text_film",
    ModelConfig(
        pathway_mode="per_class",
        fusion_mode="film",
        conditioner=ConditionerConfig(kind="lightweight_text"),
    ),
)
register_model_preset(
    "shared_fixed_byte_text_cross_attention",
    ModelConfig(
        pathway_mode="shared",
        fusion_mode="cross_attention",
        conditioner=ConditionerConfig(kind="lightweight_text"),
    ),
)
register_model_preset(
    "shared_fixed_text_cross_attention",
    ModelConfig(
        pathway_mode="shared",
        fusion_mode="cross_attention",
        conditioner=ConditionerConfig(kind="lightweight_text"),
    ),
)
register_model_preset(
    "shared_pretrained_text",
    ModelConfig(
        pathway_mode="shared",
        fusion_mode="film",
        conditioner=ConditionerConfig(kind="pretrained_text"),
    ),
)
register_model_preset(
    "per_class_pretrained_text",
    ModelConfig(
        pathway_mode="per_class",
        fusion_mode="film",
        conditioner=ConditionerConfig(kind="pretrained_text"),
    ),
)
register_model_preset(
    "legacy_per_class_byte_film",
    ModelConfig(
        pathway_mode="per_class",
        fusion_mode="film",
        conditioner=ConditionerConfig(kind="lightweight_text"),
    ),
)
register_model_preset(
    "centernet_shared",
    ModelConfig(
        architecture="centernet_baseline",
        pathway_mode="shared",
        fusion_mode="none",
        conditioner=ConditionerConfig(kind="none"),
    ),
)
register_model_preset(
    "centernet_baseline",
    ModelConfig(
        architecture="centernet_baseline",
        pathway_mode="shared",
        fusion_mode="none",
        conditioner=ConditionerConfig(kind="none"),
    ),
)

# ── Ablation control presets (P0-B / P0-C) ────────────────────────────────────

# Direct architecture control: FloorPlanDetector with shared pathway and no
# conditioning. Compare against this to isolate the effect of conditioning;
# comparing against centernet_baseline mixes in architecture differences.
register_model_preset(
    "shared_no_condition",
    ModelConfig(
        architecture="floorplan_unconditioned",
        pathway_mode="shared",
        fusion_mode="none",
        conditioner=ConditionerConfig(kind="none"),
    ),
)

# Budget-matched shared-wide: same architecture as floorplan_base but with a
# larger model_dim so parameter count is comparable to per_class_no_text.
# Use this as the budget-matched control when comparing against per-class
# variants. Measure actual parameter counts before drawing conclusions.
register_model_preset(
    "shared_wide",
    ModelConfig(
        architecture="floorplan_unconditioned",
        pathway_mode="shared",
        fusion_mode="none",
        conditioner=ConditionerConfig(kind="none"),
        model_dim=1024,
        depth_per_class=4,
        num_heads=8,
    ),
)

# Budget-matched per-class-small: per-class routing with a smaller model_dim
# and shallower depth so parameter count is close to floorplan_base / shared
# presets. Compare against shared_no_condition for routing effect.
register_model_preset(
    "per_class_small",
    ModelConfig(
        architecture="floorplan_detector",
        pathway_mode="per_class",
        fusion_mode="none",
        conditioner=ConditionerConfig(kind="none"),
        model_dim=72,
        depth_per_class=1,
        num_heads=8,
    ),
)

# ── Dual-pathway (image + SVG vector) presets ──────────────────────────────────

# Main dual-pathway model: floorplan_base (class embedding + FiLM) plus the
# SVG stroke branch fused into the image tokens via cross-attention. Compare
# against floorplan_base on the same seeds to isolate the vector-source gain.
register_model_preset(
    "dual_pathway",
    ModelConfig(
        vector=VectorBranchConfig(enabled=True),
    ),
)

# Control: vector encoder present and trained, but its output is never used
# downstream. Isolates the fusion contribution from the extra capacity.
register_model_preset(
    "dual_no_fusion",
    ModelConfig(
        vector=VectorBranchConfig(enabled=True, fusion="cross_attention"),
        conditioner=ConditionerConfig(kind="none"),
        fusion_mode="none",
    ),
)

