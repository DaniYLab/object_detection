"""
Model configuration — encoder specs matching pretrained targets.

VAE:  Flux VAE (AutoencoderKL) from black-forest-labs/FLUX.2-klein
Text: T5-v1.1-XXL encoder from google/t5-v1_1-xxl

Stubs match these dimensions so weights can be swapped in without
changing downstream code.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VAEConfig:
    """
    Flux VAE (AutoencoderKL) configuration.
    Source: black-forest-labs/FLUX.2-klein-9B/vae/config.json
    """

    _class_name: str = "AutoencoderKL"
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 16
    block_out_channels: list[int] = field(default_factory=lambda: [128, 256, 512, 512])
    down_block_types: list[str] = field(default_factory=lambda: [
        "DownEncoderBlock2D", "DownEncoderBlock2D",
        "DownEncoderBlock2D", "DownEncoderBlock2D",
    ])
    up_block_types: list[str] = field(default_factory=lambda: [
        "UpDecoderBlock2D", "UpDecoderBlock2D",
        "UpDecoderBlock2D", "UpDecoderBlock2D",
    ])
    layers_per_block: int = 2
    act_fn: str = "silu"
    norm_num_groups: int = 32
    mid_block_add_attention: bool = True
    sample_size: int = 1024
    scaling_factor: float = 0.3611
    shift_factor: float = 0.1159
    use_quant_conv: bool = False
    use_post_quant_conv: bool = False
    force_upcast: bool = True

    # Derived constants
    downsample_factor: int = 8     # 4 down blocks × stride 2 = 2^4... nhưng FLUX = 8x
    # Output: [B, 16, H/8, W/8]


@dataclass
class TextEncoderConfig:
    """
    T5-v1.1-XXL encoder configuration.
    Source: google/t5-v1_1-xxl (Hugging Face)

    FLUX uses only the encoder half of T5 (not the decoder).
    Output dim = d_model = 4096.
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
    max_length: int = 512          # max token length for text prompts

    # For our model: T5 output (4096) will be projected to model_dim
    # via nn.Linear(4096, model_dim)


@dataclass
class ModelConfig:
    """
    Full FloorPlanDetector configuration.
    """

    # Image
    image_size: int = 512

    # Internal model dimension (all modules project to this)
    model_dim: int = 512

    # VAE
    vae: VAEConfig = field(default_factory=VAEConfig)

    # Text encoder
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)

    # Per-class architecture
    num_classes: int = 35
    depth_per_class: int = 2       # OLB stack depth per class pathway

    # Attention
    num_heads: int = 8
    dropout: float = 0.1

    # Derived
    @property
    def latent_size(self) -> int:
        return self.image_size // self.vae.downsample_factor  # 512/8 = 64

    @property
    def latent_channels(self) -> int:
        return self.vae.latent_channels  # 16

    @property
    def text_output_dim(self) -> int:
        return self.text_encoder.d_model  # 4096
