"""
Full FloorPlanCAD Detection Model.

Pipeline:
  1. Image  → VAE Encoder    → image tokens [B, H*W, D]
  2. Text   → Text Encoder   → text tokens  [B, L, D]
  3. Early Fusion: Cross-Attention(text query → image key/value)
  4. Per-class pass through ObjectLearningBlock (class-conditioned)
  5. Aggregate all class features → Heatmap head [B, 35, H, W]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks.object_learning_block import ObjectLearningBlock


# ── Lightweight VAE Encoder stub ──────────────────────────────────────────────
# Replace with: from diffusers import AutoencoderKL

class VAEEncoderStub(nn.Module):
    """
    Placeholder VAE encoder.
    Replace with Flux VAE (AutoencoderKL) for production.
    Output: latent [B, latent_channels, H/8, W/8]
    """

    def __init__(self, in_channels: int = 3, latent_channels: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),   # /2
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),           # /4
            nn.SiLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),          # /8
            nn.SiLU(),
            nn.Conv2d(256, latent_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ── Lightweight Text Encoder stub ─────────────────────────────────────────────
# Replace with: from transformers import T5EncoderModel / CLIPTextModel

class TextEncoderStub(nn.Module):
    """
    Placeholder text encoder — maps tokenized text to embeddings.
    Replace with T5EncoderModel for production.
    """

    def __init__(self, vocab_size: int = 32000, embed_dim: int = 512, max_len: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos = nn.Embedding(max_len, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, L]
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos(pos)
        return self.norm(x)                             # [B, L, D]


# ── Early Fusion: Cross-Attention (text → image) ──────────────────────────────

class EarlyFusion(nn.Module):
    """
    Cross-attention: text tokens as query, image tokens as key/value.
    Output: enriched image tokens [B, img_len, D].
    """

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.norm_img = nn.LayerNorm(dim)
        self.norm_txt = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        img_tokens: torch.Tensor,   # [B, img_len, D]
        txt_tokens: torch.Tensor,   # [B, txt_len, D]
    ) -> torch.Tensor:
        img = self.norm_img(img_tokens)
        txt = self.norm_txt(txt_tokens)
        # Query=text, Key/Value=image → text attends to image
        fused, _ = self.cross_attn(query=txt, key=img, value=img)
        # Project back to image space via mean pooling text output + residual
        fused_mean = fused.mean(dim=1, keepdim=True).expand_as(img_tokens)
        return img_tokens + self.proj(fused_mean)


# ── Heatmap Prediction Head ───────────────────────────────────────────────────

class HeatmapHead(nn.Module):
    """
    Converts feature map to per-class heatmap.
    Input : [B, D, H, W]
    Output: [B, num_classes, H_out, W_out]
    """

    def __init__(self, in_dim: int, out_channels: int = 3) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)             # [B, 35, H, W]


# ── Full Model ────────────────────────────────────────────────────────────────

class FloorPlanDetector(nn.Module):
    """
    FloorPlanCAD Multimodal Detection Model.

    Args:
        image_size      : input image size (square)
        latent_channels : VAE latent channels (16 for Flux VAE)
        model_dim       : internal feature dimension
        num_classes     : number of object classes (35)
        num_blocks      : depth of ObjectLearning stack
        num_heads       : attention heads
    """

    def __init__(
        self,
        image_size: int = 512,
        latent_channels: int = 16,
        model_dim: int = 512,
        num_classes: int = 1,  # Only 1 class for class-agnostic
        num_blocks: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.latent_size = image_size // 8       # VAE 8x downsampling
        self.model_dim = model_dim
        self.num_classes = num_classes

        # ── Encoders ─────────────────────────────────────────────────────────
        self.vae_encoder = VAEEncoderStub(3, latent_channels)
        self.text_encoder = TextEncoderStub(embed_dim=model_dim)

        # Project VAE latent → model_dim
        self.img_proj = nn.Linear(latent_channels, model_dim)

        # ── Early Fusion ──────────────────────────────────────────────────────
        self.early_fusion = EarlyFusion(model_dim, num_heads)

        # ── Object Learning Blocks (shared, class-conditioned) ────────────────
        self.blocks = nn.ModuleList([
            ObjectLearningBlock(model_dim, num_classes, num_heads, dropout)
            for _ in range(num_blocks)
        ])

        # ── Per-class feature projection ──────────────────────────────────────
        # After blocks: project each class pass separately
        self.class_proj = nn.Linear(model_dim, model_dim)

        # ── Heatmap Head (1 for center, 2 for size) ───────────────────────────
        self.heatmap_head = HeatmapHead(model_dim, out_channels=3)

        # ── Spatial reshape helper ────────────────────────────────────────────
        self.out_norm = nn.LayerNorm(model_dim)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """image [B,3,H,W] → tokens [B, h*w, D]"""
        lat = self.vae_encoder(image)               # [B, C, h, w]
        B, C, h, w = lat.shape
        lat = lat.flatten(2).transpose(1, 2)        # [B, h*w, C]
        return self.img_proj(lat), h, w             # [B, h*w, D], h, w

    def forward(
        self,
        image: torch.Tensor,                        # [B, 3, H, W]
        text_ids: torch.Tensor,                     # [B, L]  tokenized text
        class_ids: torch.Tensor,                    # [B]     active class per sample
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            center_heatmap: [B, 1, h, w] (sigmoid applied)
            size_map:       [B, 2, h, w] (ReLU applied)
        """
        B = image.shape[0]

        # ── 1. Encode ─────────────────────────────────────────────────────────
        img_tokens, h, w = self.encode_image(image)     # [B, h*w, D]
        txt_tokens = self.text_encoder(text_ids)         # [B, L, D]

        # ── 2. Early Fusion ───────────────────────────────────────────────────
        fused = self.early_fusion(img_tokens, txt_tokens)  # [B, h*w, D]

        # ── 3. Object Learning (class-conditioned) ────────────────────────────
        x = fused
        for block in self.blocks:
            x = block(x, class_ids)                      # [B, h*w, D]

        # ── 4. Reshape → spatial ──────────────────────────────────────────────
        x = self.out_norm(x)
        x = x.transpose(1, 2).reshape(B, self.model_dim, h, w)  # [B, D, h, w]

        # ── 5. Heatmap head ───────────────────────────────────────────────────
        out = self.heatmap_head(x)                       # [B, 3, h, w]
        
        center_heatmap = torch.sigmoid(out[:, 0:1, :, :])
        size_map = F.relu(out[:, 1:3, :, :])
        
        return {
            "center_heatmap": center_heatmap,
            "size_map": size_map,
        }


if __name__ == "__main__":
    model = FloorPlanDetector(image_size=512, model_dim=512, num_classes=1, num_blocks=4)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.1f}M")

    image    = torch.randn(2, 3, 512, 512)
    text_ids = torch.randint(0, 32000, (2, 16))
    cls_ids  = torch.tensor([0, 0])

    out = model(image, text_ids, cls_ids)
    print(f"center_heatmap: {out['center_heatmap'].shape}")   # [2, 1, 64, 64]
    print(f"size_map      : {out['size_map'].shape}")         # [2, 2, 64, 64]
