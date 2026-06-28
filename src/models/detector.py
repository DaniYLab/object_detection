"""
FloorPlanCAD Detection Model — Conditioned Reflex Architecture.

Pipeline:
  1. Image  → VAE Encoder    → latent [B, 16, H/8, W/8] → img_tokens [B, h*w, D]
  2. 35 fixed texts → Text Encoder → txt_tokens [35, L, D]
  3. Per-class: EarlyFusion(image, text[c]) → class_blocks[c] → heatmap[c]
  4. CenterNet Head → center_heatmap + size_map
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks.object_learning_block import ObjectLearningBlock
from .config import VAEConfig, TextEncoderConfig


# ── VAE Encoder ───────────────────────────────────────────────────────────────
# Stub matches Flux VAE output: [B, 16, H/8, W/8]
# Swap with: AutoencoderKL.from_pretrained("black-forest-labs/FLUX.2-klein-9B", subfolder="vae")

class VAEEncoderStub(nn.Module):
    """
    Trainable VAE encoder stub — matches Flux VAE dimensions.

    Config (from FLUX.2-klein):
      block_out_channels: [128, 256, 512, 512]
      latent_channels:    16
      downsample:         8× (3 stride-2 stages)
      act_fn:             silu

    Output: [B, 16, H/8, W/8]
    """

    def __init__(self, cfg: VAEConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or VAEConfig()
        ch = cfg.block_out_channels  # [128, 256, 512, 512]

        self.encoder = nn.Sequential(
            # Stage 1: 3 → 128, /2
            nn.Conv2d(cfg.in_channels, ch[0], 3, stride=2, padding=1),
            nn.SiLU(),
            # Stage 2: 128 → 256, /4
            nn.Conv2d(ch[0], ch[1], 3, stride=2, padding=1),
            nn.SiLU(),
            # Stage 3: 256 → 512, /8
            nn.Conv2d(ch[1], ch[2], 3, stride=2, padding=1),
            nn.SiLU(),
            # Stage 4: 512 → 512 (no stride, matches Flux 4th block)
            nn.Conv2d(ch[2], ch[3], 3, padding=1),
            nn.SiLU(),
            # Project → latent channels
            nn.Conv2d(ch[3], cfg.latent_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)  # [B, 16, H/8, W/8]


# ── Text Encoder ──────────────────────────────────────────────────────────────
# Stub matches T5-v1.1-XXL output: [B, L, 4096] → projected to model_dim
# Swap with: T5EncoderModel.from_pretrained("google/t5-v1_1-xxl")

class TextEncoderStub(nn.Module):
    """
    Trainable text encoder stub — matches T5-v1.1-XXL dimensions.

    Config (from google/t5-v1_1-xxl):
      vocab_size: 32128
      d_model:    4096
      num_heads:  64
      num_layers: 24

    Stub: Embedding + Positional + LayerNorm → project 4096 → model_dim.
    """

    def __init__(self, cfg: TextEncoderConfig | None = None, model_dim: int = 512) -> None:
        super().__init__()
        cfg = cfg or TextEncoderConfig()

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=0)
        self.pos = nn.Embedding(cfg.max_length, cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, model_dim, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos(pos)
        x = self.norm(x)              # [B, L, 4096]
        return self.proj(x)           # [B, L, model_dim]


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

# Fixed text prompts — one per class (built into model, not external input)
from src.data.dataset import CLASS_NAMES, TEXT_TEMPLATE, NUM_CLASSES

CLASS_TEXTS = [TEXT_TEMPLATE.format(cls=name) for name in CLASS_NAMES]


def _tokenize_fixed(texts: list[str], max_len: int = 32, vocab_size: int = 32000) -> torch.Tensor:
    """Hash-based tokenizer for fixed class texts. Deterministic."""
    tokens = []
    for text in texts:
        ids = [hash(word) % (vocab_size - 1) + 1 for word in text.lower().split()]
        ids = ids[:max_len] + [0] * max(0, max_len - len(ids))
        tokens.append(ids)
    return torch.tensor(tokens, dtype=torch.long)


class FloorPlanDetector(nn.Module):
    """
    FloorPlanCAD Multimodal Detection Model.

    Architecture (Conditioned Reflex):
      Input : image [B, 3, H, W]  — chỉ 1 ảnh đầu vào
      Built-in: 35 fixed text prompts, mỗi text = 1 HEAD (block)

      1. Image → VAE Encoder → image tokens (encode 1 lần)
      2. 35 texts → Text Encoder → 35 text token sets
      3. Mỗi class: EarlyFusion(image, text[c]) → class_blocks[c] → heatmap[c]
      4. Output: 35 × (center_heatmap + size_map)

    Training: class_ids chọn class nào tính loss.
    Inference: lấy hết 35 outputs.
    """

    def __init__(
        self,
        image_size: int = 512,
        model_dim: int = 512,
        num_classes: int = 35,
        depth_per_class: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        vae_cfg: VAEConfig | None = None,
        text_cfg: TextEncoderConfig | None = None,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.latent_size = image_size // 8
        self.model_dim = model_dim
        self.num_classes = num_classes
        self.depth_per_class = depth_per_class

        vae_cfg = vae_cfg or VAEConfig()
        text_cfg = text_cfg or TextEncoderConfig()

        # ── Encoders ─────────────────────────────────────────────────────────
        self.vae_encoder = VAEEncoderStub(vae_cfg)
        self.text_encoder = TextEncoderStub(text_cfg, model_dim=model_dim)

        # Project VAE latent → model_dim
        self.img_proj = nn.Linear(vae_cfg.latent_channels, model_dim)

        # ── Fixed class text tokens (registered as buffer, not parameter) ────
        # 35 tokenized texts, one per class — part of the model, not input
        self.register_buffer(
            "class_text_ids",
            _tokenize_fixed(CLASS_TEXTS),  # [35, 32]
        )

        # ── Per-class Early Fusion ────────────────────────────────────────────
        # Each class has its own fusion layer (text differs per class)
        self.early_fusions = nn.ModuleList([
            EarlyFusion(model_dim, num_heads)
            for _ in range(num_classes)
        ])

        # ── Per-class Object Learning Blocks ──────────────────────────────────
        self.class_blocks = nn.ModuleList([
            nn.ModuleList([
                ObjectLearningBlock(model_dim, num_classes=1, num_heads=num_heads, dropout=dropout)
                for _ in range(depth_per_class)
            ])
            for _ in range(num_classes)
        ])

        # ── Spatial reshape + CenterNet Head ──────────────────────────────────
        self.out_norm = nn.LayerNorm(model_dim)
        self.heatmap_head = HeatmapHead(model_dim, out_channels=3)

    def encode_image(self, image: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """image [B,3,H,W] → tokens [B, h*w, D]"""
        lat = self.vae_encoder(image)               # [B, C, h, w]
        B, C, h, w = lat.shape
        lat = lat.flatten(2).transpose(1, 2)        # [B, h*w, C]
        return self.img_proj(lat), h, w             # [B, h*w, D], h, w

    def _process_class(
        self,
        cid: int,
        img_tokens: torch.Tensor,       # [B, h*w, D]
        h: int, w: int,
    ) -> dict[str, torch.Tensor]:
        """Process one class: early fusion + OLB + head."""
        # Text encode for this class
        txt_ids = self.class_text_ids[cid:cid+1]         # [1, 32]
        txt_ids = txt_ids.expand(img_tokens.shape[0], -1) # [B, 32]
        txt_tokens = self.text_encoder(txt_ids)            # [B, 32, D]

        # Early fusion (class-specific)
        fused = self.early_fusions[cid](img_tokens, txt_tokens)  # [B, h*w, D]

        # Object Learning Blocks
        dummy_cid = torch.zeros(img_tokens.shape[0], dtype=torch.long, device=img_tokens.device)
        x = fused
        for block in self.class_blocks[cid]:
            x = block(x, dummy_cid)

        # Reshape → spatial → head
        B = img_tokens.shape[0]
        x = self.out_norm(x)
        x = x.transpose(1, 2).reshape(B, self.model_dim, h, w)
        out = self.heatmap_head(x)

        return {
            "center_heatmap": torch.sigmoid(out[:, 0:1, :, :]),
            "size_map": F.relu(out[:, 1:3, :, :]),
        }

    def forward(
        self,
        image: torch.Tensor,               # [B, 3, H, W]
        class_ids: torch.Tensor | None = None,  # [B] — training: chọn class tính loss
    ) -> dict[str, torch.Tensor]:
        """
        Training (class_ids provided):
            Chỉ chạy block của class được chỉ định cho mỗi sample.
            Returns: center_heatmap [B,1,h,w], size_map [B,2,h,w]

        Inference (class_ids=None):
            Chạy tất cả 35 blocks.
            Returns: center_heatmap [B,35,h,w], size_map [B,35*2,h,w]
        """
        B = image.shape[0]
        img_tokens, h, w = self.encode_image(image)  # encode 1 lần

        if class_ids is not None:
            # ── Training mode: mỗi sample chạy qua block tương ứng ────────
            hm_list, sz_list = [], []
            for i in range(B):
                cid = class_ids[i].item()
                result = self._process_class(cid, img_tokens[i:i+1], h, w)
                hm_list.append(result["center_heatmap"])
                sz_list.append(result["size_map"])
            return {
                "center_heatmap": torch.cat(hm_list, dim=0),  # [B, 1, h, w]
                "size_map": torch.cat(sz_list, dim=0),          # [B, 2, h, w]
            }
        else:
            # ── Inference mode: chạy tất cả 35 blocks ─────────────────────
            all_hm, all_sz = [], []
            for cid in range(self.num_classes):
                result = self._process_class(cid, img_tokens, h, w)
                all_hm.append(result["center_heatmap"])   # [B, 1, h, w]
                all_sz.append(result["size_map"])          # [B, 2, h, w]
            return {
                "center_heatmap": torch.cat(all_hm, dim=1),  # [B, 35, h, w]
                "size_map": torch.cat(all_sz, dim=1),         # [B, 70, h, w]
            }


if __name__ == "__main__":
    model = FloorPlanDetector(
        image_size=512, model_dim=256, num_classes=35, depth_per_class=2
    )
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.1f}M")
    print(f"Per-class blocks: {len(model.class_blocks)} × {model.depth_per_class} depth")

    image = torch.randn(1, 3, 512, 512)

    # Training mode: 1 class per sample
    cls_ids = torch.tensor([4])
    out = model(image, class_ids=cls_ids)
    print(f"\n[Training] class_id=4 (chair)")
    print(f"  center_heatmap: {out['center_heatmap'].shape}")  # [1, 1, 64, 64]
    print(f"  size_map      : {out['size_map'].shape}")        # [1, 2, 64, 64]

    # Inference mode: all 35 classes
    out = model(image)
    print(f"\n[Inference] all classes")
    print(f"  center_heatmap: {out['center_heatmap'].shape}")  # [1, 35, 64, 64]
    print(f"  size_map      : {out['size_map'].shape}")        # [1, 70, 64, 64]


