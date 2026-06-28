"""
FloorPlanCAD Detection Model — Conditioned Reflex Architecture.

Pipeline:
  1. Image  → VAE Encoder    → image tokens [B, H*W, D]
  2. Text   → Text Encoder   → text tokens  [B, L, D]
  3. Early Fusion: Cross-Attention(text query → image key/value)
  4. Route through DEDICATED per-class ObjectLearningBlock (35 pathways)
  5. CenterNet Head → center_heatmap [B, 1, h, w] + size_map [B, 2, h, w]
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
        latent_channels: int = 16,
        model_dim: int = 512,
        num_classes: int = 35,
        depth_per_class: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.latent_size = image_size // 8
        self.model_dim = model_dim
        self.num_classes = num_classes
        self.depth_per_class = depth_per_class

        # ── Encoders ─────────────────────────────────────────────────────────
        self.vae_encoder = VAEEncoderStub(3, latent_channels)
        self.text_encoder = TextEncoderStub(embed_dim=model_dim)

        # Project VAE latent → model_dim
        self.img_proj = nn.Linear(latent_channels, model_dim)

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


