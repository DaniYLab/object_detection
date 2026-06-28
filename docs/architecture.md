# Kiến Trúc Hệ Thống (Architecture Reference)

## Tổng Quan Pipeline

```
Image [B, 3, 512, 512]
  │
  ├──► VAEEncoderStub ──► latent [B, 16, 64, 64] ──► flatten + proj ──► img_tokens [B, 4096, D]
  │
  │    Text "Find {class} ..." 
  │      │
  │      ├──► hash tokenizer ──► text_ids [B, 32]
  │      └──► TextEncoderStub ──► txt_tokens [B, 32, D]
  │
  ├──► EarlyFusion(img_tokens, txt_tokens) ──► fused [B, 4096, D]
  │
  ├──► Route by class_id ──► class_blocks[cid] ──► x [B, 4096, D]
  │    (35 separate Mamba+Attention stacks)
  │
  └──► HeatmapHead ──► [B, 3, 64, 64]
         ├── channel 0: sigmoid → center_heatmap [B, 1, 64, 64]
         └── channel 1-2: ReLU  → size_map       [B, 2, 64, 64]
```

## Các Module

### 1. VAEEncoderStub (`detector.py`)
- **Vai trò:** Mã hoá ảnh thành latent space (downscale 8x)
- **Hiện tại:** 3-layer Conv stub (64→128→256→16)
- **Tương lai:** Thay bằng Flux VAE (`AutoencoderKL` từ `diffusers`)
- **Output:** `[B, 16, H/8, W/8]`

### 2. TextEncoderStub (`detector.py`)
- **Vai trò:** Mã hoá text prompt thành embedding
- **Hiện tại:** Embedding + Positional + LayerNorm
- **Tương lai:** Thay bằng T5 Encoder hoặc CLIP Text
- **Output:** `[B, L, D]`

### 3. EarlyFusion (`detector.py`)
- **Vai trò:** Kết hợp text với image qua Cross-Attention
- **Cơ chế:** Query=text, Key/Value=image → text "hỏi" ảnh
- **Output:** Enriched image tokens `[B, img_len, D]`

### 4. ObjectLearningBlock (`blocks/object_learning_block.py`)
- **Vai trò:** Khối học chuyên biệt cho mỗi class
- **Cấu trúc:** Mamba (long-range) → Self-Attention (spatial) → FFN
- **Số lượng:** 35 block × `depth_per_class` layers (mặc định 2)
- **Class conditioning:** Nhận `class_id` → thêm class embedding

### 5. HeatmapHead (`detector.py`)
- **Vai trò:** Chuyển feature map thành CenterNet output
- **Output:** 3 channels:
  - Channel 0: Center heatmap (Gaussian peaks tại tâm vật thể)
  - Channel 1-2: Size map (width, height tại tâm)

## Thông Số Mô Hình

| Config | `model_dim=256` | `model_dim=512` |
|--------|-----------------|-----------------|
| Params | ~131.6M | ~500M+ |
| Per-class blocks | 35 × 2 depth | 35 × 2 depth |
| Latent size | 64×64 | 64×64 |
| Output size | 64×64 | 64×64 |

## Dataset

| Metric | Giá trị |
|--------|---------|
| Ảnh gốc (train) | 9,718 |
| Expanded samples | 44,229 (mỗi ảnh × class = 1 entry) |
| Trung bình class/ảnh | ~4.5 |
| Tổng số class | 35 |
| Image size | 512×512 (resized) |
| Output resolution | 64×64 (VAE 8x downscale) |

## Loss Functions

| Loss | Áp dụng cho | Mô tả |
|------|-------------|-------|
| Penalty-reduced Focal Loss | `center_heatmap` | Phạt nặng khi đoán sai tâm, giảm nhẹ tại vùng lân cận Gaussian |
| Masked L1 Loss | `size_map` | Chỉ tính loss kích thước tại đúng pixel tâm (dùng `mask_map`) |
