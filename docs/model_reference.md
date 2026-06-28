# Model Reference — FloorPlanDetector

> Tài liệu kỹ thuật chi tiết cho model `FloorPlanDetector` và tất cả sub-modules.
> Source: [detector.py](file:///e:/Dat/Research/src/models/detector.py), [object_learning_block.py](file:///e:/Dat/Research/src/models/blocks/object_learning_block.py)

---

## 1. Tổng Quan

`FloorPlanDetector` là model multimodal kết hợp **ảnh bản vẽ mặt bằng** + **text query** để phát hiện vị trí và kích thước các đối tượng thuộc class được chỉ định.

### Input / Output

| Mode | Input | Shape | Mô tả |
|---|---|---|---|
| **Cả hai** | `image` | `[B, 3, 512, 512]` | Ảnh bản vẽ mặt bằng (normalized [-1, 1]) |
| **Training** | `class_ids` | `[B]` | Index class cần tính loss (0–34) |
| **Inference** | *(không cần)* | — | `class_ids=None` → chạy tất cả 35 blocks |

| Mode | Output | Shape | Mô tả |
|---|---|---|---|
| **Training** | `center_heatmap` | `[B, 1, 64, 64]` | Heatmap cho class được chọn |
| | `size_map` | `[B, 2, 64, 64]` | Kích thước (w, h) cho class được chọn |
| **Inference** | `center_heatmap` | `[B, 35, 64, 64]` | Heatmap cho tất cả 35 classes |
| | `size_map` | `[B, 70, 64, 64]` | Kích thước cho tất cả 35 classes |

> 35 class texts là **built-in** (buffer trong model), không phải input bên ngoài.

### Inference

```python
# Tìm TẤT CẢ objects trong ảnh (chạy 35 blocks)
preds = model(image)   # class_ids=None
# preds["center_heatmap"]: [1, 35, 64, 64] — mỗi channel là 1 class
# Threshold 0.3 → NMS → tâm các objects → size_map → bounding box

# Training: chỉ chạy block của class cần tính loss
preds = model(image, class_ids=torch.tensor([4]))  # chair only
```

---

## 2. Kiến Trúc Chi Tiết

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FloorPlanDetector                            │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐                              │
│  │ VAEEncoderStub│    │TextEncoderStub│                              │
│  │ [3,512,512]  │    │ [B, 32]      │                              │
│  │  → Conv×3    │    │  → Embed+Pos │                              │
│  │  → stride 2  │    │  → LayerNorm │                              │
│  └──────┬───────┘    └──────┬───────┘                              │
│         │                   │                                       │
│   [B, 16, 64, 64]    [B, 32, D]                                   │
│         │                   │                                       │
│   flatten + img_proj        │                                       │
│         │                   │                                       │
│   [B, 4096, D]              │                                       │
│         │                   │                                       │
│  ┌──────┴───────────────────┴──────┐                               │
│  │          EarlyFusion            │                               │
│  │  Cross-Attention:               │                               │
│  │    Q=text, K=image, V=image     │                               │
│  │  → mean pool text output        │                               │
│  │  → project + residual to image  │                               │
│  └──────────────┬──────────────────┘                               │
│                 │                                                    │
│           [B, 4096, D]  (fused features)                           │
│                 │                                                    │
│     ┌───────────┼────── class_id routing ──────────┐               │
│     │           │                                  │               │
│  ┌──▼──┐    ┌──▼──┐    ┌──▼──┐         ┌──▼──┐   │               │
│  │ OLB │    │ OLB │    │ OLB │  . . .  │ OLB │   │               │
│  │ [0] │    │ [4] │    │ [8] │         │ [34]│   │               │
│  │×2dep│    │×2dep│    │×2dep│         │×2dep│   │               │
│  └─────┘    └──┬──┘    └─────┘         └─────┘   │               │
│                │ (chỉ 1 block active per sample)   │               │
│     └──────────┼───────────────────────────────────┘               │
│                │                                                    │
│           [B, 4096, D]                                             │
│                │                                                    │
│           out_norm (LayerNorm)                                      │
│                │                                                    │
│           reshape → [B, D, 64, 64]                                 │
│                │                                                    │
│  ┌─────────────▼─────────────┐                                     │
│  │       HeatmapHead        │                                     │
│  │  Conv(D→256) + BN + ReLU │                                     │
│  │  Conv(256→128) + ReLU    │                                     │
│  │  Conv(128→3)             │                                     │
│  └─────────────┬─────────────┘                                     │
│                │                                                    │
│           [B, 3, 64, 64]                                           │
│           ├── ch 0: sigmoid → center_heatmap                       │
│           └── ch 1-2: ReLU  → size_map (w, h)                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Sub-Modules

### 3.1 VAEEncoderStub

> File: [detector.py](file:///e:/Dat/Research/src/models/detector.py) | Config: [config.py VAEConfig](file:///e:/Dat/Research/src/models/config.py)

**Vai trò:** Nén ảnh RGB thành latent representation, giảm spatial resolution 8×.

**Config (matching Flux 2 Klein):**

| Param | Value |
|-------|-------|
| `block_out_channels` | `[128, 256, 512, 512]` |
| `latent_channels` | 16 |
| `scaling_factor` | 0.3611 |
| `downsample` | 8× (3 stride-2 stages) |

| Layer | Input → Output |
|-------|---------|
| Conv2d(3→128, k=3, s=2, p=1) + SiLU | `[B,3,512,512]` → `[B,128,256,256]` |
| Conv2d(128→256, k=3, s=2, p=1) + SiLU | → `[B,256,128,128]` |
| Conv2d(256→512, k=3, s=2, p=1) + SiLU | → `[B,512,64,64]` |
| Conv2d(512→512, k=3, p=1) + SiLU | → `[B,512,64,64]` |
| Conv2d(512→16, k=1) | → `[B,16,64,64]` |

**Upgrade path:**
```python
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.2-klein-9B", subfolder="vae")
vae.requires_grad_(False)  # freeze
latent = vae.encode(image).latent_dist.sample()  # [B, 16, 64, 64]
```

---

### 3.2 TextEncoderStub

> File: [detector.py](file:///e:/Dat/Research/src/models/detector.py) | Config: [config.py TextEncoderConfig](file:///e:/Dat/Research/src/models/config.py)

**Vai trò:** Chuyển tokenized text thành embedding vectors, project về model_dim.

**Config (matching T5-v1.1-XXL):**

| Param | Value |
|-------|-------|
| `model_name` | `google/t5-v1_1-xxl` |
| `vocab_size` | 32128 |
| `d_model` | 4096 |
| `num_heads` | 64 |
| `num_layers` | 24 |
| `max_length` | 512 |

| Component | Shape | Mô tả |
|-----------|-------|-------|
| `nn.Embedding(32128, 4096)` | → `[B, L, 4096]` | Word embedding |
| `nn.Embedding(512, 4096)` | → `[B, L, 4096]` | Positional encoding |
| `nn.LayerNorm(4096)` | → `[B, L, 4096]` | Normalize |
| `nn.Linear(4096, model_dim)` | → `[B, L, model_dim]` | Project về internal dim |

**Forward:** `embed(ids) + pos` → `LayerNorm` → `proj` → `[B, L, model_dim]`

**Upgrade path:**
```python
from transformers import T5EncoderModel
t5 = T5EncoderModel.from_pretrained("google/t5-v1_1-xxl")
t5.requires_grad_(False)  # freeze
# + nn.Linear(4096, model_dim) trainable projection
```

---

### 3.3 EarlyFusion

> File: [detector.py L72-96](file:///e:/Dat/Research/src/models/detector.py#L72-L96)

**Vai trò:** Kết hợp thông tin text vào image features thông qua Cross-Attention.

**Cơ chế chi tiết:**

```
1. Normalize: img = LayerNorm(img_tokens)         [B, 4096, D]
              txt = LayerNorm(txt_tokens)          [B, 32, D]

2. Cross-Attention:
   Query = txt   (text hỏi: "tôi đang tìm gì?")
   Key   = img   (image trả lời: "đây là những gì tôi có")
   Value = img
   → fused: [B, 32, D]  (mỗi text token đã "nhìn" vào ảnh)

3. Mean pool: fused_mean = fused.mean(dim=1)       [B, 1, D]
   → Expand thành [B, 4096, D]

4. Residual: output = img_tokens + proj(fused_mean) [B, 4096, D]
```

**Tại sao Query=text, không phải Query=image?**
- Text biết nó đang tìm gì → nó "hỏi" ảnh
- Kết quả: mỗi text token thu thập thông tin liên quan từ ảnh
- Mean pool tạo ra một "tóm tắt" text-aware → thêm vào image features

---

### 3.4 ObjectLearningBlock (OLB)

> File: [object_learning_block.py L118-179](file:///e:/Dat/Research/src/models/blocks/object_learning_block.py#L118-L179)

**Vai trò:** Khối học chuyên biệt cho mỗi class. Mỗi block xử lý spatial features qua 3 stage.

**Cấu trúc (Pre-Norm Residual):**

```
Input x: [B, 4096, D]
  │
  ├── + class_embed(class_id)           ← Class conditioning
  │
  ├── + MambaBlock(LayerNorm(x))        ← Stage 1: Long-range sequence modeling
  │     │  in_proj → Conv1d → SiLU → gated output → out_proj
  │     │  Complexity: O(N) — linear with sequence length
  │     │  Captures: global spatial patterns (walls span entire image)
  │
  ├── + SelfAttention(LayerNorm(x))     ← Stage 2: Spatial relationships
  │     │  Multi-head attention (Q, K, V from same input)
  │     │  Complexity: O(N²) — quadratic, but captures fine details
  │     │  Captures: local object arrangements (chair near table)
  │
  └── + FFN(LayerNorm(x))              ← Stage 3: Non-linear feature mixing
        │  Linear(D→4D) → GELU → Linear(4D→D)
        │  Captures: complex feature combinations

Output: [B, 4096, D]
```

**35 blocks × 2 depth:** Tổng 70 OLB instances, mỗi class có stack riêng.

---

### 3.5 MambaBlock (SSM)

> File: [object_learning_block.py L25-66](file:///e:/Dat/Research/src/models/blocks/object_learning_block.py#L25-L66)

**Vai trò:** Selective State Space Model — xử lý chuỗi dài hiệu quả O(N).

| Component | Shape | Mô tả |
|-----------|-------|-------|
| `in_proj` | `D → 2×d_inner` | Split thành x_ssm và gate z |
| `conv1d` | `d_inner, k=4, groups=d_inner` | Depthwise convolution dọc sequence |
| `norm + gated` | `d_inner` | `LayerNorm(x_conv) * SiLU(z)` — gated activation |
| `out_proj` | `d_inner → D` | Project về dim gốc |

**Tại sao Mamba?**
- 4096 tokens (64×64 spatial) → Self-Attention O(N²) = ~16.7M operations
- Mamba O(N) = ~4K operations → **4000× nhanh hơn** cho sequence processing
- Phù hợp cho floor plan: wall/dimension_line trải dài toàn bộ ảnh

**Lưu ý:** Đây là simplified Mamba (không có CUDA selective scan kernel). Khi deploy trên GPU, thay bằng `from mamba_ssm import Mamba` cho hiệu năng tốt hơn.

---

### 3.6 SelfAttention

> File: [object_learning_block.py L71-95](file:///e:/Dat/Research/src/models/blocks/object_learning_block.py#L71-L95)

Standard multi-head scaled dot-product attention.

```
Q, K, V = Linear(x) split into num_heads
attn = softmax(Q·Kᵀ / √d_k) · V
output = Linear(concat(heads))
```

| Param | Default |
|-------|---------|
| `num_heads` | 8 |
| `head_dim` | D / 8 = 32 (khi D=256) hoặc 64 (khi D=512) |
| `dropout` | 0.1 |

---

### 3.7 HeatmapHead (CenterNet)

> File: [detector.py L101-120](file:///e:/Dat/Research/src/models/detector.py#L101-L120)

**Vai trò:** Chuyển feature map thành 3-channel CenterNet output.

```
Input:  [B, D, 64, 64]   (spatial features)
  │
  Conv2d(D→256, k=3, p=1) + BatchNorm2d(256) + ReLU
  Conv2d(256→128, k=3, p=1) + ReLU
  Conv2d(128→3, k=1)         ← no activation (raw logits)
  │
Output: [B, 3, 64, 64]
  ├── channel 0 → sigmoid → center_heatmap [B, 1, 64, 64]
  └── channel 1-2 → ReLU  → size_map       [B, 2, 64, 64]
```

---

## 4. Per-Class Routing (Forward Pass)

Mỗi forward pass nhận 1 cặp `(ảnh, text, class_id)` và route vào đúng block:

```python
# Trong forward():
outputs = []
for i in range(B):
    cid = class_ids[i].item()              # e.g., 4 (chair)
    sample = fused[i:i+1]                   # [1, 4096, D]
    for block in self.class_blocks[cid]:    # class_blocks[4] → chair pathway
        sample = block(sample, dummy_cid)
    outputs.append(sample)
x = torch.cat(outputs, dim=0)              # [B, 4096, D]
```

**Training:** Dataset expand mỗi ảnh × N classes = N samples riêng biệt. Mỗi epoch xử lý hết 44,229 cặp → tất cả 35 blocks đều được train đầy đủ.

**Inference:** Chạy 35 forward passes trên cùng 1 ảnh, mỗi lần với 1 text class khác nhau → 35 kết quả detection → gộp lại thành bounding boxes cho tất cả objects.

---

## 5. Loss Functions

> File: [train.py L28-76](file:///e:/Dat/Research/train.py#L28-L76)

### 5.1 Penalty-Reduced Focal Loss (center_heatmap)

Dùng cho Gaussian heatmap, giảm penalty cho vùng gần tâm thật.

```
L_focal = -1/N × Σ {
  log(p) × (1-p)^α     nếu target = 1 (tâm đúng)
  log(1-p) × p^α × (1-target)^β   nếu target < 1 (vùng lân cận)
}
```

| Param | Value | Ý nghĩa |
|-------|-------|---------|
| α | 2.0 | Phạt nặng hơn khi prediction sai ở positive locations |
| β | 4.0 | Giảm penalty mạnh hơn cho vùng gần tâm (Gaussian tail) |

### 5.2 Masked L1 Loss (size_map)

Chỉ tính loss tại pixel tâm vật thể (mask_map = 1).

```
L_size = Σ|pred_size - target_size| × mask / Σ mask
```

### 5.3 Combined Loss

```
L_total = 1.0 × L_focal + 0.1 × L_size
```

Size loss weight nhỏ (0.1) vì focal loss quan trọng hơn — model cần tìm đúng tâm trước, rồi mới dự đoán kích thước.

---

## 6. Thông Số & Scaling

| Config | `model_dim=256` | `model_dim=512` |
|--------|-----------------|-----------------|
| VAEEncoderStub | 0.66M | 0.66M |
| TextEncoderStub | 8.4M | 16.8M |
| img_proj | 4K | 8K |
| EarlyFusion | 0.79M | 3.15M |
| **35 × OLB (2 deep)** | **119.8M** | **479M** |
| HeatmapHead | 1.97M | 3.5M |
| **Total** | **~131.6M** | **~503M** |

> [!NOTE]
> 91% params nằm trong 35 per-class OLB. Đây là thiết kế có chủ đích — mỗi class cần đủ capacity để học pattern riêng. Nếu cần giảm params, giảm `depth_per_class` từ 2 → 1 (chia đôi OLB params).

---

## 7. Stub → Production Upgrade Path

| Component | Stub hiện tại | Production target | Frozen? |
|-----------|--------------|-------------------|---------|
| VAEEncoderStub | 3-layer Conv, random init | Flux VAE (`AutoencoderKL`) | ✅ Yes |
| TextEncoderStub | Embedding + Pos, random init | T5-Base hoặc CLIP Text | ✅ Yes |
| Tokenizer | `hash(word) % vocab_size` | T5 Tokenizer / CLIP Tokenizer | — |
| MambaBlock | Simplified (no CUDA kernel) | `mamba_ssm.Mamba` | ❌ Trainable |

**Lưu ý:** Khi thay encoder, cần thêm `nn.Linear(encoder_dim, model_dim)` projection nếu encoder output dim ≠ model_dim.
