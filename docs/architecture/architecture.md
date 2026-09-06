# Kiến trúc hệ thống

## Tổng quan

Repository triển khai một **Configurable Conditioned CenterNet**. Pathway, conditioner và fusion là ba trục độc lập để tạo baseline/ablation; per-class text architecture không còn là giả định bắt buộc.

```text
Image [B,3,H,W]
  -> ConvImageEncoder, stride 8
  -> feature [B,C,H/8,W/8]
  -> flatten + linear projection + learned 2D position
  -> image tokens [B,(H/8)*(W/8),D]

Condition
  none | class embedding | byte text | optional pretrained text
  -> condition tokens + valid-token mask + pooled condition

Image tokens + condition
  -> fusion: none | add | FiLM | cross-attention | FiLM+cross-attention
  -> pathway: shared | per-class
       GatedSpatialMixer2D
       corrected scaled-dot-product SelfAttention
       FFN
  -> GroupNorm CenterNet head
  -> center heatmap + positive size + fractional offset
```

Output stride hiện được khóa ở 8. `image_size` phải chia hết cho 8; config khác bị từ chối trước forward/target loss.

## Image encoder

`ConvImageEncoder` là CNN trainable ba stage stride-2. Nó không phải VAE, không sample latent distribution và không chứa Flux weights. Tên `VAEConfig`/`VAEEncoderStub` chỉ được giữ như compatibility alias cho code/checkpoint cũ.

## Conditioning

### None

Không truyền class/text information. Dùng cho shared CenterNet baseline và ablation `per-class no text`.

### Class embedding

Một learned embedding theo class ID. Đây là default nhẹ và rõ nhất cho closed-vocabulary 35-class detection.

### Lightweight byte text

- Tokenize UTF-8 bytes, không hash collision.
- PAD mask tường minh.
- Masked mean chỉ dùng token hợp lệ.
- Cho phép runtime text và fixed class prompt fallback.
- Random-init; không được mô tả là pretrained language understanding.

### Optional pretrained text

Lazy Hugging Face backend, truyền tokenizer attention mask và freeze backbone mặc định. Dependency nằm trong `requirements-pretrained.txt`; default tests không download model.

## Fusion

- `none`: image tokens không được điều chế.
- `add`: cộng pooled condition.
- `film`: condition sinh channel-wise gamma/beta.
- `cross_attention`: image tokens query condition tokens và dùng padding mask.
- `film_cross_attention`: FiLM trước, cross-attention sau.

## Spatial pathway

### GatedSpatialMixer2D

```text
Linear D -> 2*inner
-> content/gate split
-> content reshape về [B,C,H,W]
-> depthwise Conv2d
-> LayerNorm
-> multiply SiLU(gate)
-> Linear về D
```

Module được đặt tên theo computation thật. Nó không phải Mamba/SSM và không có disconnected SSM parameters.

### SelfAttention

Q/K/V dùng layout `[B,heads,length,head_dim]`. `scaled_dot_product_attention` normalize trên key dimension. Query-chunk fallback giảm peak attention memory nhưng compute vẫn là O(L²); tài liệu không tuyên bố linear-time attention.

### Shared và per-class routing

- Shared: một fusion/pathway stack xử lý mọi class condition.
- Per-class: một stack cho mỗi class.

Selected-class batch được group theo class ID rồi scatter về thứ tự ban đầu; không còn gọi shared head từng sample. All-class shared inference replicate condition theo class chunk, còn ảnh chỉ encode một lần.

## Detection head

Query output:

```text
center_heatmap [B,1,h,w]   = sigmoid(center logits)
size_map       [B,2,h,w]   = softplus(raw size)
offset_map     [B,2,h,w]   = sigmoid(raw offset)
```

All-class output:

```text
center_heatmap [B,C,h,w]
size_map       [B,2C,h,w]
offset_map     [B,2C,h,w]
```

Head dùng GroupNorm thay BatchNorm để không phụ thuộc batch statistics của class routing. Softplus giữ size dương và vẫn có gradient khi raw size âm.

## Unconditioned pathway control

`shared_no_condition` dùng cùng ConvImageEncoder, learned 2-D positional embedding, `GatedSpatialMixer2D → SelfAttention → FFN` stack và GroupNorm CenterNet head design như `floorplan_base`, nhưng bỏ conditioner/fusion. Head trả multi-class `5C` channels trong một pass; selected-query training chỉ gather channels của class được hỏi.

Multi-class head là bắt buộc để control hợp lệ: shared query head 1-channel không nhận class signal sẽ tạo cùng output cho mọi class và không thể học class discrimination. Control này đo hiệu ứng conditioning mà vẫn cho từng class output riêng.

## Shared CenterNet baseline

`centernet_baseline` dùng chung image encoder contract, output stride, target builder, decoder và evaluator nhưng không text/class routing. Head trả multi-class 5C channels trong một pass. Đây là project-native control baseline, không phải reproduction chính thức của CenterNet paper.

## Data flow

```text
PNG + SVG
-> canonical metadata schema v2
-> explicit thing/stuff policy
-> image-level index
-> deterministic train/val manifest + untouched test
-> expand train/val thành (image,class) queries
-> paired augmentation + resize
-> CenterNet heatmap/size/offset targets + collision stats
```

Evaluation dùng image-level dataset, reusable decoder và AP50/AP50:95. Chi tiết nằm tại:

- `docs/data_semantics.md`
- `docs/research_protocol.md`
- `docs/baselines.md`

## Loss

```text
L = 10 * focal(center)
  + 1 * SmoothL1(size at centers)
  + 1 * SmoothL1(offset at centers)
```

Target được tạo trực tiếp ở output resolution và giữ exact heatmap peak 1.0.

## Parameter accounting

Không hard-code parameter estimate trong tài liệu vì preset/pathway/conditioner thay đổi số lượng lớn. Đo trực tiếp:

```python
model = build_model("floorplan_base")
params = sum(parameter.numel() for parameter in model.parameters())
trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
```

Mọi experiment report phải lưu resolved `ModelConfig` và hai con số trên.
