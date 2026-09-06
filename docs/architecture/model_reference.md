# Model Reference — Configurable Conditioned CenterNet

Tài liệu này mô tả implementation hiện tại trong:

- `src/models/config.py`
- `src/models/factory.py`
- `src/models/detector.py`
- `src/models/conditioning.py`
- `src/models/baseline.py`
- `src/models/blocks/object_learning_block.py`

Các tên compatibility như `VAEConfig`, `VAEEncoderStub` và `TextEncoderStub` vẫn tồn tại để giảm breakage cho code/API cũ. Chúng không cam kết compatibility với checkpoint pre-schema và không có nghĩa model active dùng VAE pretrained hoặc language model pretrained.

## 1. Public construction API

Model nên được tạo từ preset/config thay vì hard-code constructor assumptions:

```python
from src.models import ModelConfig, build_model, get_model_preset

model = build_model("floorplan_base")

config = get_model_preset("per_class_no_text")
model = build_model(config)

restored_config = ModelConfig.from_dict(checkpoint["model_config"])
model = build_model(restored_config)
```

`build_model` dispatch theo `ModelConfig.architecture`:

- `floorplan_detector` -> `FloorPlanDetector`;
- `centernet_baseline` -> `SharedCenterNetBaseline`.

`ModelConfig` được validate khi tạo:

- `output_stride` hiện phải bằng `8`;
- `image_size` phải dương và chia hết cho `8`;
- `model_dim` phải chia hết cho `num_heads`;
- `pathway_mode` là `shared` hoặc `per_class`;
- model config và nested conditioner/image config có thể serialize bằng `to_dict()`.

## 2. Input/output contract

### Query mode

Input:

```text
image     [B,3,H,W]
class_ids [B]
texts     optional string/list
```

Output:

```text
center_heatmap [B,1,H/8,W/8]
size_map       [B,2,H/8,W/8]
offset_map     [B,2,H/8,W/8]
```

Mỗi sample trong batch có thể query class khác nhau. `FloorPlanDetector` group sample theo class, chạy pathway tương ứng rồi scatter output về thứ tự ban đầu.

### All-class mode

Khi bỏ `class_ids`:

```text
center_heatmap [B,C,H/8,W/8]
size_map       [B,2C,H/8,W/8]
offset_map     [B,2C,H/8,W/8]
```

Conditioned detector encode ảnh một lần. Shared pathway replicate image tokens theo bounded class chunks; per-class pathway reuse cùng encoded tokens và chạy từng specialist stack. `class_chunk_size` giới hạn số class được xử lý trong mỗi chunk nhưng không đổi output contract.

`SharedCenterNetBaseline` tạo all-class maps bằng head `5C` channels trong một pass và có thể gather query output theo `class_ids`.

## 3. Image encoder

`ConvImageEncoder` là CNN trainable của project với fixed stride 8:

```text
RGB image
  -> Conv2d stride 2 + GroupNorm + SiLU
  -> Conv2d stride 2 + GroupNorm + SiLU
  -> Conv2d stride 2 + GroupNorm + SiLU
  -> optional stride-1 stages từ block_out_channels còn lại
  -> 1x1 projection tới latent_channels
```

Với input `[B,3,H,W]`, output phải là:

```text
[B, latent_channels, H/8, W/8]
```

`FloorPlanDetector` flatten feature 2D thành tokens, project tới `model_dim`, rồi cộng learned 2D positional embedding. Nếu runtime spatial size khác preset `image_size` nhưng vẫn chia hết cho 8, positional grid được bilinear-interpolate.

Không có latent distribution, sampling, KL loss hoặc external VAE weights trong encoder này.

## 4. Conditioning backends

Mọi conditioner trả `ConditioningOutput`:

```text
tokens         [B,L,D]
attention_mask [B,L]      # True = valid token
pooled         [B,D]
```

### `none`

`NoConditioner` trả zero signal và mask rỗng. Dùng cho no-conditioning ablation.

### `class_embedding`

`ClassEmbeddingConditioner` lookup một learned vector theo `class_id`, biểu diễn như một valid token. Đây là default của `floorplan_base`.

### `lightweight_text`

`ByteTextConditioner`:

- tokenize deterministic theo UTF-8 bytes;
- byte `0..255` map tới token ID `1..256`, ID `0` dành cho padding;
- dùng learned token/position embedding, MLP projection và LayerNorm;
- masked mean chỉ pool valid tokens;
- trainable từ random initialization.

Backend này không chứa pretrained linguistic knowledge và không đủ để tự tuyên bố open-vocabulary detection.

### `pretrained_text`

`LazyHFTextConditioner`:

- giữ construction lazy: chưa import `transformers`, load tokenizer/model hoặc truy cập network;
- `materialize()` là API explicit, idempotent để register HF submodule và khởi tạo projection trước strict checkpoint load hoặc optimizer construction;
- training materialize pretrained conditioner trước khi tạo optimizer, nên `freeze_pretrained=False` backbone parameters không bị bỏ sót;
- checkpoint restore chỉ materialize khi state dict thực sự chứa `conditioner.hf_model.*`;
- truyền tokenizer attention mask vào pooling/fusion;
- freeze backbone mặc định;
- dùng `trust_remote_code=False`;
- có thể đặt revision hoặc local-files-only qua `ConditionerConfig`.

Dependencies riêng:

```bash
python -m pip install -r requirements-pretrained.txt
```

Weights phải có trong local cache hoặc được download khi runtime cho phép. Core/lightweight setup không cần dependency này và không tự download model.

## 5. Fusion

`EarlyFusion` nhận image tokens `X` và `ConditioningOutput`.

| Mode | Hành vi |
|---|---|
| `none` | Trả image tokens không điều chế |
| `add` | Project pooled condition rồi cộng vào mọi image token |
| `film` | Sinh channel-wise `gamma`, `beta`; áp dụng `X * (1 + gamma) + beta` |
| `cross_attention` | Image tokens query condition tokens, có padding mask |
| `film_cross_attention` | FiLM trước, image-to-condition cross-attention sau |
| `current` | Compatibility mode: condition tokens query image tokens, mean-pool rồi broadcast |

No-condition rows được xử lý an toàn: empty mask không tạo NaN và không thêm condition delta.

## 6. Shared và per-class pathways

### Shared

Một `EarlyFusion` và một stack `ObjectLearningBlock` dùng chung cho mọi class. Class/text condition là tín hiệu phân biệt query.

### Per-class

Mỗi class có:

- một `EarlyFusion` riêng;
- một stack `ObjectLearningBlock` riêng.

Per-class routing tự tăng specialist capacity. Vì vậy muốn kết luận text có ích phải so `per_class + text` với `per_class + no text`, không chỉ so với shared baseline nhỏ hơn.

## 7. ObjectLearningBlock

Mỗi block là pre-norm residual stack:

```text
x = x + GatedSpatialMixer2D(LayerNorm(x))
x = x + SelfAttention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

### GatedSpatialMixer2D

```text
Linear D -> 2 * inner
  -> split content, gate
  -> content reshape [B,inner,H,W]
  -> depthwise Conv2d
  -> flatten + LayerNorm
  -> multiply SiLU(gate)
  -> Linear inner -> D
```

Đây là gated depthwise 2D convolution. Nó không phải Mamba, selective scan hay state-space model. Mọi learned parameter đều tham gia forward path.

### SelfAttention

Q/K/V được reshape theo canonical layout `[B,heads,length,head_dim]`. PyTorch scaled-dot-product attention được dùng khi khả dụng. Query-chunk fallback giảm peak memory nhưng vẫn tính attention với mọi key; computational complexity vẫn là `O(L^2)`.

### FFN

```text
Linear D -> 4D -> GELU -> dropout -> Linear 4D -> D -> dropout
```

## 8. Detection heads

### Conditioned query head

`HeatmapHead` dùng:

```text
Conv3x3 -> GroupNorm -> SiLU
Conv3x3 -> GroupNorm -> SiLU
Conv1x1 -> 5 raw channels
```

Activations:

```text
center_heatmap = sigmoid(raw[:,0:1])
size_map       = softplus(raw[:,1:3])
offset_map     = sigmoid(raw[:,3:5])
```

- heatmap nằm trong `[0,1]`;
- size dương;
- fractional offsets nằm trong `[0,1]` về mặt activation contract.

GroupNorm tránh phụ thuộc batch statistics khi selected-class batch bị group theo class.

### Shared CenterNet baseline head

Baseline head xuất `5C` raw channels, reshape thành `[B,C,5,h,w]`, rồi áp dụng cùng sigmoid/softplus/sigmoid contract. Query mode gather class-specific maps từ all-class tensor.

## 9. Data and target contract

Training dùng `FloorPlanQueryDataset`: image-level split được chọn trước, sau đó mỗi image được expand thành một query cho mỗi class có mặt trong image.

Target được sinh trực tiếp ở output resolution:

```text
center_heatmap [1,h,w]
size_map       [2,h,w]
offset_map     [2,h,w]
mask_map       [1,h,w]
```

Với bbox input-pixel `[x0,y0,x1,y1)`:

- bbox được scale sang output grid;
- center cell là `floor(center_float)`;
- size lưu theo output-cell units;
- offset lưu phần lẻ của center;
- Gaussian peak tại center được giữ chính xác bằng `1.0`;
- regression chỉ tính ở center cells có `mask_map=1`.

Nếu nhiều object cùng class rơi vào một output cell, default collision policy `largest` giữ size/offset của bbox có area lớn hơn. Heatmap vẫn max-composite; `TargetStats` ghi collision/replacement/ignored counts.

## 10. Loss

Implementation trong `src/training/losses.py`:

```text
L = focal_weight  * CenterNetFocal(center)
  + size_weight   * SmoothL1(size at masked centers)
  + offset_weight * SmoothL1(offset at masked centers)
```

Default CLI weights:

```text
focal_weight  = 10
size_weight   = 1
offset_weight = 1
```

`focal_loss` yêu cầu target có exact `1.0` positives ở cùng spatial resolution với prediction. Target builder đáp ứng contract này trực tiếp, không tạo full-resolution heatmap rồi bilinear-downsample.

## 11. Decoder và evaluation

`CenterNetDecoder`:

1. áp dụng local-maximum suppression bằng max-pool;
2. giữ scored peaks theo threshold và top-k per class;
3. đọc size/offset ở peak cell;
4. tính center trong input-pixel coordinates bằng output stride;
5. tạo và clip xyxy boxes;
6. trả per-image `boxes`, `scores`, `labels`.

`evaluate.py` chạy image-level dataset và tính AP50/AP50:95. Validation loss trong `train.py` không thay thế detection metrics.

## 12. Checkpoint construction

Checkpoint mới lưu `model_config` hoàn chỉnh. Cách load đúng:

```python
from src.data.constants import CLASS_NAMES
from src.models import ModelConfig, build_model
from src.training.checkpoint import load_checkpoint, restore_training_state

checkpoint = load_checkpoint("checkpoints/best.pt", map_location="cpu")
config = ModelConfig.from_dict(checkpoint["model_config"])
model = build_model(config)
restore_training_state(
    checkpoint,
    model=model,
    expected_model_config=config,
    expected_class_names=CLASS_NAMES,
    expected_output_stride=config.output_stride,
    weights_only=True,
    strict=True,
)
model.eval()
```

Không reconstruct model mới bằng một tập constructor arguments phỏng đoán nếu checkpoint đã có serializable config. Checkpoint pre-schema từ kiến trúc lịch sử bị từ chối sớm bằng `CheckpointError`: state dict đó không tương thích với kiến trúc hiện tại và không có migration tự động đã được xác minh.

## 13. Parameter accounting

Không hard-code parameter table trong tài liệu. Preset, architecture, pathway, depth, conditioner và optional pretrained backend đều làm count thay đổi.

```python
from src.models import build_model

model = build_model("floorplan_base")
total = sum(parameter.numel() for parameter in model.parameters())
trainable = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)
print(f"total={total:,} trainable={trainable:,}")
```

Mọi experiment report phải ghi:

- preset name;
- resolved `model.config.to_dict()`;
- measured total/trainable parameter count;
- input size/output stride;
- conditioner preload/freeze state;
- latency/memory nếu có đo thực tế.

## 14. Claims boundary

Implementation cho phép kiểm tra các giả thuyết về class conditioning, fixed text, pretrained text và specialist pathways. Repository không tự chứng minh:

- spatial mixer tốt hơn Mamba/SSM;
- text-conditioned preset tốt hơn no-text baseline;
- random-init byte text hiểu semantic ngoài prompt đã train;
- per-class architecture tốt hơn sau khi kiểm soát parameter budget;
- bất kỳ AP/mAP value nào trước khi evaluator thực sự chạy trên manifest được công bố.
