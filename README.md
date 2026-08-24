# FloorPlanCAD Object Detection

Repository này triển khai các baseline **CenterNet stride 8** cho FloorPlanCAD, gồm một detector có thể condition theo class/text và một shared CenterNet control baseline. Code hiện tại dùng CNN trainable của project, `GatedSpatialMixer2D`, self-attention và FFN; không dùng Mamba/SSM, không dùng VAE pretrained và không mặc định có pretrained language semantics.

## Contracts chính

- Input luôn là ảnh floor plan đầy đủ; không dùng ground-truth crop làm input.
- Metadata canonical là **schema v2**, có source SHA-256, build settings, class mapping và fingerprint.
- Split được tạo ở **image level** trước khi expand thành `(image, class)` query.
- `train` và `val` lấy từ `train_set_1` + `train_set_2`; `test_set` được giữ nguyên cho đánh giá cuối.
- Model và target hiện chỉ hỗ trợ `output_stride=8`.
- Query head trả 5 channels: center `1`, size `2`, fractional offset `2`.
- Validation loss dùng để chọn checkpoint; detection report dùng AP50 và AP50:95.
- Checkpoint schema v2 lưu model config, optimizer/scheduler/RNG state và data fingerprints để resume có kiểm tra.

## Kiến trúc

```text
Image [B,3,H,W]
  -> ConvImageEncoder, stride 8
  -> image tokens + learned 2D positional embedding

Condition
  none | learned class embedding | lightweight UTF-8 byte text
       | optional Hugging Face pretrained text
  -> condition tokens + valid-token mask + pooled condition

Image tokens + condition
  -> fusion: none | add | FiLM | cross-attention | FiLM+cross-attention
  -> pathway: shared | per-class
       GatedSpatialMixer2D
       scaled-dot-product SelfAttention
       FFN
  -> GroupNorm CenterNet query head [5 channels]
       center_heatmap [B,1,H/8,W/8]
       size_map       [B,2,H/8,W/8]
       offset_map     [B,2,H/8,W/8]
```

Khi inference tất cả class, conditioned detector encode ảnh một lần rồi xử lý class theo chunk. Output được ghép thành:

```text
center_heatmap [B,C,H/8,W/8]
size_map       [B,2C,H/8,W/8]
offset_map     [B,2C,H/8,W/8]
```

`centernet_baseline` dùng shared CNN pathway và một head `5C` channels trong một pass. Đây là project-native control baseline, không phải reproduction chính thức của CenterNet paper.

Chi tiết:

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/model_reference.md`](docs/model_reference.md)
- [`docs/model_math_deep_dive.md`](docs/model_math_deep_dive.md)
- [`docs/design_philosophy.md`](docs/design_philosophy.md)

## Dataset và metadata v2

Dataset active được đọc trực tiếp từ cấu trúc gốc; builder không copy ảnh.

```text
data/FloorPlanCAD_original/
  train_set_1/
    sample.png
    sample.svg
    sample_meta.json
  train_set_2/
  test_set/
  splits.json
```

Metadata v2 lưu bbox float theo convention `[x0, y0, x1, y1)`, source hashes, parser settings, `stuff_policy`, `unknown_policy`, statistics và fingerprint. Default object benchmark loại các annotation `instance-id=-1` bằng `--stuff-policy exclude`. Unknown semantic IDs mặc định bị bỏ với warning/statistics (`--unknown-policy warn`); dùng `--unknown-policy error` hoặc `--strict` khi muốn từ chối source có annotation không hợp lệ. Các policy khác tạo benchmark khác và không nên trộn kết quả.

### Tạo metadata lần đầu

```bash
python scripts/data/build_dataset.py \
  --data-root ./data/FloorPlanCAD_original \
  --stuff-policy exclude
```

Builder bỏ qua metadata đã tồn tại. Nó chỉ ghi đè khi người dùng chủ động thêm `--force`; không dùng flag đó cho verification thông thường.

### Kiểm tra metadata mà không ghi file

```bash
python scripts/data/build_dataset.py \
  --data-root ./data/FloorPlanCAD_original \
  --stuff-policy exclude \
  --validate-only
```

### Tạo deterministic split manifest

```bash
python scripts/data/build_splits.py \
  --data-root ./data/FloorPlanCAD_original \
  --output ./data/FloorPlanCAD_original/splits.json \
  --seed 1337 \
  --val-fraction 0.10
```

Manifest lưu image index, train/val/test image IDs, class distribution, seed, strategy và fingerprint. Script từ chối thay manifest đang có nếu không truyền `--force`.

Kiểm tra split computation trong memory, không ghi file:

```bash
python scripts/data/build_splits.py \
  --data-root ./data/FloorPlanCAD_original \
  --seed 1337 \
  --val-fraction 0.10 \
  --validate-only
```

Xem thêm [`docs/data_semantics.md`](docs/data_semantics.md) và [`docs/research_protocol.md`](docs/research_protocol.md).

## Cài đặt

Cài PyTorch build phù hợp với CUDA/CPU trước, sau đó cài core dependencies:

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
# Ví dụ CUDA; chọn index URL phù hợp với máy của bạn.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Development tests:

```bash
python -m pip install -r requirements-dev.txt
```

Pretrained text là **optional**. Chỉ cài khi dùng preset có `conditioner.kind=pretrained_text`:

```bash
python -m pip install -r requirements-pretrained.txt
```

Preset pretrained lazy-load Hugging Face tokenizer/model ở forward đầu tiên. Nó cần weights đã cache hoặc quyền truy cập mạng; core tests và lightweight presets không download model.

Trên server Linux có thể dùng `bash setup_server.sh`. Script setup cài dependencies, chuẩn bị/kiểm tra metadata và manifest, rồi chạy verification; nó không start training.

## Model presets

Preset được resolve thành `ModelConfig` serializable qua `src.models.build_model`.

| Preset | Architecture | Pathway | Conditioner | Fusion |
|---|---|---|---|---|
| `floorplan_base` | conditioned detector | shared | class embedding | FiLM |
| `shared_no_condition` | unconditioned pathway control (multi-class 5C head) | shared | none | none |
| `shared_wide` | parameter-matched unconditioned pathway control | shared | none | none |
| `per_class_small` | parameter-matched specialist control | per-class | none | none |
| `per_class_no_text` | conditioned detector | per-class | none | none |
| `shared_fixed_byte_text_film` | conditioned detector | shared | random-init byte text | FiLM |
| `per_class_fixed_byte_text_film` | conditioned detector | per-class | random-init byte text | FiLM |
| `shared_fixed_byte_text_cross_attention` | conditioned detector | shared | random-init byte text | cross-attention |
| `shared_pretrained_text` | conditioned detector | shared | optional pretrained text | FiLM |
| `per_class_pretrained_text` | conditioned detector | per-class | optional pretrained text | FiLM |
| `centernet_baseline` / `centernet_shared` | shared CenterNet baseline | shared | none | none |

`floorplan_tiny` là preset nhỏ cho development/smoke. Lightweight text là UTF-8 byte encoder trainable từ random initialization; không được mô tả như open-vocabulary pretrained semantics.

`shared_no_condition` không dùng query head class-agnostic. Nó giữ ConvImageEncoder, positional embedding và GatedSpatialMixer/SelfAttention/FFN pathway như `floorplan_base`, nhưng dùng multi-class `5C` head để mỗi class có output riêng. Đây là control trực tiếp hợp lệ cho hiệu ứng conditioning; một query head 1-channel không có class signal sẽ không thể biết class đang được hỏi.

Liệt kê preset từ code:

```bash
python -c "from src.models import list_model_presets; print('\n'.join(list_model_presets()))"
```

### Đo parameter count

Không copy một estimate cố định vì architecture, pathway, conditioner và preset thay đổi số lượng tham số. Đo trực tiếp từ config thực sự dùng:

```python
from src.models import build_model

model = build_model("floorplan_base")
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print({"total": total, "trainable": trainable})
```

Experiment report phải lưu resolved `model.config.to_dict()` cùng hai số đo này.

## Verification không training

Các command sau không chạy training epoch và không gọi `optimizer.step()`:

```bash
python -m compileall -q train.py evaluate.py src scripts tests
python -m pytest -q
python scripts/dev/smoke_models.py \
  --device cpu \
  --image-size 32 \
  --model-dim 16 \
  --depth 1 \
  --all-lightweight-presets
```

Dataset read-only checks nằm trong [`docs/verification.md`](docs/verification.md). Không dùng `python train.py --epochs 1` như smoke test vì command đó thực sự cập nhật weights và ghi checkpoint.

## Training launcher cho run tương lai

Training yêu cầu manifest có non-empty `val` split. Test không được dùng trong training/validation.

```bash
python train.py \
  --data-root ./data/FloorPlanCAD_original \
  --manifest ./data/FloorPlanCAD_original/splits.json \
  --preset floorplan_base \
  --seed 1337 \
  --ckpt-dir ./checkpoints \
  --batch-size 4 \
  --num-workers 4 \
  --precision bf16 \
  --epochs 50 \
  --lr 1e-5 \
  --focal-weight 10 \
  --size-weight 1 \
  --offset-weight 1 \
  --warmup-steps 500 \
  --sampler balanced \
  --balance-power 0.5
```

`--precision` hỗ trợ `fp32`, `bf16`, `fp16`, hoặc `auto`. Với pathway attention ở 512×512, BF16 được khuyến nghị trên GPU hỗ trợ để dùng scaled-dot-product attention hiệu quả hơn và giảm VRAM; FP16 dùng GradScaler, còn BF16 không cần scaler. Precision được lưu trong runtime config/checkpoint để exact resume kiểm tra tương thích.

`run_train.sh` cung cấp cùng contract trong một tmux launcher và dùng batch size cố định/conservative; script không tự đo VRAM để thay batch size. Chỉnh biến đầu file hoặc environment trước khi chủ động chạy.

### Resume và weights-only

Exact resume khôi phục model, optimizer, scheduler, RNG và DataLoader generator state. Nó từ chối run có model/data/runtime fingerprints không tương thích:

```bash
python train.py \
  --data-root ./data/FloorPlanCAD_original \
  --manifest ./data/FloorPlanCAD_original/splits.json \
  --preset floorplan_base \
  --seed 1337 \
  --resume ./checkpoints/last.pt
```

Để chỉ nạp weights và bắt đầu optimizer/schedule mới:

```bash
python train.py \
  --data-root ./data/FloorPlanCAD_original \
  --manifest ./data/FloorPlanCAD_original/splits.json \
  --preset floorplan_base \
  --seed 1337 \
  --resume ./checkpoints/best.pt \
  --weights-only
```

Checkpoint schema v2 lưu cùng payload contract cho `best.pt`, `last.pt` và periodic checkpoints: resolved model config, preset, class mapping, output stride, epoch/global step, optimizer/scheduler state, RNG state, metrics và split/metadata fingerprints.

Checkpoint pre-schema từ kiến trúc lịch sử không được hỗ trợ bởi exact resume, `--weights-only`, evaluation hoặc visualization. Kiến trúc và state-dict keys đã thay đổi, không có artifact migration được xác minh, nên loader từ chối sớm với `CheckpointError` thay vì thử đoán config rồi báo raw missing/unexpected keys. Hãy dùng checkpoint schema v2 được tạo bởi kiến trúc hiện tại.

## Evaluation

`evaluate.py` chạy image-level inference/metrics, không training. Nó hỗ trợ checkpoint nội bộ hoặc prediction JSON từ detector bên ngoài.

### Validation report từ checkpoint

Dùng `val` để kiểm tra decoder và so sánh trong quá trình phát triển:

```bash
python evaluate.py \
  --data-root ./data/FloorPlanCAD_original \
  --manifest ./data/FloorPlanCAD_original/splits.json \
  --split val \
  --checkpoint ./checkpoints/best.pt \
  --report ./outputs/evaluation_val.json \
  --save-predictions ./outputs/predictions_val.json \
  --image-size 512 \
  --batch-size 1 \
  --class-chunk-size 4
```

### Held-out test policy

Chỉ chạy test sau khi preset, checkpoint, threshold, top-k và mọi hyperparameter đã được chốt bằng train/val. Không dùng test AP/loss để chọn checkpoint hoặc tune decoder.

```bash
python evaluate.py \
  --data-root ./data/FloorPlanCAD_original \
  --manifest ./data/FloorPlanCAD_original/splits.json \
  --split test \
  --checkpoint ./checkpoints/best.pt \
  --report ./outputs/evaluation_test.json \
  --image-size 512
```

Evaluator báo AP50, AP50:95, per-class counts/AP, decoder settings, checkpoint hash và data/manifest fingerprints. AP implementation dùng IoU sweep 0.50:0.05:0.95 và 101-point interpolation; nó không implement đầy đủ mọi COCO area/crowd/max-detections variant.

### External prediction JSON

External model có thể xuất schema model-agnostic sau:

```json
{
  "format": "object_detection_predictions",
  "schema_version": 1,
  "box_format": "xyxy",
  "coordinate_space": "image",
  "predictions": [
    {
      "image_id": "test_set/example",
      "boxes": [[10.0, 20.0, 50.0, 70.0]],
      "scores": [0.92],
      "labels": [4]
    }
  ]
}
```

`image_id` phải khớp manifest; boxes nằm trong coordinate space của input đã resize theo `--image-size`; labels dùng index của `CLASS_NAMES`.

```bash
python evaluate.py \
  --data-root ./data/FloorPlanCAD_original \
  --manifest ./data/FloorPlanCAD_original/splits.json \
  --split test \
  --predictions-json ./outputs/external_predictions.json \
  --report ./outputs/external_evaluation_test.json \
  --image-size 512
```

Xem thêm [`docs/baselines.md`](docs/baselines.md).

## Project structure

```text
├── train.py                         # Train/val query loop + schema-v2 checkpoints
├── evaluate.py                      # Checkpoint/external JSON detection evaluation
├── run_train.sh                     # Manual future-training tmux launcher
├── setup_server.sh                  # Environment/data verification; no training
├── colab_train.ipynb                # Manual Colab workflow
├── requirements.txt
├── requirements-dev.txt
├── requirements-pretrained.txt
├── src/
│   ├── data/
│   │   ├── metadata.py              # Metadata schema v2
│   │   ├── splits.py                # Deterministic image-level manifest
│   │   ├── dataset.py               # Image-level + expanded query datasets
│   │   ├── targets.py               # Stride-8 CenterNet targets/collision stats
│   │   └── transforms.py
│   ├── models/
│   │   ├── config.py                # ModelConfig + preset registry
│   │   ├── factory.py               # build_model
│   │   ├── conditioning.py
│   │   ├── detector.py
│   │   ├── baseline.py
│   │   └── blocks/object_learning_block.py
│   ├── evaluation/                  # Decoder, AP metrics, prediction JSON I/O
│   └── training/                    # Losses, reproducibility, checkpoints
├── scripts/
│   ├── data/build_dataset.py
│   ├── data/build_splits.py
│   └── dev/smoke_models.py
├── docs/
└── tests/
```

## Research status

Repository cung cấp implementation, contracts và evaluation protocol. Nó không kèm claim rằng conditioned variants tốt hơn baseline, không công bố mAP chưa đo, và không suy diễn pretrained/open-vocabulary capability từ fixed prompts hoặc random-init text encoder. Mọi kết luận phải dựa trên cùng metadata, split manifest, decoder và held-out evaluation protocol.
