# Nhật Ký Phát Triển (Development Log)

Tài liệu này ghi lại thay đổi implementation và contract. Nó không phải experiment log: không có AP/mAP, training curve hoặc kết luận hiệu quả nào được công bố nếu chưa chạy protocol đánh giá tương ứng.

## Phase 0 — Dataset ingestion ban đầu

- Download/read FloorPlanCAD từ cấu trúc gốc gồm PNG + SVG.
- Chuyển pipeline sang đọc trực tiếp source directories; không copy ảnh và không dùng visual crop làm model input.
- Duy trì mapping 35 class canonical trong `src/data/constants.py`.

## Phase 1 — SVG metadata pipeline

Implementation active nằm ở `src/data/metadata.py` và `scripts/data/build_dataset.py`.

- Parse bbox từ SVG path bằng `svgpathtools` khi dependency có sẵn, với fallback parser cho test/simple paths.
- Tính đúng SVG `viewBox`, kể cả origin khác `(0,0)`.
- Merge thing paths theo `(semantic-id, instance-id)`.
- Tách policy cho annotation `instance-id=-1`:
  - `exclude` cho default object benchmark;
  - `merge_by_class` cho region-level experiment;
  - `path_instances` cho legacy-style pseudo instances.
- Chuẩn hóa bbox float theo `[x0,y0,x1,y1)` và clip/validate theo image bounds.

### Metadata schema v2

Metadata v2 bổ sung:

- source image/SVG SHA-256;
- parser/build settings;
- class-mapping fingerprint;
- content/build fingerprint;
- parser statistics và warnings;
- explicit schema version và bbox convention.

Builder không tự overwrite metadata hiện có. `--validate-only` không ghi file; `--force` chỉ dành cho migration được người dùng chủ động kiểm soát.

## Phase 2 — Leakage-free split contract

Implementation active nằm ở `src/data/splits.py` và `scripts/data/build_splits.py`.

- Index mỗi physical image một lần trước query expansion.
- Chia `train`/`val` chỉ từ `train_set_1` + `train_set_2`.
- Giữ toàn bộ `test_set` làm untouched test split.
- Dùng deterministic multilabel rare-class-aware strategy với seed và validation fraction được lưu.
- Ghi `splits.json` gồm image index, split records, class distribution và fingerprint.
- Validate không có image ID xuất hiện ở nhiều split.

Pipeline training hiện yêu cầu một manifest có `val` không rỗng. Cách cũ dùng `test_set` làm validation đã bị loại khỏi contract active.

## Phase 3 — Dataset và target correction

Implementation active nằm ở `src/data/dataset.py`, `src/data/transforms.py` và `src/data/targets.py`.

- `FloorPlanImageDataset` yield một image cho image-level evaluation.
- `FloorPlanQueryDataset` expand train/val image thành `(image, class)` sau khi split đã được chọn.
- Train augmentation biến đổi ảnh và bbox cùng nhau; evaluation resize deterministic.
- CenterNet target được tạo trực tiếp ở output resolution, hiện cố định stride 8.
- Query target gồm:
  - `center_heatmap [1,h,w]`;
  - `size_map [2,h,w]`;
  - `offset_map [2,h,w]`;
  - `mask_map [1,h,w]`.
- Gaussian center peak được giữ chính xác bằng `1.0` để khớp focal-loss positive rule.
- Fractional offsets được thêm để decode center chính xác hơn trong cell.
- Collision policy và `TargetStats` làm rõ trường hợp nhiều object cùng class rơi vào một output cell.

Không có hard-coded expanded-sample count trong tài liệu vì số query phụ thuộc metadata policy, source state và split manifest.

## Phase 4 — Model architecture correction

Implementation active nằm ở `src/models/`.

### Image pathway

- Thay mô tả VAE/pretrained cũ bằng `ConvImageEncoder` trainable stride 8.
- Giữ `VAEConfig`/`VAEEncoderStub` như compatibility names, không coi đó là VAE computation.
- Thêm learned 2D positional embedding cho image tokens.

### Conditioning

- `none` cho no-conditioning controls.
- learned `class_embedding` cho closed-vocabulary default.
- UTF-8 `lightweight_text` với padding mask và masked pooling.
- lazy optional `pretrained_text` qua Hugging Face, freeze mặc định.

Lightweight text được random-initialize. Fixed prompts không được mô tả như bằng chứng open-vocabulary hoặc pretrained semantic understanding.

### Fusion và routing

- Hỗ trợ `none`, additive, FiLM, image-to-text cross-attention và FiLM + cross-attention.
- Tách `shared` và `per_class` pathways thành hai trục ablation độc lập với conditioner/fusion.
- Selected-class batch được group theo class thay vì loop từng sample.
- All-class shared inference encode ảnh một lần và xử lý condition theo class chunks.

### Spatial block

- Thay tên/claim “Mamba-like” bằng `GatedSpatialMixer2D`, đúng với computation depthwise Conv2d + gate.
- Không còn claim state-space/selective-scan hoặc linear-time Mamba.
- Sửa self-attention về canonical `[B,heads,length,head_dim]` layout.
- Query-chunk fallback giảm peak memory nhưng không thay đổi quadratic attention compute.

### Detection head

- Query head active xuất 5 channels:
  - 1 center logit/probability;
  - 2 positive size values qua softplus;
  - 2 fractional offsets qua sigmoid.
- Dùng GroupNorm thay BatchNorm để tránh phụ thuộc batch statistics của class grouping.
- Thêm `SharedCenterNetBaseline` với all-class `5C` head làm project-native control.

## Phase 5 — Preset/config API

- `ModelConfig` và nested configs có serializable `to_dict()`/`from_dict()`.
- `build_model()` dispatch architecture từ config hoặc preset.
- Preset registry bao phủ shared/per-class, no condition, class embedding, lightweight text, optional pretrained text và shared CenterNet baseline.
- `train.py` dùng `--preset` làm primary model selection; các CLI model flags cũ chỉ còn là explicit overrides được validate.

Parameter count không được hard-code. Đo từ model thực tế:

```python
from src.models import build_model

model = build_model("floorplan_base")
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

## Phase 6 — Loss, reproducibility và checkpoints

- Combined loss hiện gồm CenterNet focal, masked Smooth L1 size và masked Smooth L1 offset.
- Default CLI weights là `10 / 1 / 1`; đây là config default, không phải kết quả tối ưu đã được chứng minh.
- Training seed được truyền tới Python, NumPy, PyTorch, DataLoader workers, shuffle/balanced sampler generator.
- Optional deterministic mode được expose bằng `--deterministic`.
- Linear warmup + cosine scheduler state được checkpoint.

### Checkpoint schema v2

Mỗi best/last/periodic payload dùng cùng schema và lưu:

- model state + resolved model config/preset;
- optimizer/scheduler/scaler state khi có;
- epoch, global step, best metric và metrics;
- Python/NumPy/PyTorch/DataLoader RNG state;
- class mapping/output stride;
- split manifest và metadata fingerprints;
- runtime config ảnh hưởng exact resume.

Exact resume reject mismatch thay vì silently restart. `--weights-only` là đường dẫn explicit để nạp weights và bắt đầu optimizer/schedule mới. Cả hai đường dẫn chỉ chấp nhận checkpoint schema v2; checkpoint pre-schema của kiến trúc lịch sử bị từ chối sớm vì không có migration tương thích đã được xác minh.

## Phase 7 — Evaluation infrastructure

Implementation active nằm ở `src/evaluation/` và `evaluate.py`.

- Reusable CenterNet decoder cho query/all-class layouts.
- Local-peak suppression, threshold/top-k, offset-aware xyxy decode và clipping.
- Dependency-free IoU, one-to-one matching, AP50 và AP50:95 với 101-point interpolation.
- Per-class counts và AP report.
- Versioned external prediction JSON để đánh giá YOLO/Faster R-CNN/detector khác trên cùng split/metric implementation.
- Evaluation report lưu checkpoint/prediction hash, decoder config và data/manifest provenance.

`evaluate.py --split val` dành cho development/model selection. `--split test` chỉ dành cho final held-out reporting sau khi mọi lựa chọn đã chốt.

## Phase 8 — Launcher và documentation alignment

- `setup_server.sh` chỉ setup/data verification; không start training.
- `run_train.sh` là manual future-training launcher dùng preset, manifest, seed và fixed batch size; không auto-size từ VRAM.
- Colab workflow dùng metadata v2, deterministic manifest, train/val split, preset/model API, checkpoint schema v2 và `evaluate.py`.
- README/model docs loại bỏ VAE/Mamba/pretrained claims không khớp computation và loại training epoch khỏi smoke instructions.

## Trạng thái bằng chứng

Đã có implementation và automated contract tests cho architecture, conditioning, data, targets, evaluation, reproducibility và checkpoints. Tài liệu này không khẳng định:

- một preset đã hội tụ;
- conditioned model tốt hơn baseline;
- pretrained text tốt hơn lightweight/class embedding;
- một AP/mAP cụ thể;
- một parameter/latency/memory number cố định.

Các claim tương lai phải theo `docs/research_protocol.md`, ghi metadata/split fingerprints, measured parameter count và đánh giá trên held-out data mà không tune bằng test set.
