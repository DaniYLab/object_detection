# Đánh giá nghiên cứu FloorPlanCAD Object Detection

**Ngày đánh giá:** 2026-07-23  
**Phạm vi:** Working tree hiện tại của repository  
**Trạng thái:** Implementation và research protocol; chưa có kết quả thực nghiệm chính thức

## 1. Kết luận tổng quan

Repository là một **research scaffold được xây dựng khá nghiêm túc**, với metadata versioned, split chống leakage, target/decoder nhất quán, checkpoint có fingerprint và protocol ablation rõ ràng. Tuy nhiên, nó **chưa phải một nghiên cứu đã có bằng chứng thực nghiệm** và hiện còn một số confound quan trọng cần xử lý trước khi chạy training quy mô lớn.

Tóm tắt:

- **Engineering contracts:** tốt.
- **Tính kỷ luật trong claims:** tốt; tài liệu không bịa AP/mAP và không gọi byte encoder là open-vocabulary.
- **Ablation hiện tại:** chưa đủ sạch để cô lập tác dụng của conditioning.
- **Training design:** có một lỗ hổng lớn về negative class queries.
- **Bằng chứng thực nghiệm:** chưa có checkpoint, split manifest, logs hay AP report trong repository.
- **Khuyến nghị:** chưa nên chạy toàn bộ ma trận 50 epoch; nên sửa các vấn đề P0, chạy data audit và pilot nhỏ trước.

## 2. Kiểm chứng đã thực hiện

Trên working tree tại thời điểm đánh giá:

- `python -m compileall -q train.py evaluate.py src scripts tests`: thành công.
- `python -m pytest -q`: **84 passed in 6.45s**.
- Model smoke: **12 lightweight presets passed**.
- Không tìm thấy checkpoint, training log, evaluation report, split manifest hoặc dataset artifact trong repository.
- Test suite chỉ dùng fixture/tensor nhỏ, không chạy epoch và không gọi `optimizer.step()`, đúng với phạm vi mô tả trong `docs/verification.md`.

Repository cũng tự giới hạn claims một cách đúng đắn: nó cung cấp implementation, contracts và evaluation protocol nhưng chưa tuyên bố conditioned variants tốt hơn baseline hoặc công bố mAP chưa đo.

## 3. Câu hỏi nghiên cứu

Giả thuyết chính là:

> Class ID hoặc text conditioning có giúp detector tập trung vào đúng loại đối tượng trong floor plan tốt hơn detector không conditioning hay không?

Ba trục được thiết kế để ablate độc lập:

### 3.1 Pathway

- Shared.
- Per-class specialist.

### 3.2 Conditioner

- None.
- Learned class embedding.
- Random-init UTF-8 byte encoder.
- Pretrained Hugging Face encoder.

### 3.3 Fusion

- None.
- Additive.
- FiLM.
- Cross-attention.
- FiLM + cross-attention.

Các câu hỏi được thiết kế theo hướng falsifiable. Tài liệu cũng nhận diện đúng rằng lợi ích của per-class routing không được quy cho text nếu chưa có control tương ứng.

## 4. Luồng dữ liệu đầy đủ

### 4.1 PNG/SVG → metadata v2

`scripts/data/build_dataset.py` và `src/data/metadata.py` thực hiện:

1. Đọc PNG và SVG tương ứng.
2. Parse SVG `viewBox`.
3. Nhóm các path theo `(semantic-id, instance-id)`.
4. Áp dụng `stuff_policy`, `unknown_policy` và `min_size`.
5. Chuyển bbox từ SVG coordinate sang pixel coordinate.
6. Clip bbox về image bounds.
7. Ghi source SHA-256, build settings, class mapping fingerprint, parser statistics và canonical bbox `[x0,y0,x1,y1)`.

Điểm tốt:

- Có xử lý non-zero SVG viewBox origin.
- Phân biệt thing/stuff rõ ràng.
- Metadata có thể phân biệt benchmark được build bằng policy khác nhau.
- File được ghi atomically.

### 4.2 Metadata → image-level split manifest

`src/data/splits.py`:

1. Index mỗi physical image đúng một lần.
2. Train pool chỉ lấy từ `train_set_1` và `train_set_2`.
3. `test_set` được giữ nguyên.
4. Tạo validation bằng rare-class-aware greedy multilabel split.
5. Kiểm tra không có image leakage.
6. Lưu class distribution và fingerprint.

Đây là một trong các phần mạnh nhất của repository: query expansion diễn ra **sau** split, nên cùng một floor plan không thể đi vào cả train và validation.

### 4.3 Image records → query dataset

Training dùng `FloorPlanQueryDataset`:

```text
một image có class A, B, C
→ (image, A)
→ (image, B)
→ (image, C)
```

Mỗi query:

1. Lọc bbox chỉ giữ class đang được hỏi.
2. Augment image và bbox cùng nhau.
3. Resize về kích thước cố định.
4. Tạo CenterNet target cho class đó.
5. Trả về fixed prompt, ví dụ `Find chair in this floor plan drawing`.

### 4.4 Bbox → CenterNet targets

Box được scale trực tiếp sang output grid stride 8:

```text
box center float
→ floor(center) thành output cell
→ fractional remainder thành offset
→ width/height trong output-grid units
→ Gaussian heatmap quanh center
```

Output:

```text
center_heatmap [1,h,w]
size_map       [2,h,w]
offset_map     [2,h,w]
mask_map       [1,h,w]
```

Nếu hai bbox cùng class rơi vào cùng output cell:

- Heatmap vẫn được max-composite.
- Chỉ một size/offset target được giữ.
- Mặc định giữ box có area lớn hơn.
- Collision được log.

Đây là limitation có chủ đích và đã được tài liệu hóa.

### 4.5 Query → model

```text
Image
→ ConvImageEncoder, stride 8
→ flatten thành N spatial tokens
→ linear projection + learned positional embedding

Class/text
→ conditioner tokens + mask + pooled embedding

Image tokens + condition
→ fusion
→ shared/per-class ObjectLearningBlock stack
→ GroupNorm CenterNet head
→ heatmap + size + offset
```

Mỗi `ObjectLearningBlock` gồm:

```text
GatedSpatialMixer2D
→ SelfAttention
→ FFN
```

### 4.6 Loss → optimizer → checkpoint

Loss:

```text
L = 10 × focal(center)
  + 1 × SmoothL1(size)
  + 1 × SmoothL1(offset)
```

Training loop:

```text
batch
→ selected-class forward
→ loss
→ backward
→ gradient clipping
→ AdamW step
→ scheduler step
```

Checkpoint lưu model, optimizer, scheduler, RNG, config và data fingerprints.

### 4.7 Image-level inference → AP

Evaluation không dùng query dataset. Mỗi image được đọc một lần và model sinh output cho toàn bộ class:

```text
image
→ all-class model output
→ local peak suppression
→ top-k per class
→ offset/size box decoding
→ AP50/AP50:95
```

## 5. Những điểm làm tốt

### 5.1 Claims discipline

Repository phân biệt rõ:

- Gated convolution không phải Mamba/SSM.
- CNN encoder không phải VAE pretrained.
- Byte text encoder không có pretrained semantics.
- Fixed class prompts không chứng minh open-vocabulary detection.

Đây là điểm tích cực đối với một repository nghiên cứu.

### 5.2 Target, decoder và box conventions nhất quán

- Bbox là continuous half-open `xyxy`.
- Size/offset được tạo trực tiếp ở output resolution.
- Gaussian center được ép chính xác về `1.0`, phù hợp với `target.eq(1)` trong focal loss.
- Regression channel layout giữa conditioned model, baseline và decoder là nhất quán.

### 5.3 Data leakage được kiểm soát tốt

Image-level split trước query expansion là lựa chọn đúng. Manifest cũng kiểm tra:

- Complete partition.
- Source provenance.
- Metadata summary/fingerprint.
- Test source không lọt vào train/val.

### 5.4 Checkpoint contract

Checkpoint lưu:

- Model state.
- Optimizer/scheduler state.
- RNG Python, NumPy, CPU/CUDA.
- DataLoader generator state.
- Model config fingerprint.
- Class mapping.
- Split/metadata fingerprints.

### 5.5 Component test coverage

Test suite kiểm tra metadata, splits, transforms, targets, attention, fusion, model contracts, losses, decoder, AP, prediction JSON và checkpoint round-trip.

Tuy nhiên, đây vẫn là **component/synthetic verification**, không phải bằng chứng model train được hoặc hội tụ.

## 6. Các vấn đề quan trọng

### P0 — Training chỉ tạo positive class queries

Đây là vấn đề quan trọng nhất.

Query dataset chỉ tạo `(image, class)` đối với class **đã xuất hiện trong image**. Do đó:

- Query `chair` chỉ được train trên những image đã có chair.
- Model không nhận query `chair` trên image không có chair với empty heatmap.
- Baseline cũng chỉ gather head của class hiện diện trong sample.
- Trong evaluation, model lại được query cho tất cả 35 class trên mọi image.

Hệ quả có thể xảy ra:

- False positive cao cho absent classes.
- Heatmap score calibration không tốt.
- Validation loss không nhìn thấy lỗi hallucination theo class.
- Balanced sampler chỉ cân bằng positive queries, không giải quyết negative evidence.

#### Khuyến nghị

Sample thêm absent-class queries:

```text
mỗi positive query
+ K class âm không có trong image
→ empty heatmap/size/offset target
```

Focal loss đã có zero-positive branch, nhưng cần xem lại weighting: negative-only sample hiện dùng mean, trong khi positive sample dùng negative sum chia cho số positive. Hai scale này chưa tương thích để sử dụng trực tiếp cho hard-negative sampling.

Nên thêm metric:

- False-positive rate cho class absent khỏi image.
- AP khi chia image thành present/absent query groups.
- Score distribution positive so với absent queries.

### P0 — Baseline chưa cô lập tác dụng của conditioning

Protocol muốn so:

```text
shared no-conditioning
vs
shared class-embedding + FiLM
```

Nhưng preset `centernet_baseline` dùng residual convolutional backbone và multi-class `5C` head trong một pass. `floorplan_base` lại dùng GatedSpatialMixer2D, self-attention, FFN và query-conditioned five-channel head.

Registry hiện chưa có preset:

```text
architecture = floorplan_detector
pathway = shared
conditioner = none
fusion = none
```

Do đó, nếu `floorplan_base` tốt hơn `centernet_baseline`, chưa thể biết lợi ích đến từ:

- Class embedding.
- FiLM.
- Self-attention.
- Gated mixer.
- Query head thay vì multi-class head.
- Khác biệt capacity/optimization.

#### Khuyến nghị

Thêm hai baseline riêng:

1. **Architecture control**

```text
FloorPlanDetector
+ shared pathway
+ NoConditioner
+ fusion none
```

2. **Project-native CNN baseline**

```text
SharedCenterNetBaseline
```

Baseline thứ nhất mới là control trực tiếp cho conditioning. Baseline thứ hai trả lời câu hỏi rộng hơn về toàn bộ kiến trúc.

### P0 — Per-class variants bị confound bởi parameter count

Parameter count đo trực tiếp từ code tại thời điểm đánh giá:

| Preset | Parameters |
|---|---:|
| `centernet_baseline` | 3,674,031 |
| `floorplan_base` | 5,268,741 |
| `shared_fixed_byte_text_film` | 5,400,197 |
| `shared_fixed_byte_text_cross_attention` | 5,335,941 |
| `per_class_no_text` | 85,707,781 |
| `per_class_fixed_byte_text_film` | 99,664,517 |

Per-class pathway nhân toàn bộ fusion/block stack theo số class. Vì vậy:

- `per_class_no_text` lớn hơn baseline khoảng 23 lần.
- `per_class_fixed_byte_text_film` gần 100 triệu parameters.
- So per-class với shared không thể kết luận specialist routing tốt hơn nếu chưa có budget-matched control.

#### Khuyến nghị

So sánh theo hai trục:

- **Unconstrained capacity:** kiến trúc nào đạt AP cao nhất.
- **Matched budget:** giữ parameters/FLOPs gần tương đương.

Có thể thêm:

- Shared-wide model khoảng 85–100M parameters.
- Per-class-small với model dimension/depth giảm mạnh.
- Efficiency curves: AP theo parameter count, latency và memory.

### P1 — Chi phí attention/all-class inference cao

Với input `512×512` và stride 8:

```text
latent grid = 64×64
N = 4096 tokens
N² = 16,777,216 query-key pairs/head/block
```

Đây mới chỉ là một class query. Conditioned all-class inference encode image một lần nhưng vẫn chạy fusion và ObjectLearningBlock cho từng class/chunk.

Như vậy:

- Baseline sinh 35 classes trong một pass.
- Conditioned shared model chạy pathway cho khoảng 35 class conditions.
- Per-class model dùng 35 specialist stacks.
- Train loop hiện không dùng AMP/autocast hoặc gradient accumulation.

#### Khuyến nghị

Trước full training cần benchmark:

- Batch 1/2/4.
- FP32, BF16 và FP16.
- Forward, backward, peak VRAM.
- All-class latency.
- Throughput so với CNN baseline.
- Class chunk sizes 1/2/4/8.

Nếu không đạt yêu cầu, cân nhắc:

- Windowed attention.
- Attention trên downsampled tokens.
- Multi-scale CNN/FPN.
- Shared trunk chạy một lần, chỉ condition lightweight head.
- Một multi-class pass thay vì lặp toàn bộ pathway 35 lần.

### P1 — Checkpoint được chọn bằng positive-query validation loss

Best checkpoint hiện được chọn bằng `val_loss`. Validation vẫn dùng query dataset chỉ chứa class hiện diện, nên không trực tiếp đo:

- Absent-class false positives.
- Decoder quality.
- AP.
- Score calibration.
- All-class inference behavior.

#### Khuyến nghị

- Chạy image-level validation AP sau mỗi epoch hoặc mỗi N epoch.
- Chọn checkpoint bằng metric prespecified, ví dụ AP50:95.
- Có thể lưu đồng thời `best_val_loss.pt` và `best_val_ap.pt`.
- Chỉ tune threshold/top-k bằng val, sau đó khóa trước test.

### P1 — Evaluation chưa enforce provenance của checkpoint

Checkpoint lưu `split_manifest_fingerprint`, nhưng evaluator hiện không từ chối khi checkpoint được train bằng manifest khác với manifest đang evaluate.

Hệ quả:

- Có thể vô tình đánh giá checkpoint từ benchmark khác.
- Có thể trộn metadata/stuff policy khác nhau mà evaluator vẫn chạy.
- Protocol yêu cầu fingerprint consistency nhưng code chưa enforce.

#### Khuyến nghị

Đối với internal checkpoint evaluation:

```text
checkpoint.split_manifest_fingerprint
must equal
current_manifest.fingerprint
```

Có thể thêm `--allow-manifest-mismatch` cho trường hợp có chủ đích, nhưng mặc định nên fail closed.

### P1 — External prediction protocol chưa enforce fair comparison

Prediction schema cho phép optional metadata, nhưng `read_predictions()` bỏ metadata và chỉ trả prediction records. Do đó evaluator không kiểm tra được:

- Split manifest fingerprint.
- Class-name ordering.
- Stuff policy.
- Input image size.
- Producer decoder threshold.
- Max detections.
- Metadata fingerprint.

Ngoài ra:

- Docs yêu cầu label thuộc `[0, NUM_CLASSES)` nhưng code chỉ kiểm tra label không âm.
- Internal predictions được áp dụng threshold/top-k/peak suppression.
- External predictions không được áp dụng các CLI settings này, nhưng report vẫn ghi `threshold`, `topk` và `peak_kernel`.

Đây là protocol mismatch cần sửa trước khi so model nội bộ với YOLO, Faster R-CNN hoặc detector ngoài repository.

### P2 — Source drift chưa được kiểm tra tự động khi training

Metadata có hàm kiểm tra raw PNG/SVG SHA-256, nhưng dataset construction chỉ load metadata và không hash lại raw files. Metadata builder mặc định cũng bỏ qua file metadata đã tồn tại thay vì validate source.

Nếu PNG hoặc SVG bị thay đổi mà metadata chưa rebuild:

- Manifest có thể vẫn trông hợp lệ.
- Training có thể dùng image mới với annotation cũ.
- Report vẫn lưu fingerprint cũ.

#### Khuyến nghị

Trước official run:

```bash
python scripts/data/build_dataset.py \
  --data-root ./data/FloorPlanCAD_original \
  --stuff-policy exclude \
  --validate-only \
  --strict

python scripts/data/build_splits.py \
  --data-root ./data/FloorPlanCAD_original \
  --seed 1337 \
  --val-fraction 0.10 \
  --validate-only
```

Tốt hơn nữa là thêm optional preflight source validation vào `train.py`.

### P2 — SVG transform bị bỏ qua

Nếu SVG path có thuộc tính `transform`, parser chỉ ghi warning và bỏ qua transform.

Impact phụ thuộc dataset thực tế. Đây chưa phải confirmed dataset corruption vì hiện không có data để đếm, nhưng cần audit:

- Bao nhiêu path có `transform`.
- Những class nào bị ảnh hưởng.
- Strict validation có fail không.

Không nên chạy official benchmark nếu số này khác zero mà chưa hỗ trợ transform.

### P2 — Square resize làm méo aspect ratio

`ResizeNormalize` resize trực tiếp mọi ảnh về square, không letterbox. Box vẫn đúng về mặt hình học trong resized coordinate system, nhưng floor plan chữ nhật có thể bị kéo giãn mạnh.

Điều này có thể làm biến dạng:

- Door/window proportions.
- Text/symbol appearance.
- Relative object sizes.

Nên ablate:

```text
direct square resize
vs
aspect-ratio preserving letterbox
```

### P2 — Heatmap initialization và stable-logit path chưa được sử dụng

Loss hỗ trợ nhận `center_logits` để dùng `logsigmoid` ổn định, nhưng cả detector và baseline chỉ trả sigmoid heatmap. Heatmap head cũng chưa thấy CenterNet-style negative prior bias initialization.

Với heatmap lớn `64×64`, output ban đầu quanh 0.5 có thể tạo negative focal term rất lớn.

Đây không phải correctness bug, nhưng là training-stability risk nên xử lý trước các run dài:

- Trả cả `center_logits`.
- Dùng logits path trong loss.
- Initialize center bias theo prior probability đã được định nghĩa trong protocol.
- Ghi rõ giá trị này trong experiment config.

## 7. Những limitation đã biết, không phải bug

### 7.1 Center collisions

Representation chỉ có một regression slot cho mỗi class/cell. Repository đã ghi và report collision đúng. Cần đo collision rate theo class và object size trước khi kết luận model architecture yếu.

### 7.2 Metric không phải full COCO evaluator

Implementation có:

- IoU 0.50–0.95.
- 101-point interpolation.
- One-to-one matching.

Nhưng không có:

- Area ranges.
- Crowd handling.
- COCO max-detections variants.

Điều này đã được khai báo đúng trong source và docs.

### 7.3 Byte text không có semantic generalization mặc định

Byte conditioner là encoder random-init. Với fixed prompts, nó có thể chỉ hoạt động như một class code phức tạp hơn class embedding.

Muốn chứng minh language semantics cần thêm:

- Synonym/paraphrase tests.
- Unseen descriptions.
- Runtime prompt perturbation.
- Held-out textual formulations.

### 7.4 Transform validation không quá muộn

Invalid boxes không được phép đi vào target generation:

- Train transform validate box trước geometry.
- Validate lại sau resize.
- `apply_transform()` validate thêm trước khi return.

### 7.5 Weighted sampler resume chưa phải confirmed bug

Cùng một `torch.Generator` được truyền cho sampler và loader, rồi state được checkpoint/restore. Thiết kế có khả năng tái lập đúng ở epoch boundary.

Tuy nhiên, test hiện chỉ kiểm tra RNG round-trip, chưa kiểm tra:

```text
N epoch liên tục
==
K epoch + checkpoint + resume N-K epoch
```

Nên bổ sung integration test này trước khi tuyên bố exact numerical resume.

## 8. Trạng thái bằng chứng

Hiện tại repository chứng minh được:

- Tensor contracts nhất quán.
- Unit tests pass.
- Forward smoke pass.
- Checkpoint schema có thể round-trip.
- Split/metadata logic hoạt động trên synthetic fixtures.

Repository chưa chứng minh được:

- Model train ổn định ở 512×512.
- Model hội tụ.
- Conditioned model tốt hơn baseline.
- Byte/pretrained text cung cấp lợi ích.
- Per-class routing đáng chi phí.
- AP50 hoặc AP50:95 cụ thể.
- Runtime/VRAM khả thi.
- Exact resume tạo cùng final weights.
- Dataset thật không có transform/source/metadata issues.

Vì vậy, đánh giá khoa học hiện tại là:

> **Một implementation/protocol có tiềm năng, chưa phải một kết quả nghiên cứu được xác nhận.**

## 9. Protocol đề xuất trước official test

### Giai đoạn 0 — Data audit

Trên dataset thật, xuất report:

- Số image train/val/test.
- Số class còn GT sau `stuff_policy=exclude`.
- Số class có zero instances.
- Count theo class trước/sau `min_size=8`.
- Số SVG path có unsupported transform.
- Gaussian radius distribution.
- Collision rate theo class và object size.
- Image aspect-ratio distribution.
- Metadata/source validation failures.

### Giai đoạn 1 — Sửa control experiments

Thêm:

1. `floorplan_shared_no_condition`.
2. Positive + sampled absent-class queries.
3. Heatmap logits và prior bias.
4. Val AP checkpoint selection.
5. Manifest consistency check trong evaluator.

### Giai đoạn 2 — Pilot

Dùng:

- 32–100 images hoặc limited query set.
- 2–5 epochs.
- Không dùng test.

Theo dõi:

- Loss finite.
- Gradient norm.
- Score distribution.
- Negative-query false positives.
- Collision rate.
- VRAM/latency.

### Giai đoạn 3 — Ablation chính

Ma trận nên tách rõ:

| ID | Architecture | Pathway | Conditioner | Fusion |
|---|---|---|---|---|
| A0 | CNN CenterNet | shared | none | none |
| A1 | FloorPlanDetector | shared | none | none |
| B | FloorPlanDetector | shared | class embedding | FiLM |
| C | FloorPlanDetector | shared | byte text | FiLM |
| D | FloorPlanDetector | shared | byte text | cross-attention |
| E | FloorPlanDetector | per-class | none | none |
| F | FloorPlanDetector | per-class | byte text | FiLM |
| G | FloorPlanDetector | shared | pretrained text | FiLM |

Giải thích comparison:

- A0 so với A1 đo tác dụng của pathway architecture.
- A1 so với B mới đo conditioning.
- E so với F đo text ngoài specialist routing.
- C so với F đo routing nhưng phải kèm matched-budget controls.

Mỗi configuration nên chạy ít nhất 3 seeds và báo:

- Mean ± standard deviation.
- AP50, AP50:95.
- Per-class AP.
- AP theo object size.
- Absent-class false-positive rate.
- Parameters/FLOPs.
- Peak memory.
- Training và inference latency.
- Collision statistics.

## 10. Luồng tóm tắt

```text
PNG + SVG
→ canonical metadata v2 + source/build fingerprints
→ image-level index
→ deterministic train/val/test manifest
→ positive-only (image, class) query expansion
→ paired augmentation + square resize
→ stride-8 heatmap/size/offset targets
→ class/text conditioner
→ fusion
→ shared hoặc per-class spatial mixer + attention + FFN
→ CenterNet query head
→ focal + masked size/offset losses
→ AdamW + warmup/cosine
→ schema-v2 checkpoint
→ image-level all-class inference
→ local peak suppression + top-k decoding
→ AP50/AP50:95 report
```

Hai thay đổi quan trọng nhất:

```text
positive-only query expansion
→ positive + absent-class negative query sampling
```

và:

```text
CNN baseline so với conditioned architecture
→ thêm same-architecture no-conditioning control
```

## 11. Thứ tự hành động đề xuất

1. Thêm shared no-conditioning control.
2. Thêm absent-class negative-query sampling và metric liên quan.
3. Enforce manifest/class/data provenance trong evaluator.
4. Chạy strict data audit trên dataset thật.
5. Benchmark VRAM/latency ở 512×512.
6. Chạy pilot train/val nhỏ.
7. Chỉ sau đó mới chạy multi-seed ablation và held-out test.
