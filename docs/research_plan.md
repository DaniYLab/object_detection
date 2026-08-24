# Research Plan: Conditioned CenterNet cho FloorPlanCAD

**Ngày tạo:** 2026-07-23  
**Cập nhật:** 2026-07-31  
**Trạng thái:** Core ablation A0/A1/B hoàn tất cho seed 1337; cần thêm 2 seeds trước kết luận cuối  
**Giả thuyết trung tâm:** Class ID hoặc text conditioning giúp detector tập trung vào đúng loại đối tượng trong floor plan tốt hơn detector không conditioning.

---

## 1. Trạng thái hiện tại

### Đã hoàn thành

| Hạng mục | Trạng thái |
|---|---|
| Metadata pipeline v2 (SHA-256, bbox xyxy, stuff_policy, fingerprint) | ✅ |
| Image-level split chống leakage (train/val từ train_set_1+2, test giữ nguyên) | ✅ |
| Query dataset expansion sau split | ✅ |
| CenterNet targets stride-8 (heatmap + size + offset + mask) | ✅ |
| Conditioners: class_embedding, byte_text, pretrained_text | ✅ |
| Fusion: none / add / FiLM / cross-attention / FiLM+cross-attention | ✅ |
| Pathway: shared / per-class | ✅ |
| ObjectLearningBlock: GatedSpatialMixer2D → SelfAttention → FFN | ✅ |
| GroupNorm CenterNet head (5 channels) | ✅ |
| Focal loss + masked SmoothL1 size/offset | ✅ |
| AdamW + warmup/cosine + balanced sampler | ✅ |
| Checkpoint schema v2 (model/optimizer/RNG/fingerprints) | ✅ |
| Decoder: local peak → top-k → offset-aware decode | ✅ |
| AP50 / AP50:95 evaluator (101-point interpolation) | ✅ |
| Prediction JSON schema (internal + external models) | ✅ |
| Test suite: 90 tests pass, lightweight preset smoke pass | ✅ |
| Data audit: 15,663 images, 0 SVG transform issues, 1.53% collisions | ✅ |
| Absent-class negative query sampling | ✅ |
| Direct no-conditioning multi-class pathway control | ✅ |
| Val AP checkpoint selection + manifest provenance enforcement | ✅ |
| BF16/FP16 mixed precision support | ✅ |
| Seed-1337 A0/A1/B training + full-val evaluation | ✅ |

### Còn lại

- Replicate A1 vs B trên ít nhất 2 seeds nữa.
- Text/fusion/routing ablations C–G.
- External detector protocol hardening và threshold calibration.

### Kết quả sơ bộ seed 1337

- A0 `centernet_baseline`: **val** AP50 0.3819, AP50:95 0.1908; **test** AP50 0.3529, AP50:95 0.1922.
- A1 `shared_no_condition`: **val** AP50 0.3513, AP50:95 0.1685; **test** AP50 0.3334, AP50:95 0.1773.
- B `floorplan_base`: **val** AP50 0.4021, AP50:95 0.2072; **test** AP50 0.3896, AP50:95 0.2170.
- B − A1: **+5.08 AP50**, **+3.86 AP50:95** on val; **+5.63 AP50**, **+3.97 AP50:95** on test.
- B − A0: **+2.02 AP50**, **+1.64 AP50:95** on val; **+3.67 AP50**, **+2.48 AP50:95** on test.
- B also lowers absent-query detection rate versus A1 and A0 on both val and test.

Chi tiết và per-class/absent-query analysis: `docs/experiment_log.md`, `outputs/conditioning_analysis.json`, `outputs/conditioning_analysis_test.json`.

---

## 2. Các vấn đề phải sửa trước khi training

### P0 — Bắt buộc sửa trước bất kỳ experiment nào

#### P0-A: Training chỉ thấy positive queries

**Vấn đề:** `FloorPlanQueryDataset` chỉ tạo `(image, class)` khi class *có mặt* trong image. Trong evaluation, model được query tất cả 35 class trên mọi image. Model chưa bao giờ thấy "empty heatmap expected" — dẫn đến false positive cao cho absent classes, score calibration sai, và validation loss không đo được hallucination.

**Sửa:**
- Thêm absent-class negative query sampling vào `FloorPlanQueryDataset`: với mỗi positive query, sample thêm K class không có trong image, tạo empty heatmap/size/offset target.
- Tỷ lệ `K` là hyperparameter cần báo cáo; khởi đầu với K=1 hoặc K=2.
- Xem xét focal loss normalization khi batch chứa cả positive và negative-only samples: `num_pos==0` branch dùng `mean`, `num_pos>0` branch dùng `sum/N_pos` — hai thang đo cần được căn chỉnh.

**File:** `src/data/dataset.py`, `src/training/losses.py`

**Metric mới cần thêm:**
- False-positive rate cho absent classes.
- Score distribution: positive queries vs. absent-class queries.
- AP chia theo present/absent query groups.

---

#### P0-B: Thiếu architecture control preset

**Vấn đề:** `centernet_baseline` dùng CNN đơn giản với multi-class 5C head một pass. `floorplan_base` dùng GatedSpatialMixer + SelfAttention + FiLM. Nếu `floorplan_base` tốt hơn, không thể biết lợi ích đến từ conditioning hay từ kiến trúc pathway.

**Sửa:** Thêm preset:
```python
"shared_no_condition": ModelConfig(
    architecture="floorplan_detector",
    pathway_mode="shared",
    conditioner=ConditionerConfig(kind="none"),
    fusion_mode="none",
    ...
)
```
Đây mới là control trực tiếp để đo conditioning. `centernet_baseline` vẫn giữ để trả lời câu hỏi rộng hơn.

**File:** `src/models/presets.py`

---

#### P0-C: Per-class confound bởi parameter count

**Vấn đề:** Per-class pathway nhân toàn bộ stack theo số class:

| Preset | Parameters |
|---|---:|
| `centernet_baseline` | 3,674,031 |
| `floorplan_base` | 5,268,741 |
| `per_class_no_text` | 85,707,781 |
| `per_class_fixed_byte_text_film` | 99,664,517 |

So `per_class` với `shared` mà không có budget-matched control không thể kết luận specialist routing tốt hơn.

**Sửa:**
- Thêm `shared_wide` preset: cùng architecture FloorPlanDetector, shared pathway, nhưng `model_dim` lớn hơn để parameter count gần với `per_class_no_text`.
- Thêm `per_class_small` preset: per-class pathway nhưng `model_dim` và `depth_per_class` giảm để match `floorplan_base`.
- Mọi bảng kết quả phải kèm parameter count và FLOPs đo thực tế.

**File:** `src/models/presets.py`

---

### P1 — Sửa trước official training run

#### P1-A: Checkpoint selection bằng validation loss, không phải AP

**Vấn đề:** Val loss trên positive-only queries không đo absent-class false positives, decoder quality, hay AP. Best checkpoint có thể không phải checkpoint tốt nhất về detection.

**Sửa:**
- Chạy image-level validation AP sau mỗi N epoch (N=5 hoặc 10).
- Lưu `best_val_loss.pt` và `best_val_ap50.pt` riêng biệt.
- Chọn checkpoint cho test evaluation bằng `val AP50:95` (prespecified).

**File:** `train.py`, `src/training/` (checkpoint logic)

---

#### P1-B: Evaluator không enforce manifest fingerprint

**Vấn đề:** Checkpoint lưu `split_manifest_fingerprint` nhưng `evaluate.py` không từ chối khi checkpoint được train bằng manifest khác. Có thể vô tình so sánh checkpoints từ benchmark khác nhau.

**Sửa:**
```python
if checkpoint.get("split_manifest_fingerprint") != manifest.fingerprint:
    raise EvaluationError(
        "Manifest mismatch. Train và evaluate phải dùng cùng split manifest. "
        "Dùng --allow-manifest-mismatch để bỏ qua (không khuyến nghị)."
    )
```

**File:** `evaluate.py`

---

#### P1-C: VRAM/latency chưa được đo

**Vấn đề:** 512×512 → 4096 spatial tokens. N²=16.7M query-key pairs/head/block. Shared model chạy pathway 35 lần (per chunk). Per-class model chạy 35 specialist stacks tuần tự. Không có AMP. Training có thể không khả thi trên GPU thông thường.

**Phải benchmark trước khi commit:**
- Batch 1/2/4, FP32 + BF16.
- Forward, backward, peak VRAM.
- All-class latency (throughput/image).
- Throughput so với `centernet_baseline`.
- Chunk sizes 1/2/4/8.

**Nếu VRAM không đủ, cân nhắc theo thứ tự:**
1. BF16 mixed precision (AMP) — thêm vào `train.py` trước.
2. Gradient accumulation.
3. Giảm `model_dim` hoặc `depth_per_class`.
4. Windowed attention hoặc attention trên downsampled tokens.

---

#### P1-D: center_logits unused, không có negative prior bias

**Vấn đề:** `FloorPlanDetector` trả `center_heatmap` (đã sigmoid). `centernet_loss()` kiểm tra `preds.get("center_logits")` nhưng luôn `None`. Path `logsigmoid` ổn định số học không được dùng. Với heatmap 64×64 ban đầu gần 0.5, focal penalty sẽ lớn.

**Sửa:**
- Detector trả thêm `center_logits` trong prediction dict.
- `centernet_loss()` dùng `logits` path thay vì `pred.clamp`.
- Initialize center bias theo CenterNet-style: `bias = -log((1-π)/π)` với `π ≈ 0.01`.
- Ghi giá trị `π` và bias init trong experiment config.

**File:** `src/models/detector.py`, `src/models/baseline.py`, `src/models/config.py`

---

### P2 — Nên sửa trước official benchmark

#### P2-A: Source drift không được kiểm tra tự động

Dataset construction load metadata nhưng không hash lại raw PNG/SVG. Nếu source file thay đổi mà metadata chưa rebuild, training dùng image mới với annotation cũ.

**Sửa:** Thêm optional preflight source validation vào `train.py`:
```bash
python train.py ... --validate-sources
```

#### P2-B: SVG transform bị bỏ qua

Nếu SVG path có `transform` attribute, parser chỉ ghi warning và bỏ qua. Cần audit trước khi chạy official benchmark.

**Action:** Chạy `python scripts/data/build_dataset.py --validate-only --strict` và đếm số path có transform bị bỏ qua.

#### P2-C: Square resize làm méo aspect ratio

`ResizeNormalize` resize trực tiếp về square, không letterbox. Floor plan hình chữ nhật có thể bị kéo giãn. Nên ablate:
- Direct square resize (hiện tại)
- Aspect-ratio preserving letterbox

#### P2-D: External prediction protocol chưa enforce fair comparison

`read_predictions()` bỏ metadata ngoài prediction records. Evaluator không kiểm tra class-name ordering, stuff policy, hay input image size khi đánh giá detector ngoài repository.

**Sửa:** Validate metadata khi đọc external prediction JSON, ít nhất kiểm tra class mapping và manifest fingerprint.

---

## 3. Các giai đoạn thực hiện

### Giai đoạn 0 — Data audit (trước mọi training)

**Mục tiêu:** Hiểu dataset thực tế trước khi commit model selection.

**Việc cần làm:**

1. Chạy strict metadata validation trên dataset thật:
```bash
python scripts/data/build_dataset.py \
  --data-root ./data/FloorPlanCAD_original \
  --stuff-policy exclude \
  --validate-only --strict
```

2. Xuất report với các số liệu sau:
   - Số image train/val/test sau split.
   - Số class còn GT sau `stuff_policy=exclude` và `min_size=8`.
   - Số class có zero instances (sẽ bị bỏ khỏi AP macro average).
   - Số SVG path có `transform` attribute bị bỏ qua.
   - Histogram collision rate theo class và object size.
   - Histogram Gaussian radius theo class.
   - Distribution aspect ratio của ảnh gốc.
   - Source SHA-256 validation failures (nếu có).

3. Xem xét kết quả trước khi chuyển sang Giai đoạn 1. Nếu SVG transform count khác zero, cần quyết định: sửa parser hoặc loại các ảnh bị ảnh hưởng khỏi benchmark.

**Deliverable:** `outputs/data_audit_report.json` + tóm tắt ngắn trong `docs/`.

---

### Giai đoạn 1 — Sửa và bổ sung (trước pilot training)

**Mục tiêu:** Đảm bảo training distribution, controls và evaluation đúng trước khi tốn compute.

**Checklist:**

- [ ] **P0-A:** Thêm absent-class negative query sampling vào `FloorPlanQueryDataset`.
- [ ] **P0-A:** Thêm metric false-positive rate cho absent classes.
- [ ] **P0-A:** Xem xét focal loss normalization khi mixed batch.
- [ ] **P0-B:** Thêm preset `shared_no_condition` vào `src/models/presets.py`.
- [ ] **P0-C:** Thêm `shared_wide` và `per_class_small` presets.
- [ ] **P1-A:** Thêm val AP checkpoint selection vào `train.py`.
- [ ] **P1-B:** Enforce manifest fingerprint trong `evaluate.py`.
- [ ] **P1-D:** Detector trả `center_logits`, dùng logits path trong loss, thêm prior bias init.
- [ ] **P2-A:** Thêm `--validate-sources` flag vào `train.py`.
- [ ] Cập nhật test suite cho mọi thay đổi trên.

**Không cần làm trong giai đoạn này:**
- P2-C (letterbox): ablate sau khi có baseline kết quả.
- P2-D (external prediction): sửa khi cần so sánh với detector ngoài.

---

### Giai đoạn 2 — Pilot training

**Mục tiêu:** Xác nhận training loop hoạt động, loss finite, không có NaN, VRAM ổn định, model bắt đầu hội tụ.

**Setup:**
- Dùng 32–100 images (hoặc limited query set).
- 5–10 epochs.
- Preset: `centernet_baseline` và `floorplan_base`.
- Không dùng test set.
- Device: `cpu` hoặc GPU nhỏ nếu có.

**Benchmark VRAM/latency phải chạy trước:**
```bash
# Đo trên hardware thực tế
python scripts/dev/smoke_models.py \
  --device cuda \
  --image-size 512 \
  --all-lightweight-presets \
  --profile
```

**Theo dõi trong pilot:**
- Loss finite và giảm.
- Gradient norm (không explode/vanish).
- Score distribution: positive queries vs. absent-class queries.
- False-positive rate cho absent classes.
- Peak VRAM từng preset.
- Throughput (images/second).

**Tiêu chí pass:** Loss giảm ổn định trong 5 epoch, không có NaN, VRAM trong giới hạn hardware.

**Nếu VRAM không đủ:** Thêm AMP (BF16) trước khi giảm model size. Ghi rõ setting được dùng cho mọi run sau đó.

---

### Giai đoạn 3 — Ablation chính

**Mục tiêu:** Trả lời các câu hỏi nghiên cứu với bằng chứng AP thực tế.

**Điều kiện để bắt đầu:**
- Giai đoạn 0–2 hoàn thành.
- Data audit không có blocker (zero SVG transform issues hoặc đã sửa).
- P0/P1 fixes đã merged.
- Pilot training pass.
- VRAM/latency đã được đo và documented.

**Ma trận thí nghiệm tối thiểu:**

| ID | Architecture | Pathway | Conditioner | Fusion | Câu hỏi được trả lời |
|---|---|---|---|---|---|
| A0 | SharedCenterNet | shared | none | none | Baseline CNN |
| A1 | FloorPlanDetector | shared | none | none | Kiến trúc pathway có giúp không? |
| B | FloorPlanDetector | shared | class_embedding | FiLM | Class conditioning có giúp không? |
| C | FloorPlanDetector | shared | byte_text (fixed) | FiLM | Text (vs class embed) trên shared? |
| D | FloorPlanDetector | shared | byte_text (fixed) | cross_attention | FiLM vs cross-attention |
| E | FloorPlanDetector | per-class | none | none | Specialist routing (unconstrained budget) |
| F | FloorPlanDetector | per-class | byte_text (fixed) | FiLM | Text + routing (unconstrained) |
| E' | FloorPlanDetector | per-class-small | none | none | Specialist routing (budget-matched) |
| A1' | FloorPlanDetector | shared-wide | none | none | Shared với budget tương đương E |
| G | FloorPlanDetector | shared | pretrained_text | FiLM | Pretrained semantics có giá trị không? |

**Comparisons xác định:**
- A0 vs A1: pathway architecture effect.
- A1 vs B: conditioning effect (isolate).
- B vs C: class embedding vs byte text (shared).
- C vs D: FiLM vs cross-attention (fixed, shared).
- E vs F: text ngoài specialist routing.
- E' vs A1': budget-matched routing comparison.
- C vs G: lightweight vs pretrained text.

**Yêu cầu mỗi run:**
- Ít nhất 3 seeds khác nhau.
- Report: mean ± std AP50, AP50:95.
- Per-class AP50 và per-class ground-truth count.
- Absent-class false-positive rate.
- Resolved `ModelConfig` và parameter count đo từ code.
- Peak VRAM, training và inference latency.
- Collision statistics từ data audit.
- Checkpoint selection metric: `val AP50:95`.

---

### Giai đoạn 4 — Held-out test

**Mục tiêu:** Báo cáo kết quả cuối cùng trên test set, không tune thêm sau bước này.

**Quy tắc bất biến:**
- Chỉ chạy sau khi preset, checkpoint, threshold, top-k và mọi hyperparameter đã chốt bằng train/val.
- Không dùng test AP để chọn lại checkpoint hoặc thay đổi decoder settings.
- Mỗi preset chỉ được chạy test **một lần**.

**Command:**
```bash
python evaluate.py \
  --data-root ./data/FloorPlanCAD_original \
  --manifest ./data/FloorPlanCAD_original/splits.json \
  --split test \
  --checkpoint ./checkpoints/best_val_ap.pt \
  --report ./outputs/evaluation_test_<preset>_<seed>.json \
  --image-size 512
```

**Deliverable:** JSON report cho mỗi preset, kèm tóm tắt bảng trong `docs/`.

---

## 4. Câu hỏi nghiên cứu và điều kiện falsification

| Câu hỏi | Điều kiện bác bỏ |
|---|---|
| Class embedding + FiLM có tốt hơn no-conditioning? | B không tốt hơn A1 trên ít nhất 2/3 seeds |
| Per-class routing có tốt hơn shared (budget-matched)? | E' không tốt hơn A1' sau kiểm soát parameters |
| Fixed byte text thêm giá trị ngoài class routing? | F không tốt hơn E trên ít nhất 2/3 seeds |
| FiLM có tốt hơn cross-attention (shared)? | D không tốt hơn C một cách nhất quán |
| Pretrained text có tốt hơn lightweight text? | G không tốt hơn C sau kiểm soát params |

---

## 5. Những điều KHÔNG được làm

- Không ghi AP/mAP nếu chưa chạy `evaluate.py` trên held-out split.
- Không gọi `GatedSpatialMixer2D` là Mamba hoặc SSM.
- Không gọi byte encoder là open-vocabulary hoặc pretrained.
- Không gọi fixed class prompts là runtime text-query detection.
- Không so kết quả từ hai manifest/stuff_policy khác nhau.
- Không dùng test loss/AP để chọn checkpoint, threshold, hoặc hyperparameters.
- Không hardcode parameter count trong tài liệu — đo từ model thực tế.

---

## 6. Tóm tắt thứ tự hành động

```
[NGAY BÂY GIỜ]
1. Data audit trên dataset thật (Giai đoạn 0)
   → xuất report, quyết định SVG transform issue

[TRƯỚC PILOT]
2. P0-A: Thêm absent-class negative query sampling
3. P0-B: Thêm preset shared_no_condition
4. P0-C: Thêm shared_wide và per_class_small presets
5. P1-A: Val AP checkpoint selection
6. P1-B: Enforce manifest fingerprint trong evaluator
7. P1-D: center_logits + prior bias init
8. Update tests

[PILOT]
9. Benchmark VRAM/latency ở 512×512
10. Pilot train A0 + A1, 5-10 epochs, 32-100 images
11. Xác nhận: loss finite, no NaN, VRAM ổn định

[ABLATION]
12. Chạy ma trận A0–G với ≥3 seeds
13. Báo cáo AP50/AP50:95, per-class AP, FP rate, params, latency

[CUỐI CÙNG]
14. Held-out test — mỗi preset một lần, không tune thêm
```

---

## 7. Định nghĩa "thành công"

**Thành công tối thiểu:** Ít nhất một conditioned variant (B, C, hoặc D) vượt A1 (same-architecture no-conditioning control) một cách nhất quán trên ≥2/3 seeds, với delta AP50 > 1 điểm tuyệt đối.

**Thành công mở rộng:** Text conditioning (C hoặc G) cải thiện AP so với class embedding (B), cho thấy thông tin ngôn ngữ mang thêm signal ngoài class ID.

**Kết quả null cũng có giá trị:** Nếu conditioning không cải thiện AP, đó vẫn là kết quả nghiên cứu hợp lệ và có thể publish. Protocol được thiết kế để phát hiện null result đáng tin cậy, không chỉ confirm hypothesis.

---

*Xem thêm: `docs/research_protocol.md`, `docs/design_philosophy.md`, `docs/architecture.md`, `docs/research_assessment.md`*
