# Nhật Ký Phát Triển (Development Log)

## Phase 0: Khởi Tạo Dự Án
- Download FloorPlanCAD dataset từ HuggingFace (FiftyOne format)
- Chuyển đổi từ FiftyOne → folder structure: `{sample_id}/original.png` + `{class}_{n}.png`
- Build `metadata.json` cho mỗi sample chứa bounding box thông tin

## Phase 1: Dataset Pipeline
- Viết `scripts/data/build_dataset.py` để parse SVG annotations → bounding boxes
- **Bug quan trọng:** Ban đầu dùng regex parse SVG path → tọa độ sai (snap về 0,0) vì Arc command chứa tham số không phải tọa độ
- **Fix:** Chuyển sang `svgpathtools` library cho robust bbox extraction
- Viết `scripts/data/generate_metadata.py` để sinh `metadata.json` từ SVG

## Phase 2: Class Mapping
- Phát hiện 7 class ID không có tên (class_06, class_08, class_10, class_25, class_26, class_32, class_35)
- Nhận diện bằng cách xem ảnh crop → mapping đúng tên
- Rename 209,164 file crop trên disk cho đồng bộ
- Tổng: 35 classes, từ `annotation_text` đến `window_blind`

## Phase 3: Model Architecture V1
- Implement `FloorPlanDetector` với:
  - VAEEncoderStub (placeholder cho Flux VAE)
  - TextEncoderStub (placeholder cho T5/CLIP)
  - EarlyFusion (Cross-Attention)
  - ObjectLearningBlock (Mamba + Self-Attention)
  - HeatmapHead → 35 channel output
- **Sai lầm:** Dùng 4 shared blocks + class embedding thay vì 35 per-class blocks
- **Sai lầm:** Output 35 channels (multi-class heatmap) thay vì per-query CenterNet

## Phase 4: Sửa Sai Thiết Kế ← **Hiện tại**

### Vấn đề 1: Bỏ text conditioning
- **Sai:** Hard-code text thành `"Find object..."` — model mất tác nhân kích thích
- **Fix:** Khôi phục text per-class: `"Find {class_name} in this floor plan drawing"`

### Vấn đề 2: Shared blocks vs Per-class blocks  
- **Sai:** 4 shared blocks dùng chung cho tất cả class (chỉ conditioning bằng embedding addition)
- **Fix:** 35 separate block stacks, mỗi class có đường dây thần kinh riêng

### Vấn đề 3: Random chọn 1 class per epoch
- **Sai:** Mỗi sample chỉ random 1 class → nhiều class bị bỏ sót
- **Fix:** Expand dataset: mỗi (ảnh, class) = 1 sample → 44,229 entries, train đầy đủ mọi class mỗi epoch

### Vấn đề 4: Visual Crops là input
- **Sai:** Load crop ảnh làm input cho model 
- **Fix:** Bỏ hẳn — khi inference không có crop (đó là thứ cần tìm!)

### Kết quả hiện tại:
- Dataset: 44,229 expanded samples ✓
- Model: 35 per-class blocks × 2 depth, ~131.6M params ✓
- Loss: Focal (center) + L1 (size) ✓
- Output: CenterNet style (center heatmap + size map) ✓

## Phase 5: Training & Evaluation (TODO)
- [ ] Setup training trên server (GPU)
- [ ] Train baseline: 50 epochs, lr=1e-5, focal_weight=10.0, warmup=500 steps
- [ ] Implement inference script: text query → bounding boxes trên ảnh mới
- [ ] Visualize predictions vs ground truth
- [ ] Thay stub encoders bằng pretrained (Flux VAE, T5)

## Phase 6: Diffusion Output (TODO — thiết kế gốc)
- [ ] Sau khi model CenterNet hoạt động ổn, thử thêm Diffusion head
- [ ] Combine output tất cả class → diffusion vẽ bounding boxes lên ảnh
- [ ] So sánh đường line gen vs ground truth line để tính loss
