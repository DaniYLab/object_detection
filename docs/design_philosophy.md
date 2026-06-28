# Triết Lý Thiết Kế: Phản Xạ Có Điều Kiện (Conditioned Reflex Learning)

## Bối Cảnh

Dự án này phát triển một mô hình deep learning để phát hiện đối tượng (object detection) trên bản vẽ mặt bằng kiến trúc (floor plan CAD). Khác với các phương pháp detection truyền thống (YOLO, Faster R-CNN), thiết kế lấy cảm hứng từ **lý thuyết phản xạ có điều kiện của Pavlov** trong sinh học thần kinh.

## Ý Tưởng Cốt Lõi

### 1. Tác Nhân Kích Thích (Stimulus) = Text Prompt

Thay vì cho model "nhìn" ảnh rồi tự detect tất cả mọi thứ, ta dùng **Text prompt** như một **tác nhân kích thích**:

```
Input:  Ảnh bản vẽ + "Find chair in this floor plan drawing"
Output: Vị trí và kích thước các ghế trong ảnh
```

Model **không tự quyết** sẽ tìm gì. Con người (hoặc hệ thống) **ra lệnh** thông qua ngôn ngữ.

### 2. Đường Dây Thần Kinh Riêng Biệt = Per-Class Blocks

Mỗi loại đối tượng (class) có **một bộ não riêng** — một stack of Object Learning Blocks chuyên biệt:

```
class_blocks[0]  → Chuyên nhận diện annotation_text
class_blocks[4]  → Chuyên nhận diện chair
class_blocks[8]  → Chuyên nhận diện door_double
class_blocks[30] → Chuyên nhận diện wall
... (35 pathways tổng cộng)
```

Khi text prompt là `"Find chair"` → `class_id = 4` → dữ liệu **chỉ đi qua `class_blocks[4]`**. Các block khác không được kích hoạt.

### 3. Phản Xạ Không Điều Kiện → Phản Xạ Có Điều Kiện

- **Phase 1 (Không điều kiện):** Model được cho xem ảnh + text cụ thể + ground truth heatmap → học cách phản ứng.
- **Phase 2 (Có điều kiện):** Qua nhiều epochs, mỗi block tích lũy "kinh nghiệm" riêng. Chỉ cần nghe text "Find chair" → block chair tự kích hoạt các neuron đúng mà không cần bất kỳ gợi ý thị giác nào khác.

### 4. Tại Sao Không Dùng Visual Crops Làm Input?

Crops (ảnh cắt nhỏ của từng đối tượng) **chỉ tồn tại trong training set**. Khi inference trên một bản vẽ hoàn toàn mới:
- Bạn **không có** crops — vì đó chính là thứ bạn đang cố tìm!
- Bạn **chỉ có** ảnh mới + text query

Do đó, input duy nhất hợp lệ cho model là: **Ảnh gốc + Text**.

## Sơ Đồ Kiến Trúc

```
                    ┌─────────────┐
                    │  Text Prompt │  "Find chair in this floor plan drawing"
                    │  (Stimulus)  │
                    └──────┬──────┘
                           │ Text Encoder
                           ▼
┌──────────┐        ┌─────────────┐
│  Image   │───────►│ Early Fusion │  Cross-Attention (text queries image)
│ (VAE Enc)│        │             │
└──────────┘        └──────┬──────┘
                           │
                           ▼ class_id routing
              ┌────────────┼────────────┐
              │            │            │
         ┌────▼────┐  ┌────▼────┐  ┌────▼────┐
         │ Block 0 │  │ Block 4 │  │Block 30 │  ← 35 per-class pathways
         │ annot.  │  │ chair ★ │  │  wall   │     (Mamba + Self-Attention)
         └─────────┘  └────┬────┘  └─────────┘
                           │ (only block 4 is active)
                           ▼
                    ┌─────────────┐
                    │ CenterNet   │  center_heatmap [1, H, W]
                    │   Head      │  size_map       [2, H, W]
                    └─────────────┘
```

## Quy Tắc Thiết Kế Bất Biến

1. **Text là input duy nhất ngoài ảnh** — không có crops, không có visual templates.
2. **Mỗi class có block riêng** — không share weights giữa các class.
3. **Mỗi epoch train đầy đủ tất cả class** — dataset expanded: mỗi (ảnh, class) = 1 sample.
4. **Output là CenterNet** — Gaussian center heatmap + size regression, phân biệt từng instance riêng rẽ.
