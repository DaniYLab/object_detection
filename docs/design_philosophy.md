# Triết Lý Thiết Kế: Conditioned Detection như một giả thuyết có thể kiểm chứng

## Bối cảnh

Dự án nghiên cứu object detection trên bản vẽ FloorPlanCAD. “Conditioned Reflex” là trực giác thiết kế: một điều kiện về class hoặc text có thể hướng image features tới loại đối tượng cần tìm. Đây là **giả thuyết cần ablation và detection metrics**, không phải kết luận đã được chứng minh.

## Input hợp lệ

Model luôn nhận ảnh floor plan hoàn chỉnh. Không dùng object crop hoặc visual template làm input vì chúng không tồn tại khi inference trên ảnh mới.

Conditioning có thể là:

- không conditioning;
- learned class embedding;
- lightweight text encoder random-init;
- optional pretrained text encoder;
- runtime text hoặc fixed class prompts.

Lightweight text encoder không có pretrained language semantics. Fixed prompts cũng không tự biến model thành open-vocabulary detector.

## Ba trục phải tách riêng

### 1. Pathway

- `shared`: mọi class dùng chung feature-processing stack;
- `per_class`: mỗi class có stack riêng.

Per-class pathway có nhiều capacity hơn và class ID đã trực tiếp chọn module. Vì vậy hiệu quả của nó không được quy cho text nếu chưa so với `per_class + no text`.

### 2. Conditioning

- `none`;
- `class_embedding`;
- `lightweight_text`;
- `pretrained_text`.

### 3. Fusion

- `none`;
- additive condition;
- FiLM;
- image-to-text cross-attention;
- FiLM + cross-attention.

Các trục này được cấu hình độc lập qua model presets để tạo ablation công bằng.

## Kiến trúc khái quát

```text
Image
  -> stride-8 ConvImageEncoder
  -> image tokens + 2D positional embedding

Condition (none / class / text)
  -> conditioner tokens + padding mask + pooled condition

image tokens + condition
  -> selected fusion
  -> shared hoặc per-class pathway
       GatedSpatialMixer -> corrected SelfAttention -> FFN
  -> CenterNet head
       center heatmap + size + fractional offset
```

`GatedSpatialMixer` là depthwise spatial convolution có gate, không phải Mamba hoặc selective state-space model. Self-attention dùng scaled dot-product attention normalize theo key dimension.

## Runtime behavior

Query mode:

```text
image + class_id + optional text
-> prediction cho class được yêu cầu
```

All-class mode:

```text
image
-> encode image một lần
-> evaluate class conditions theo chunk
-> concatenate outputs
```

Text preset dùng fixed prompt khi caller không truyền runtime text. Preset không dùng text không được coi text là nguồn thông tin ngầm.

## Điều gì sẽ chứng minh hoặc bác bỏ giả thuyết?

Các so sánh tối thiểu:

1. Shared CenterNet, không conditioning.
2. Shared + class embedding.
3. Per-class pathways, không text.
4. Per-class pathways + fixed text.
5. Shared + fixed text.
6. Shared + pretrained text.
7. FiLM so với cross-attention khi giữ nguyên các yếu tố khác.

Nếu cấu hình 4 không tốt hơn 3, fixed text không cung cấp giá trị ngoài routing. Nếu 3 tốt hơn 1 nhưng có nhiều tham số hơn đáng kể, cần kiểm soát parameter budget trước khi kết luận specialist pathways tốt hơn.

Protocol split, AP metrics và report requirements nằm tại `docs/research_protocol.md`.

## Nguyên tắc bất biến mới

1. Ảnh hoàn chỉnh là visual input; không dùng ground-truth crop làm gợi ý.
2. Train/val/test chia ở image level; test không tham gia model selection.
3. Output detection là center heatmap + size + offset ở stride 8.
4. Mọi padding text phải được mask.
5. Mọi tên module phải mô tả đúng computation thực tế.
6. Text, routing và specialist capacity phải được ablate độc lập.
7. Chỉ detection AP trên held-out data mới được dùng làm bằng chứng hiệu quả.
