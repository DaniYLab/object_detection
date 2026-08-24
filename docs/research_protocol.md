# Research Protocol

Tài liệu này định nghĩa cách kiểm chứng Conditioned Reflex/Conditioned CenterNet mà không làm rò rỉ test set. Nó là protocol cho các thí nghiệm tương lai; repository không ghi kết quả nếu chưa thực sự chạy training và evaluation.

## 1. Định nghĩa bài toán

- Input chính: một ảnh floor plan hoàn chỉnh.
- Output detection: `boxes`, `scores`, `labels` trong hệ 35 semantic classes.
- Query model có thể nhận `class_id` và optional runtime text; all-class inference phải encode ảnh một lần rồi tổng hợp output của mọi class.
- Text encoder random-init chỉ là learned conditioning. Chỉ backend pretrained mới có thể được đánh giá về khả năng hiểu text ngoài các prompt đã học.
- Annotation có `instance-id=-1` là stuff, không mặc nhiên được coi là object instance. Mỗi report phải ghi `stuff_policy`.

## 2. Split bắt buộc

1. Chia ở **image level** trước khi expand `(image, class)`.
2. `train` và `val` chỉ lấy từ `train_set_1` + `train_set_2`.
3. Default validation fraction: 10%, seed 1337.
4. Toàn bộ `test_set` được giữ nguyên cho báo cáo cuối cùng.
5. Mỗi run phải lưu split-manifest fingerprint. Không so sánh hai run dùng manifest khác nhau như cùng một benchmark.
6. Không dùng test loss/AP để chọn checkpoint, threshold, decoder settings hoặc hyperparameters.

## 3. Metadata benchmark

Mỗi report phải ghi:

- metadata schema/version;
- class-mapping fingerprint;
- source/build fingerprint;
- `min_size_px`;
- `stuff_policy`;
- target output stride;
- collision count/rate.

Benchmark object-detection v2 dùng `stuff_policy=exclude`. `merge_by_class` và `path_instances` là protocol khác, không được trộn kết quả.

## 4. Metrics

Metrics chính:

- AP50;
- AP50:95 với IoU thresholds 0.50, 0.55, ..., 0.95;
- per-class AP50/AP50:95;
- số ground-truth và prediction mỗi class;
- macro average chỉ trên class có ít nhất một ground-truth instance.

Matching là one-to-one trong cùng image và class, prediction được xét theo score giảm dần. Metric report phải lưu threshold, top-k, output stride và clipping convention.

Validation loss chỉ dùng cho tối ưu/checkpoint selection. Nó không thay thế detection metrics.

## 5. Ablation matrix

Tối thiểu phải chạy các cấu hình sau trên cùng split và decoder:

| ID | Pathway | Conditioning | Fusion | Câu hỏi |
|---|---|---|---|---|
| A | shared | none | none | Shared CenterNet baseline |
| B | shared | class embedding | FiLM | Class conditioning có giúp không? |
| C | per-class | none | none | Lợi ích đến từ specialist capacity hay không? |
| D | per-class | lightweight fixed text | FiLM | Fixed text thêm gì ngoài routing? |
| E | shared | lightweight fixed text | FiLM | Text khi không có per-class routing |
| F | shared | pretrained text | FiLM | Pretrained semantics có giá trị không? |
| G | shared/per-class | cùng conditioner | cross-attention | FiLM so với spatial text attention |

Để quy kết tác dụng cho text, D phải được so với C. Để quy kết tác dụng cho per-class pathways, D phải được so với E. Không được chỉ so mô hình lớn hơn với baseline nhỏ rồi kết luận conditioning tốt hơn.

Mỗi bảng kết quả phải kèm:

- tổng/trainable parameter count;
- peak memory nếu đo được;
- latency per image;
- seed;
- checkpoint-selection metric;
- class-balanced sampler settings;
- augmentation settings.

## 6. Reproducibility

Mỗi checkpoint hoàn chỉnh cần lưu:

- model preset/config;
- epoch và global step;
- optimizer/scheduler/scaler state;
- Python/NumPy/PyTorch/DataLoader RNG state;
- class mapping;
- split/metadata fingerprints;
- best metric và validation metrics.

Báo cáo chính thức nên dùng nhiều seed. Một run đơn chỉ là smoke/baseline, không đủ để khẳng định cải thiện ổn định.

## 7. Không được tuyên bố khi chưa có bằng chứng

- Không gọi spatial gated convolution là Mamba/SSM.
- Không gọi lightweight random-init text encoder là open-vocabulary language understanding.
- Không gọi fixed class prompts là runtime text-query detection.
- Không ghi mAP/baseline numbers nếu chưa chạy evaluator trên untouched test split.
- Không so trực tiếp kết quả cũ dùng `test_set` làm validation với benchmark v2.
