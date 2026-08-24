# Data and Annotation Semantics

## Thing và stuff

FloorPlanCAD là dataset panoptic 35 lớp. Thuộc tính SVG `instance-id` quyết định semantics ở annotation level:

- `instance-id >= 0`: thing instance; các path cùng `(semantic-id, instance-id)` được union thành một object bbox.
- `instance-id == -1`: stuff/uninstanced annotation; một SVG path không được mặc nhiên xem là một object độc lập.

Canonical metadata builder hỗ trợ ba policy tách biệt:

| Policy | Hành vi | Mục đích |
|---|---|---|
| `exclude` | Không tạo detection instance cho `instance-id=-1` | Default benchmark object detection v2 |
| `merge_by_class` | Union toàn bộ stuff paths cùng semantic class | Thử nghiệm region-level, không phải instance benchmark |
| `path_instances` | Mỗi path thành pseudo-instance | Tương thích metadata pipeline cũ |

Kết quả giữa ba policy không thể so trực tiếp. Metadata ghi policy và fingerprint để evaluator/checkpoint phát hiện mismatch.

## Unknown semantic IDs

Builder luôn đếm và ghi warning khi SVG chứa `semantic-id` ngoài mapping canonical. Policy được chọn tường minh:

- `--unknown-policy warn` (default): bỏ path không biết khỏi targets nhưng lưu warning/statistics trong metadata;
- `--unknown-policy error`: từ chối toàn bộ SVG, phù hợp khi xây benchmark strict.

`--strict` cũng biến mọi parser warning thành failure. `unknown_policy` là một phần của build settings/fingerprint nên không được trộn metadata tạo bằng policy khác nhau.

## Box convention

Mọi bbox chuẩn hóa dùng float `[x0, y0, x1, y1)`:

- `x0`, `y0` inclusive;
- `x1`, `y1` là biên ngoài;
- `0 <= x0 < x1 <= image_width`;
- `0 <= y0 < y1 <= image_height`.

SVG `viewBox="min_x min_y width height"` phải trừ cả `min_x/min_y` trước khi scale. Không giả định viewBox bắt đầu tại `(0,0)`.

## CenterNet collisions

Query target có một size và offset tại mỗi `(class, output cell)`. Nếu nhiều bbox cùng class rơi vào một cell:

- heatmap vẫn biểu diễn cell peak;
- policy mặc định `largest` giữ bbox có area lớn hơn cho size/offset;
- bbox còn lại vẫn là ground truth cho AP nhưng không có regression target riêng;
- `TargetStats` ghi số collision và object không được regression-supervise.

Collision tạo giới hạn recall của representation ở output stride hiện tại. Report phải công bố collision rate thay vì coi số supervised centers là số object thật.

## Metadata migration

Metadata không có schema được xem là legacy. Loader có thể đọc để tương thích, nhưng benchmark chính thức cần metadata v2 với source hash và build settings. Builder không tự động overwrite metadata; dùng `--validate-only` để kiểm tra và chỉ dùng `--force` khi chủ động chuyển benchmark.
