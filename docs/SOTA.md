# SOTA comparison

## Bảng được cung cấp

> `Ours (source table)` là model được gọi là “Ours” trong bảng nguồn, **không phải** `floorplan_base` của repository này.

| Method | Door | Window | Stair | Appliance | Furniture | Equipment | Wall | Parking lot | F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HRNetV2-W18 [42] | 0.821 | 0.620 | 0.845 | 0.597 | 0.726 | 0.880 | 0.620 | 0.610 | 0.656 | 0.683 |
| HRNetV2-W48 [42] | 0.811 | 0.640 | 0.847 | 0.651 | 0.754 | 0.889 | 0.624 | 0.577 | 0.666 | 0.693 |
| DeepLabv3+-R50 [5] | 0.828 | 0.659 | 0.856 | 0.684 | 0.763 | 0.895 | 0.630 | 0.664 | 0.680 | 0.705 |
| DeepLabv3+-R101 [5] | 0.837 | 0.666 | 0.852 | 0.725 | **0.780** | 0.895 | 0.634 | **0.669** | 0.688 | 0.714 |
| **Ours (source table)** | **0.848** | **0.709** | **0.857** | **0.769** | 0.764 | **0.926** | **0.814** | 0.539 | **0.806** | **0.798** |

## Thống kê so với previous best

| Metric/category | Ours | Previous best | Absolute delta | Relative delta |
|---|---:|---:|---:|---:|
| Door | 0.848 | 0.837 | +0.011 | +1.3% |
| Window | 0.709 | 0.666 | +0.043 | +6.5% |
| Stair | 0.857 | 0.856 | +0.001 | +0.1% |
| Appliance | 0.769 | 0.725 | +0.044 | +6.1% |
| Furniture | 0.764 | 0.780 | −0.016 | −2.1% |
| Equipment | 0.926 | 0.895 | +0.031 | +3.5% |
| Wall | 0.814 | 0.634 | **+0.180** | **+28.4%** |
| Parking lot | 0.539 | 0.669 | **−0.130** | **−19.4%** |
| F1 | 0.806 | 0.688 | **+0.118** | **+17.2%** |
| Weighted F1 | 0.798 | 0.714 | **+0.084** | **+11.8%** |

Tóm tắt:

- Đứng đầu **6/8 category**.
- Cải thiện lớn nhất ở `Wall`: +0.180 absolute.
- Hai category kém previous best: `Furniture` −0.016 và `Parking lot` −0.130.
- F1 tổng tăng +0.118; Weighted F1 tăng +0.084 so với previous best.
- Mean đơn giản của 8 category (không phải cột F1 chính thức): HRNet-W18 0.715, HRNet-W48 0.724, DeepLab-R50 0.747, DeepLab-R101 0.757, Ours-source 0.778.

## Kết quả detector trong repository này

Cùng FloorPlanCAD manifest seed 1337. Checkpoint được chọn trước bằng validation AP; test được chạy một lần theo yêu cầu người dùng, không dùng để tune.

### Validation (1,016 images, 7,133 GT boxes, 30 classes có GT)

| Model | Params | AP50 | AP50:95 |
|---|---:|---:|---:|
| A0 `centernet_baseline` | 3.674M | 0.3819 | 0.1908 |
| A1 `shared_no_condition` | 4.887M | 0.3513 | 0.1685 |
| **B `floorplan_base`** | 5.269M | **0.4021** | **0.2072** |

### Held-out test (5,502 images, 27,095 GT boxes, 29 classes có GT)

| Model | AP50 | AP50:95 | Δ AP50 test−val | Δ AP50:95 test−val |
|---|---:|---:|---:|---:|
| A0 `centernet_baseline` | 0.3529 | 0.1922 | −0.0290 | +0.0014 |
| A1 `shared_no_condition` | 0.3334 | 0.1773 | −0.0179 | +0.0088 |
| **B `floorplan_base`** | **0.3896** | **0.2170** | −0.0125 | +0.0098 |

Trong held-out detection protocol:

- B − A1 (conditioning effect): **+0.0563 AP50**, **+0.0397 AP50:95**.
- B − A0: **+0.0367 AP50**, **+0.0248 AP50:95**.
- B cải thiện AP50 trên 24/29 test classes có GT so với A1; 4 classes giảm và 1 gần tie.
- Absent-query detection rate tại threshold 0.05: A0 61.55%, A1 64.89%, B **50.91%**.
- Mean max score trên present queries: A0 0.6250, A1 0.6060, B **0.6701**.

Conditioning effect tăng nhẹ từ validation (+5.08 AP50) sang test (+5.63 AP50), nên lợi ích không biến mất ngoài validation split.

Nguồn: `outputs/ablation_summary.json`, `outputs/conditioning_analysis.json`, `outputs/conditioning_analysis_test.json`, `outputs/evaluation_test_*_seed1337.json`, `docs/experiment_log.md`.

## Có so trực tiếp hai bảng được không?

**Không.** Việc đặt `0.4021 AP50` cạnh `0.806 F1` và kết luận model kém hơn là sai vì:

1. Bảng SOTA dùng **F1 theo 8 broad categories**, có `Wall` và `Parking lot`; repository báo **bounding-box AP** trên 30 object classes.
2. Benchmark hiện tại dùng `stuff_policy=exclude`, nên `wall` có **0 GT instance** và bị loại khỏi AP macro.
3. F1 phụ thuộc một operating threshold cụ thể; AP tích phân precision–recall qua score ranking và IoU thresholds.
4. Chưa xác nhận cùng split, resize, label grouping, instance/stuff semantics, post-processing hoặc matching rule.
5. HRNet/DeepLab thường là semantic-segmentation architectures; detector hiện tại là object detection.

Vì vậy, bảng SOTA chỉ dùng làm **tham khảo task khác**, không được dùng để claim repository đạt hoặc không đạt SOTA.

## Protocol cần thiết để so sánh công bằng

Chọn một trong hai hướng:

### A. Detection benchmark (khuyến nghị cho repository hiện tại)

- Chạy mọi baseline trên cùng `splits.json`, `stuff_policy`, image size và class mapping.
- Xuất box predictions theo `object_detection_predictions` schema.
- Đánh giá chung bằng `evaluate.py`: AP50, AP50:95, per-class AP và absent-class false-positive rate.

### B. Reproduce bảng F1 segmentation

- Xác định chính xác paper/source, category mapping và định nghĩa F1/Weighted F1.
- Thêm segmentation/group-class head hoặc quy tắc rasterization tương thích.
- Dùng đúng split và evaluator của paper.
- Báo F1 theo 8 categories riêng; không suy ra từ box AP.

Có thể tính một **supplemental detection F1@IoU=0.5** cho broad categories, nhưng chỉ sau khi category mapping và score threshold được prespecify trên validation. Con số đó vẫn không tương đương semantic-segmentation F1 nếu bảng nguồn dùng pixel masks.
