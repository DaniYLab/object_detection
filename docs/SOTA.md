# SOTA comparison

## Provenance của bảng nguồn

Repository hiện **không xác định được paper/source** của bảng này. Không có
title, authors, URL, bibliography hoặc định nghĩa evaluator; `[42]` và `[5]`
chỉ là citation indices chưa được resolve. Vì vậy toàn bộ bảng được xem là
user-supplied reference cho đến khi source paper và code đánh giá được xác minh.

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

### Historical detector reports (exploratory held-out evaluation)

The repository detector results below are reported for auditability. Checkpoints,
thresholds and decoder settings were selected before the historical test runs;
those test values are not an optimization target for future work. The A0/A1/B
rows share the historical manifest `f83fcffd...`, while later V/V2 reports use a
different manifest as documented below.

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

Trong held-out detection protocol (historical A0/A1/B manifest):

- B − A1 (conditioning effect): **+0.0563 AP50**, **+0.0397 AP50:95**.
- B − A0: **+0.0367 AP50**, **+0.0248 AP50:95**.
- B cải thiện AP50 trên 24/29 test classes có GT so với A1; 4 classes giảm và 1 gần tie.
- Absent-query detection rate tại threshold 0.05: A0 61.55%, A1 64.89%, B **50.91%**.
- Mean max score trên present queries: A0 0.6250, A1 0.6060, B **0.6701**.

These are historical single-seed held-out observations and must not be used to
choose future architecture, sampling, thresholds or decoder settings. The
conditioning claim for future research is based on the three-seed validation
replication in `docs/experiment_log.md`.

Nguồn: `outputs/ablation_summary.json`, `outputs/conditioning_analysis.json`, `outputs/conditioning_analysis_test.json`, `outputs/evaluation_test_*_seed1337.json`, `docs/experiment_log.md`.

**Manifest caveat:** các test report A0/A1/B lịch sử dùng fingerprint
`f83fcffd...` với 27,095 GT. Các vector report V/V2 về sau dùng fingerprint
`a916b9fa...` với 26,588 GT. Vì vậy không được dùng các bảng đó để tính V−B test
delta như một controlled comparison; xem `docs/experiment_log.md`.

## Có so trực tiếp hai bảng được không?

**Không.** Việc đặt `0.4021 AP50` cạnh `0.806 F1` và kết luận model kém hơn là sai vì:

1. Bảng SOTA dùng **F1 theo 8 broad categories**, có `Wall` và `Parking lot`; repository báo **bounding-box AP** trên 30 object classes.
2. Benchmark hiện tại dùng `stuff_policy=exclude`, nên `wall` có **0 GT instance** và bị loại khỏi AP macro.
3. F1 phụ thuộc một operating threshold cụ thể; AP tích phân precision–recall qua score ranking và IoU thresholds.
4. Chưa xác nhận cùng split, resize, label grouping, instance/stuff semantics, post-processing hoặc matching rule.
5. HRNet/DeepLab thường là semantic-segmentation architectures; detector hiện tại là object detection.

Vì vậy, bảng SOTA chỉ dùng làm **tham khảo task khác**, không được dùng để claim repository đạt hoặc không đạt SOTA.

## Fresh schema-v2 speed-quality result (not an aligned SOTA comparison)

A new validation-only run used the frozen
`data/FloorPlanCAD_original/splits_v2.json` manifest (fingerprint
`203f63588e906637f4dbcf6d1a001e67d0e688016c9b0c799c3ea89b438e8e9a`).
`dual_pathway_late`, seed 1337, was trained on `train_new`, selected on
`val_select`, and calibrated only on `val_calibration`; `test_set` was not
accessed.

| Role | AP50 | AP50:95 | Checkpoint/decoder |
|---|---:|---:|---|
| `val_select` (full 1,016 images) | 0.1598 | 0.0668 | `best_val_ap.pt`, threshold 0.05, top-k 100, peak 3 |
| `val_calibration` baseline | 0.1553 | 0.0646 | threshold 0.05, top-k 100, peak 3 |
| `val_calibration` calibrated | 0.1553 | 0.0646 | all per-class thresholds remained 0.05 |

The global validation-only threshold sweep also selected 0.05. These detector AP
values must not be compared numerically with the source table's segmentation F1.
The late model was also profiled against V2: it has 11.86% fewer parameters and
9.87x–15.17x lower synchronized all-class latency depending on class chunk size,
but the comparison is a speed-oriented architecture comparison, not a
capacity-matched or pure conditioning-stage ablation. Historical V2 quality
reports use a different manifest and therefore do not provide a controlled
quality delta.


Chọn một trong hai hướng:

### A. Detection benchmark (khuyến nghị cho repository hiện tại)

- Chạy mọi baseline trên cùng schema-v2 manifest, `stuff_policy`, image size và class mapping.
- Xuất box predictions theo `object_detection_predictions` schema.
- Đánh giá chung bằng `evaluate.py`: AP50, AP50:95, per-class AP và absent-class false-positive rate.
- Dùng `train_new` cho training, `val_select` cho checkpoint/architecture selection và
  `val_calibration` cho decoder calibration; giữ `test_set` cho một final report.

### B. Reproduce bảng F1 segmentation

- Xác định chính xác paper/source, category mapping và định nghĩa F1/Weighted F1.
- Thêm segmentation/group-class head hoặc quy tắc rasterization tương thích.
- Dùng đúng split và evaluator của paper.
- Báo F1 theo 8 categories riêng; không suy ra từ box AP.

Có thể tính một **supplemental detection F1@IoU=0.5** cho broad categories, nhưng chỉ sau khi category mapping và score threshold được prespecify trên validation. Con số đó vẫn không tương đương semantic-segmentation F1 nếu bảng nguồn dùng pixel masks.
