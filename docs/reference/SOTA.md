# SOTA comparison

## Provenance của bảng nguồn

**Đã xác định** (2026-08-27): bảng nguồn đến từ bài báo **FloorPlanCAD** của
Fan et al., ICCV 2021 — arXiv:2105.07147 (https://arxiv.org/abs/2105.07147),
dataset tại https://floorplancad.github.io/. `[42]` = HRNet (Wang et al., TPAMI
2020); `[5]` = DeepLabv3+ (Chen et al., ECCV 2018). Các cột F1/Weighted F1 khớp
chính xác Table của bài: HRNet-W18 0.656/0.683, W48 0.666/0.693, DeepLab-R50
0.680/0.705, R101 0.688/0.714, Ours 0.806/0.798.

Bản chất metric (đây chính là lý do KHÔNG được đặt cạnh box AP):

- F1 của bài là **entity-level** (từng graphic primitive — line/arc/curve trong
  SVG), KHÔNG phải pixel mask cũng không phải bounding box.
- Weighted F1 trọng số theo `log(1 + L)` (L = độ dài entity), giảm ảnh hưởng
  của các nét quá dài.
- Các baseline pixel (HRNet/DeepLab) được chuyển thành entity label bằng
  majority sample dọc theo entity, không đo trực tiếp bằng mask.
- Task semantic không dùng confidence threshold; IoU>0.5 chỉ dùng cho matching.

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

1. Bảng SOTA dùng **entity-level F1** theo 8 broad categories (thực ra bài gốc
   dùng 30 class: 28 thing + wall + parking), có `Wall` và `Parking lot`;
   repository báo **bounding-box AP** trên 30 object classes.
2. F1 nguồn là **entity-level** (graphic primitive), không phải pixel mask cũng
   không phải box; Weighted F1 lại còn trọng số theo `log(1+độ dài entity)`.
3. Benchmark hiện tại dùng `stuff_policy=exclude`, nên `wall` có **0 GT instance**
   trong MỌI role của manifest v2 và bị loại khỏi AP macro.
4. F1 phụ thuộc một operating threshold cụ thể; AP tích phân precision–recall
   qua score ranking và IoU thresholds.
5. Chưa xác nhận cùng split, resize, label grouping, instance/stuff semantics,
   post-processing hoặc matching rule.
6. HRNet/DeepLab thường là semantic-segmentation architectures; detector hiện
   tại là object detection.

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


## Benchmark object-detection thực sự (cùng task với repo)

Bài FloorPlanCAD (Table instance/panoptic symbol detection) cũng báo kết quả
object detection — đây mới là nhóm comparator **cùng task** (instance/box
detection) với CenterNet của repo, khác hẳn cột entity-F1 segmentation:

| Detector | AP50 | mAP |
|---|---:|---:|
| Faster R-CNN R101 | 0.602 | 0.452 |
| FCOS | 0.624 | 0.453 |
| YOLOv3 | 0.639 | 0.413 |

Đây là target "chạm/vượt benchmark" hợp lệ cho một detector. Cần lưu ý: để so
được công bằng vẫn phải xác minh split, class grouping, input resolution và
matching rule của bài; trước khi xác minh, các số trên chỉ là mục tiêu tham
khảo cùng task, không phải baseline tái lập tại repo.


## Related work — ArchCAD-400K / DPSS (arXiv:2503.22346)

Bài kế nhiệm trực tiếp của FloorPlanCAD: *ArchCAD-400K: An Open Large-Scale
Architectural CAD Dataset and New Baseline for Panoptic Symbol Spotting*
(Luo et al., arXiv:2503.22346, 2025). Nó dùng đúng FloorPlanCAD làm benchmark
phụ và xác nhận chéo provenance bảng SOTA của repo: baseline DeepLabv3+ (R101)
trên FloorPlanCAD semantic spotting = **F1 68.8 / wF1 71.4**, khớp chính xác
0.688 / 0.714 trong bảng nguồn.

**Đề xuất DPSS (Dual-Pathway Symbol Spotter)** — image branch (HRNetV2-W48) +
point-cloud primitive branch (PointTransformerV2) + adaptive gated fusion +
DETR/Mask2Former decoder. Đây chính là dạng dual-pathway image+vector mà repo
đang theo, độc lập xác nhận hướng kiến trúc là đúng.

Benchmark ladder cùng task (FloorPlanCAD, no priors):

| Track | DeepLabv3+ | Faster R-CNN | SymPointV2 | DINO | DPSS |
|---|---:|---:|---:|---:|---:|
| Semantic F1 | 68.8 | — | 87.0 | — | **92.0** |
| Semantic wF1 | 71.4 | — | — | — | **93.2** |
| Instance mAP | — | — | 57.5 | 47.5 | **61.5** |
| Instance AP50 | — | 0.602 | — | 0.649 | — |

Ghi chú: instance "mAP" của bài là symbol-spotting matching riêng, không trực
tiếp là COCO AP50:95 của repo; AP50 (Faster R-CNN 0.602, DINO 0.649) là đích
gần nhất cho một box detector. Khoảng cách kiến trúc đáng lưu ý: DPSS dùng
point-transformer primitive encoder + DETR/Mask2Former decoder, thay vì
CenterNet heatmap box-decoder — giới hạn của repo có thể nằm ở decoder lẫn
vector-encoder, cần test độc lập trên validation (không suy từ test).

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
