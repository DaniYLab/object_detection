# Data Audit — FloorPlanCAD (stuff excluded)

**Ngày:** 2026-07-25
**Nguồn:** `outputs/data_audit_report.json` (script `scripts/data/audit_dataset.py`)
**Settings:** `image_size=512`, `output_stride=8`, `stuff_policy=exclude`, `min_size=8px`, `collision_policy=largest`, `gaussian_min_overlap=0.7`

## Tổng quan

| Hạng mục | Giá trị |
|---|---:|
| Tổng ảnh (train+val+test) | 15,663 |
| Tổng instance (sau stuff exclude, min_size 8) | 81,983 |
| Class còn GT | 30 / 35 |
| Class zero-instance | 5 |
| Collision rate tổng | 1.53% (1,253 / 81,983) |
| SVG transform bị bỏ qua | 0 |

## Split (manifest seed=1337, val_fraction=0.10)

| Split | Images | Instances | Classes with GT |
|---|---:|---:|---:|
| train | 9,145 | 47,755 | 30 |
| val | 1,016 | 7,133 | 30 |
| test | 5,502 | 27,095 | 29 |

## Class zero-instance (bị loại khỏi AP macro average)

`annotation_text`, `dimension_line`, `door_single`, `symbol_misc`, `wall`

Đây là các class stuff (`instance-id=-1`) bị loại bởi `stuff_policy=exclude`, đúng như thiết kế benchmark object. AP macro tính trên 30 class có GT; test có 29 do một class không xuất hiện.

## Class distribution (train+val+test)

- **Rarest:** phần lớn class thiết bị/nội thất hiếm.
- **Most common:** `door_double` (22,853), `door_sliding` (12,348), `table` (6,972), `sink` (5,356), `counter` (4,794).

Distribution lệch mạnh về door classes → dùng balanced sampler (`--sampler balanced --balance-power 0.5`) trong training.

## Aspect ratio

Median = 1.000, min = 0.999, max = 1.001. Ảnh gốc gần như vuông hoàn toàn.

**Kết luận:** `ResizeNormalize` square resize không gây méo đáng kể. **P2-C (letterbox ablation) không cần thiết** cho dataset này.

## Gaussian radius (output grid, stride 8)

Overall: min=0, p25=1, median=2, p75=4, max=17. Radius nhỏ phù hợp với heatmap 64×64.

## Collision theo kích thước object (sau resize 512)

| Size bucket | Boxes | Collisions | Rate |
|---|---:|---:|---:|
| small (<32²) | 18,403 | 12 | 0.07% |
| medium (32²–96²) | 30,748 | 38 | 0.12% |
| large (≥96²) | 32,832 | 1,203 | 3.66% |

Collision tập trung ở large objects (cùng class, cùng center cell ở stride 8). Rate tổng 1.53% là chấp nhận được; không cần đổi output stride.

## Quyết định

1. **Không có blocker.** 0 SVG transform, collision thấp, aspect ratio ~1.0.
2. Tiến hành pilot training (Giai đoạn 2) với `centernet_baseline` và `shared_no_condition` / `floorplan_base`.
3. AP macro báo cáo trên 30 class có GT; ghi rõ 5 class zero-instance bị loại.
4. Bỏ qua P2-C (letterbox) — không cần thiết.

*Xem thêm: `docs/research_plan.md`, `docs/data_semantics.md`.*
