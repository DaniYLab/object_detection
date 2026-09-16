# ArchCAD public-40K panoptic benchmark protocol v0

> Updated 2026-09-16: implementation is a **local reconstruction**, not verified
> paper parity. See `archcad_benchmark_audit_20260916.md` for executed upstream
> counterexamples, the corrected raw-entity loader and the first learned pilot.

## Research boundary

Mục tiêu trước mắt là tái lập đúng ba task của ArchCAD/DPSS:

1. semantic spotting trên từng graphical primitive;
2. instance spotting bằng primitive masks;
3. panoptic symbol spotting kết hợp thing và stuff.

Written-dimension reading, CDG và reasoning là downstream tasks. Không
thêm chúng vào model trước khi benchmark perception có baseline hợp lệ.

## Evidence from paper and released code

- Paper v1: <https://arxiv.org/html/2503.22346v1>
- Official code: <https://github.com/ArchiAI-LAB/ArchCAD>
- Public data: <https://huggingface.co/datasets/jackluoluo/ArchCAD>

Paper định nghĩa mỗi primitive bằng cặp `(semantic_id, instance_id)` và DPSS
dự đoán class + mask trên primitive sequence. Official config dùng 30 semantic
classes và 800 decoder queries. Target/loss là Hungarian matching với class CE,
mask BCE và Dice.

Audit release code xác nhận taxonomy executable:

```text
semantic classes: 0..29
ignore/background: 30 (raw public semantic 100 -> 30)
thing: 1..11, 14..18, 21, 22, 25, 27, 29  (21 classes)
stuff: 0, 12, 13, 19, 20, 23, 24, 26, 28 (9 classes)
```

Official repository hiện thiếu `svgnet/data`, `dataset` và `modules/pointops`;
evaluator đi kèm không triển khai đầy đủ mọi cột F1/wF1/AP trong paper. Vì vậy
repository này tái dựng contract từ raw annotation và ghi rõ mọi chỗ chưa xác
minh bit-for-bit với evaluator tác giả.

## Canonical ground truth

Mỗi raw entity được giữ nguyên thứ tự và tạo:

```text
semantic_labels[N]
instance_ids[N]
primitive_lengths[N]
geometries[N]
segments[K] = {semantic_id, instance_key, primitive_mask[N]}
```

- Stuff: toàn bộ primitive cùng semantic class trong một chunk là một segment.
- Thing: primitives cùng raw instance string là một segment.
- Thing thiếu instance được giữ trong explicit `(semantic, unassigned:-1)`
  segment để khớp `prepare_targets` của code release, đồng thời audit như lỗi dữ liệu.
- Raw semantic `100` is mapped to ignore/background by the local adapter; the
  missing upstream loader means this conversion is a declared local policy.
- Length weight khớp code release: `round(log(1 + primitive_length), 3)`.

Label không bao giờ đi vào input feature. Model chỉ được nhận geometry, raster,
và các prior được khai báo riêng trong ablation.

## Prediction contract

```json
{
  "image_id": "chunk UUID",
  "semantic_labels": [0, 20, 20, 1],
  "instances": [
    {"semantic_id": 1, "score": 0.93, "mask": [false, false, false, true]}
  ]
}
```

Prediction arrays phải thẳng hàng 1:1 với raw primitive order.

## Metrics

Semantic track báo cả:

- macro và micro primitive F1;
- macro và micro length-weighted F1;
- per-class precision/recall/F1.

Instance track chỉ tính thing classes:

- AP50;
- AP75;
- mAP tại IoU `0.50:0.05:0.95`;
- IoU là primitive-mask IoU weighted theo primitive length.

Local panoptic track uses one-to-one matching at weighted IoU `>0.5`:

```text
RQ = TP / (TP + 0.5 FP + 0.5 FN)
SQ = sum(matched IoU) / TP
PQ = RQ * SQ
```

Report total, thing, stuff and per-class. Total/group PQ pools TP/FP/FN and IoU;
it is not class-macro PQ. Threshold is 0.1. Ignore positions have zero evaluation
weight; zero-weight segments are excluded and counted as unscorable. This
explicit local policy differs from the released evaluator. AP uses 101-point
interpolation; paper AP/F1 aggregation parity remains unverified.

## Oracle gate

```bash
python scripts/research/audit_archcad_panoptic_benchmark.py --sample-count 100
```

Identity prediction should achieve 1.0 on scorable targets; it is a sanity check,
not evidence of official metric equivalence. Audit covers 100 chunks / 24.369 primitives; phát hiện 79
thing primitives thiếu raw instance ID và giữ chúng theo released target logic.

Historical full streaming audit (2026-09-15, old curve-length implementation):

```text
41,097 chunks
14,392,530 raw entities
500,048 panoptic segments
6,031,873 semantic-100/background entities
64,441 thing primitives without instance ID
80 zero-length primitives
22,323 primitives with rounded log-length weight 0
max 81,472 entities/chunk
```

Fingerprint của toàn bộ raw JSON corpus:
`5a8ca6404b365b2f0a12938c99e095bb2244af1985aa71e29f34ecbe8cb6b873`.

The current curve-length audit is
`outputs/panoptic/archcad_public40k_audit_20260916.json`; use it for length/weight
statistics. The earlier report is retained for traceability.

`point.zip` is not 1:1 with JSON entities (e.g. 213 entities vs 261 records), but
its use by the missing DPSS loader has not been established. It is not a required
dependency for the local raw-entity dataset. JSON/SVG alignment was verified on
100 chunks; the four-point geometry representation is versioned separately and
does not claim exact DPSS preprocessing parity.

## Replication stages

1. JSON/SVG entity alignment audit completed on 100 chunks.
2. Raw-entity PyTorch dataset/collate implemented, preserving every primitive.
3. Primitive-only query/mask pilot completed (32 train / 16 val / 20 epochs);
   expand this control before fusion/conditioning experiments.
4. Raster-only baseline với image feature sampled tại primitive centers.
5. Dual-pathway adaptive fusion + Mask2Former-style primitive decoder.
6. Sau khi baseline hội tụ, thêm conditioned/unconditioned reflex ablation.

## Comparison boundary

Public local có 41.097 chunks; paper báo 413.062 chunks và split 7:1:2 theo
source drawing. Public release không kèm chunk-to-drawing mapping. Do đó:

- được báo within-public-40K validation để phát triển;
- không được gọi là reproduction của Table 4/5/6 hoặc generalization result;
- không được so trực tiếp với số paper cho đến khi có cùng data/split/evaluator.
