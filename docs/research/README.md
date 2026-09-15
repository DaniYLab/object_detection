# Main Research Direction: ArchCAD Panoptic Symbol Spotting

Đây là entry point chính. Nghiên cứu phải tái lập benchmark ArchCAD trước khi
đưa thêm OCR, CDG, room reconstruction hay language reasoning.

## Thesis

Giả thuyết ban đầu là phản xạ không điều kiện đọc toàn cảnh và phản xạ có điều
kiện tập trung vào đúng loại đối tượng. FloorPlanCAD CenterNet đã chứng minh
class conditioning và vector geometry có ích cho bbox detection. Bước hiện tại
là chuyển các insight đó sang task giàu supervision hơn mà ArchCAD thực sự định
nghĩa: gán semantic và instance cho toàn bộ graphical primitives.

```text
raster + geometry-only primitives
  -> semantic prediction per primitive
  -> instance mask prediction over primitives
  -> panoptic thing/stuff output
  -> object reconstruction and dimensions later
```

## Sources of truth

1. `archcad_benchmark_protocol.md` — benchmark contract hiện hành.
2. `../experiments/experiment_log.md` — kết quả Conditioned CenterNet đã kiểm chứng.
3. `../../src/data/archcad_panoptic.py` — raw annotation → canonical panoptic GT.
4. `../../src/evaluation/archcad_panoptic.py` — semantic/instance/panoptic metrics.
5. `../../outputs/panoptic/archcad_oracle_benchmark_v0.json` — oracle audit.
6. `../../outputs/panoptic/archcad_public40k_audit_v0.json` — full-corpus contract audit.
7. `archcad_provenance_recovery_20260915.md` — giới hạn public-40K split.

Các tài liệu `drawing_understanding_direction_20260915.md`, CDG compiler,
room-gold pack và executable QA là exploratory work. Chúng không được dùng làm
trạng thái chính hoặc chặn ArchCAD benchmark.

## Current status

| Milestone | Status | Evidence |
|---|---|---|
| FloorPlanCAD conditioned effect | Verified in old protocol | +5.37 AP50 mean over no-condition, 3 seeds |
| FloorPlanCAD vector-source effect | Verified in old protocol | +9.39 AP50 mean, 2 seeds |
| Hybrid stride-4 head | Promising, needs clean replication | +2.6–3.1 AP50:95; old leaking manifest |
| FloorPlanCAD schema-v3 dedup | Complete | cold control has not been trained |
| ArchCAD public data | Available | 41,097 aligned PNG/SVG/JSON/point/caption chunks |
| ArchCAD taxonomy contract | Complete v0 | 30 semantics = 21 thing + 9 stuff; raw 100 ignored |
| Primitive/instance target parser | Complete v0 | preserves all classes and primitive order |
| Semantic evaluator | Complete v0 | macro/micro F1 and length-weighted variants |
| Instance evaluator | Complete v0 | primitive-mask AP50/AP75/mAP |
| Panoptic evaluator | Complete v0 | PQ/SQ/RQ total, thing, stuff, per-class |
| 100-sample oracle audit | Passed | all tracks = 1.0 on 24,369 primitives |
| Public-40K full audit | Complete | 41,097 chunks / 14,392,530 entities / 500,048 segments |
| Data anomalies | Quantified | 64,441 thing primitives missing instance; 80 zero-length; 22,323 zero-weight |
| JSON/SVG → point tensor mapping | Open blocker | public point arrays are not 1:1 with JSON entities; upstream loader missing |
| Panoptic PyTorch dataset/collate | Not started | current ArchCAD dataset is bbox-only |
| Primitive-only learned baseline | Not started | first model milestone |
| Raster/dual-pathway baseline | Not started | after primitive-only control |
| Conditioned Reflex panoptic ablation | Not started | after benchmark baseline converges |
| Paper-comparable reproduction | Blocked | public subset/split differs from 413K paper benchmark |

## Upstream audit

Official ArchCAD code provides model, matcher, criterion and a partial evaluator,
but the current public repository lacks `svgnet/data`, `dataset`, and
`modules/pointops`. Its config establishes 30 semantic classes, 800 queries,
class/mask/dice matching costs and the 21/9 thing/stuff split. The local adapter
therefore reconstructs the missing data/evaluation contract without claiming
bit-for-bit equivalence prematurely.

## Required order

1. Resolve and test the exact JSON/SVG primitive → `point.zip` representation;
   do not assume raw entity order matches DPSS tensor order.
2. Freeze the public-40K development split and fingerprint; clearly mark it as
   non-comparable to the paper split.
3. Implement panoptic dataset/collate with aligned raster and verified primitive/point order.
4. Train a small primitive-only control and verify non-trivial PQ.
5. Add raster feature sampling and adaptive fusion.
6. Reproduce the DPSS-style query/mask decoder baseline.
7. Only then test conditioned vs unconditioned reflex mechanisms.

## Metrics and claims

Primary metrics are F1/wF1, AP50/AP75/mAP and PQ/SQ/RQ. Bbox AP is a historical
auxiliary metric only. Until source-drawing mapping/full data is available,
results must be labeled `ArchCAD public-40K development protocol`, not Table
reproduction or final generalization.

## Deferred work

- OCR and written dimensions.
- CDG and executable QA.
- Room/connectivity extraction.
- MLLM/LLM reasoning.
- Any model scaling before the primitive-only baseline and evaluator are stable.
