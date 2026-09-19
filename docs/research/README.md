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
  -> written-dimension reading later
```

## Sources of truth

1. `archcad_benchmark_protocol.md` — benchmark contract hiện hành.
   `archcad_benchmark_audit_20260916.md` records corrections, parity limitations and the pilot.
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
| Semantic evaluator | Local v1 tested; paper parity open | ignore handling and invalid-label regressions |
| Instance evaluator | Local v1 tested; paper parity open | primitive-mask AP; interpolation/weight convention explicit |
| Panoptic evaluator | Local v1 tested; upstream differs | one-to-one vs released matching; differential fixtures recorded |
| 100-sample oracle audit | Passed | all tracks = 1.0 on 24,369 primitives |
| Public-40K full audit | Complete | 41,097 chunks / 14,392,530 entities / 500,048 segments |
| Data anomalies | Re-audited 2026-09-16 | 64,441 thing primitives missing instance; 80 zero-length; 22,389 zero-weight |
| JSON/SVG entity mapping | Verified on 100 samples | 24,369 semantic/instance pairs; auxiliary point.zip is not a proven model dependency |
| Panoptic PyTorch dataset/collate | Implemented | raw entity indices, geometry-only features, per-image masks, split fingerprint |
| Primitive-only learned baseline | Three-seed pilot completed; superseded 2026-09-17 | the 256-image / 20-epoch control (PQ 0.11663 ± 0.01155) was not bit-reproducible and is 0.9% of the training split; current best is PQ 0.246188 |
| Best primitive-only configuration | 2026-09-20 | per-primitive semantic head on all 27,272 eligible training images with per-class grouping radii: pooled PQ 0.331524, thing PQ 0.334900, per-primitive accuracy 57.3% |
| Grouping rule | Per-class radii adopted 2026-09-20 | `PER_CLASS_PROXIMITY` in `src/data/archcad_geometric_grouping.py`, fitted on train by train PQ: +0.025724 over the uniform radius, 3.3x the resolution. Fitted against today's predicted classes, so refit after the semantics improve |
| Grouping ceiling | Measured | with today's semantics, oracle grouping scores 0.480167 and oracle semantics with the fixed grouper 0.807876, against 0.331524 for the adopted pipeline |
| Lever audit 2026-09-17/20 | Budget, capacity, data, scale features, exposure, learned grouping | budget PLATEAU at 50x; capacity criterion met by 2%; data live; exposure exhausted at 4,096 images but still climbing at the full split; scale-aware input no resolvable effect; learned edge pruning and connectivity-only grouping both worse; per-class radii the first win |
| Raster/dual-pathway baseline | Not started; motivation closed | the raw entities carry no styling channel: `rgb` is `[0,0,0]` for every entity, so appearance cannot separate classes that geometry cannot |
| Conditioned Reflex panoptic ablation | Not started | after the semantic path is stable |
| Paper-comparable reproduction | Blocked | public subset/split differs from 413K paper benchmark |

## Upstream audit

Official ArchCAD code provides model, matcher, criterion and a partial evaluator,
but the current public repository lacks `svgnet/data`, `dataset`, and
`modules/pointops`. Its config establishes 30 semantic classes, 800 queries,
class/mask/dice matching costs and the 21/9 thing/stuff split. The local adapter
therefore reconstructs the missing data/evaluation contract without claiming
bit-for-bit equivalence prematurely.

## Required order

1. Separate optimization/data-budget limitations from architectural limitations
   in the primitive-only control. The completed 3-seed pilot has low thing recall;
   inspect prespecified longer-budget curves before changing the architecture.
   Answered 2026-09-17: budget is NOT the constraint. Four levers were moved on the
   fixed split — budget 20 → 1000 epochs (PLATEAU), capacity 149K → 1.1M parameters
   (criterion met by a 2% margin), data 256 → 4,096 images (INCONCLUSIVE), and each
   landed in the same validation band as the 20-epoch control (`0.11165`), none
   exceeding it beyond the `0.016429` noise floor. Adding data does remove the
   overfitting (train/val PQ gap `+0.1092` → `−0.0034`) without improving validation,
   which points at the geometry-only representation rather than at optimization.
   See `../experiments/archcad_optimization_20260917.md` §E5 and
   `../experiments/archcad_data_scale_20260917.md`.
2. Keep the existing public-40K split fingerprint and the local v1 metric profile
   fixed. Exact paper preprocessing/evaluator parity remains a separate open audit.
3. Profile the long primitive-count tail before scaling; the pilot uses an
   explicit 1,024-primitive / 64-segment eligibility cap, with no target truncation.
4. Raw-query diagnosis found class-agnostic coverage 29.19%, correct-class
   coverage 12.15%, and final thing recall 9.61% (seed 1337); thresholding alone
   does not explain the misses. See `../experiments/archcad_primitive_replication_20260916.md`.
   Extended 2026-09-17: 24 of the 28 segments lost between 12.15% and 9.61% (86%)
   are lost inside the decoder's disjoint assignment and mask binarisation — but only
   on an under-trained model. On a converged checkpoint the same stages cost nothing
   (49 → 49 → 49 of 51), so that loss tracks optimization progress, not decoder
   design, and the decoder is not the binding constraint. See
   `../experiments/archcad_optimization_20260917.md` section E3.
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
