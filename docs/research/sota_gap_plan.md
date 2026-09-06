# Deep Research: Path to an Aligned Benchmark Result

**Date:** 2026-08-26 · **Goal:** reach or exceed the source SOTA table only after
reproducing its exact metric/protocol. Until then, the table is reference-only;
our box AP numbers must not be described as beating its segmentation F1.

## 1. Where we actually stand (diagnostic only, not a benchmark comparison)

The following aggregation maps historical V box AP50 into broad categories only to
locate failure modes. It is **not directly comparable** with source-table F1:
the metrics, labels, manifests and evaluator differ. In particular, the historical
B and V test reports use different manifest fingerprints, so their test delta is
not a controlled comparison:

| Category | V test AP50 (macro) | SOTA F1 | Gap |
|---|---:|---:|---:|
| Door | 0.567 | 0.848 | −0.28 |
| Window | 0.249 | 0.709 | −0.46 |
| Stair | 0.143 | 0.857 | −0.71 |
| Appliance | 0.548 | 0.769 | −0.22 |
| Furniture | 0.650 | 0.764 | −0.11 |
| Equipment | 0.186 | 0.926 | −0.74 |
| Parking lot | 0.799 | 0.539 | **+0.26** |

The `Parking lot` row is only a diagnostic box-AP aggregation and must not be
read as beating the source SOTA. Furniture is the closest diagnostic category;
Stair, Equipment and Window show the largest diagnostic gaps. These observations
are historical and must not drive future tuning because test has been exposed.

## 2. Historical diagnostics (quarantined, not a root-cause claim)

The following per-class table records an exploratory held-out observation from
an already exposed test run. It is retained for auditability only. Because test
has influenced discussion in this project and the historical vector reports use
a different manifest, these observations must not be treated as a confirmed
root cause or used to prioritize new experiments.

| Class | val AP50 | historical test AP50 | historical test preds : GT |
|---|---:|---:|---:|
| column | 0.772 | 0.350 | 39:1 |
| bathtub | 0.532 | 0.289 | 90:1 |
| escalator | 0.472 | 0.296 | 121:1 |
| door_revolving | 0.552 | 0.034 | 56:1 |
| elevator | 0.400 | 0.023 | 246:1 |
| ramp | 0.000 | 0.000 | 300:1 |
| escalator_stair | 0.049 | 0.060 | 403:1 |

Possible explanations include rare-class positive scarcity, split-composition
shift, and small-object sensitivity. They are hypotheses only. Future tests must
be designed from training-pool statistics and validation-only diagnostics under
the new split hierarchy, not from these held-out numbers.

## 3. What this implies for the plan

The vector branch delivers the largest confirmed gain on matching validation
manifests (+9.39 AP50 / +6.41 AP50:95 mean across two seeds). The historical
cross-manifest test delta is not a controlled claim. Remaining hypotheses include
rare-class positive starvation, small-object resolution, and training budget, but
these must be tested on a new validation hierarchy rather than selected from test
failure patterns.

## 4. Prioritized attack plan

### Phase 1 — completed diagnostic experiments

1. **Global/per-class threshold calibration:** completed on validation. All
   classes retained threshold `0.05`; no AP improvement.
2. **Rare-negative boost:** completed with a corrected zero-count floor and a
   full-validation gate. It reduced AP and greatly increased prediction count;
   reject this family of naive inverse-frequency sampling schemes.
3. **Class-balanced sampler power:** not yet run under the new validation
   hierarchy; do not tune it from the historical test table.

### Phase 2 — next accuracy experiments

4. **Hybrid stride-4 head:** first run a stride-4 collision/target audit, then
   compare a selective small-object refinement against the stride-8 anchor.
5. **Local/hierarchical attention:** reduce the 4096-token global attention
   cost while preserving a global context token. Measure AP and latency together.
6. **Sparse vector fusion:** compare dense cross-attention with geometry raster
   features or spatially indexed local fusion.

### Phase 3 — speed and capacity

7. **Shared vector/image trunk with late conditioning — completed speed track:**
   `dual_pathway_late`, seed 1337, was trained on `train_new` and selected on
   `val_select` under the frozen schema-v2 manifest. It reduces parameters by
   11.86% versus V2 and reduces synchronized all-class inference latency by
   9.87x (chunk 4) to 15.17x (chunk 35), with 59.08% to 87.28% lower peak
   allocated inference memory. Selected-query training speed is effectively
   unchanged. The late model is not capacity matched to V2 and is not a pure
   stage-placement ablation: it also removes V2 early fusion and
   class-conditioned vector FiLM.
8. **One-pass student distilled from V:** compare image-only and lightweight
   image+SVG students against V as the quality teacher. This remains the next
   accuracy-preserving speed experiment.
9. **Pretraining, longer schedule and EMA:** test independently only after the
   speed/quality anchors and new validation hierarchy are fixed.

### Latest schema-v2 result

`dual_pathway_late` seed 1337 achieved full-role AP50/AP50:95 of
`0.1598/0.0668` on `val_select` and `0.1553/0.0646` on `val_calibration`.
Per-class calibration changed no thresholds from 0.05 and produced no metric gain;
the global validation-only sweep also selected 0.05. The checkpoint was
`best_val_ap.pt`. `test_set` was not accessed. Because historical V2 quality
reports use a different manifest, this result must not be compared to them as a
controlled quality delta; run a same-manifest V2 control before making an
architecture-quality claim.

### Success gate

Do not use the unresolved source-table numbers as an optimization target until
its paper, labels, split, operating point, and evaluator are recovered. The
next internal gate is instead:

1. establish a new image-level `train_new` / `val_select` /
   `val_calibration` hierarchy and freeze its manifest;
2. report quality and synchronized latency on the same validation protocol;
3. require at least two seeds for an architecture claim and a pre-specified
   ablation/control for each causal question;
4. select one final checkpoint and decoder using validation only;
5. run the untouched test set once for the frozen final report.

Only after the source protocol is reproduced may we report an aligned F1/
Weighted-F1 comparison. Until then, report our own AP50/AP50:95 and latency as
an independent detection track, with the source table labeled reference-only.

## 5. Caveats

- The SOTA table's F1 and our AP50 are different metrics on possibly
  different matching rules; before claiming "beat SOTA" we must either (a)
  implement their exact evaluation (per-category F1 at a chosen operating
  point), or (b) state clearly we report detection AP.
- Rare-class AP is high-variance (elevator 40 GT); improvements there need
  the seed-replication discipline used throughout.
