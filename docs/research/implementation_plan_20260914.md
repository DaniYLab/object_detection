# Implementation Plan: DeepSeek-Informed Improvements

**Date:** 2026-09-14
**Source of ideas:** `docs/research/deepseek_transfer_report_20260914.md`
**Target model:** `dual_pathway_v2_thin` — 6.79M params, dual-pathway, stride-8 trunk +
hybrid stride-4 head, query-conditioned CenterNet, 35 classes
**Baseline to beat (caveated):** `val_select` AP50 0.4950 / AP50:95 0.3091, seed 1337,
schema-v2 manifest `203f6358…`, `epoch_030.pt`

> **Standing caveat.** That baseline is measured on a manifest with an 11% train/val
> content leak (§2 of the transfer report). **Phase 1 must complete before any Phase 3+
> result is interpretable.** All deltas measured before re-splitting are provisional.

**Scope decision (user, 2026-09-14):** teacher-data distillation is **out of scope**. We
have exact ground truth; the teacher's estimated boxes add noise. Only reasoning text
would be useful and it is thin (34% of rows zero-box). Recorded in the transfer report §8
for audit only.

---

## Phase 1 — Dedup and re-freeze the manifest  ⚠️ BLOCKING

**Why first.** Every existing AP number is inflated by an unknown amount. Measuring any
improvement against that baseline produces an uninterpretable delta.

### Deliverables

1. `scripts/data/dedup_splits.py` — group images by PNG SHA-256; assign each group wholly
   to one role.
2. A new manifest `data/FloorPlanCAD_original/splits_v3.json` with a new fingerprint;
   `splits_v2.json` left untouched for auditability.
3. `outputs/dedup_report.json` — counts before/after per role, group-size distribution,
   and an explicit duplicate-free assertion.
4. Evaluate the old `dual_pathway_v2_thin` checkpoint on v3 as a **diagnostic only**,
   then cold-train the control on v3 to establish the honest baseline. A v2 checkpoint
   cannot serve as the v3 baseline because repartitioning moves some old-train images
   into v3 validation.

### Algorithm

```
1. Only split the training pool (train_set_1 + train_set_2). test_set is copied
   unchanged, as in v2 — but its duplicates against train are REPORTED, not moved
   (moving them would shrink test and break comparability with historical runs).
2. Compute sha256(png) for every training-pool image.
3. Union-find: images with equal hash form a group.
4. Compute each group's class-multiset (union of member class_counts) so rare-class
   coverage can still be honoured when a group is large.
5. Assign groups (not images) to {train_new, val_select, val_calibration} using the
   existing rare-class greedy strategy, but at GROUP granularity, largest group first.
6. Assert: no group spans two roles; val fractions within tolerance; each role non-empty;
   rare-class coverage no worse than v2 where achievable.
7. Emit new role fingerprints + manifest fingerprint.
```

### Verification (all must pass before Phase 2)

- [x] Re-running the script twice on identical inputs yields fingerprint `405db7a8…`.
- [x] `sum(len(role) for role in roles) == len(training_pool)` (exact partition).
- [x] **No sha256 appears in more than one training/validation role** — the core assertion.
- [x] `test_set` image list byte-identical to v2.
- [x] Every manifest record resolves to an existing `.png` and `_meta.json`.
- [x] Report the inherited test overlap: 610 / 5,502 test images share content with
      the training pool; documenting it is the goal, not changing the frozen test role.

### Acceptance

The old v2 checkpoint measured `0.5450 / 0.3344` on v3 `val_select`, but this is invalid
as a clean baseline: 350 / 1,289 v3 validation images (27.15%) belonged to v2 `train_new`
and may have been seen during training. The increase is a protocol warning, not a model
gain. **Phase 1 is complete at the data level but remains blocking for experiments until
the control is cold-trained on `splits_v3.json`.**

---

## Phase 2 — Protocol hygiene (free, adopt now)

No training. Reduces ambiguity in every later comparison.

1. **Any new head must be droppable at inference** so compared models have identical
   inference cost (V3's MTP ablation discipline). Enforce via a `training_only` flag
   checked in `forward()`.
2. **Moderate-difficulty band** for `src/models/reasoning_selector.py`: derive
   `confidence_threshold` / `spatial_threshold_px` from the threshold band where the
   detector has moderate success rate, instead of the current hand-tuned 0.7 / 8.0.
   Two papers independently converge on this (Prover-V1.5, R1).
3. **Confirm sample masking** in any packed batch path — concatenated samples must not
   attend across boundaries.
4. **Pre-register the metric.** Every phase below is judged on **full-role `val_select`**
   (1,016 images), never the 256-image periodic subset. Evidence: the 256-subset read
   0.3389 at ep30 against a full-role 0.3091 — a +3 AP50:95 overestimate.

---

## Phase 3 — V3 batch-level class balancing + bias term

**Idea source:** transfer report §3.1 (DeepSeek-V3, arXiv 2412.19437v2).

**Hypothesis.** Our per-image balanced sampler plus 1-absent-negative-per-positive is the
"hard per-sequence constraint" V3 measured as *harmful* relative to loose batch-level
balancing, because the tight constraint prevents specialisation. Our 35-class long tail
(counter 1234 → window_bay 2) is exactly the regime this targets.

### Changes

1. **`src/data/dataset.py`** — `get_sample_weights`: keep inverse-frequency weighting but
   compute it at batch scope rather than per-image; expose `balance_scope: {"image",
   "batch"}` so the old behaviour stays runnable as a control.
2. **`train.py`** — add a per-class bias `b_c ∈ R^35`, updated after each step by the sign
   rule: `b_c -= γ` if class c's predicted-positive count exceeds its GT share,
   `b_c += γ` otherwise. **Applied to the center-logit threshold at decode, not to the
   loss** — this is what makes it gradient-free.
3. **`evaluate.py`** — the decoder must accept per-class bias so train and eval agree.

### Ablation (mandatory, like-for-like)

| Arm | Balance scope | Bias |
|---|---|---|
| A0 (control) | image (current) | off |
| A1 | batch | off |
| A2 | batch | on |

Same manifest, same seed, same 30 epochs, same everything else.

### Acceptance
- Report AP50, AP50:95, **and per-class AP** for the long tail (window_bay, window_blind,
  ramp, door_revolving, elevator) — an aggregate gain that hides a long-tail loss is a fail.
- Reject if A2 < A0 on full-role val_select.

---

## Phase 4 — Attention sink + q/k RMSNorm in the cross-attention fusion

**Idea source:** transfer report §3.3 (DeepSeek-V4, arXiv 2606.19348v1).

**Hypothesis.** The absent-class negative query must express "no object of this class
here" through the same softmax that spreads mass over spatial locations. A learnable
per-head sink logit gives it an explicit null attractor, reducing absent-class false
positives — which is what AP50 punishes. The repo already measures this: model B emits
≥1 prediction on 50.24% of absent queries at threshold 0.05.

### Changes

1. **`src/models/detector.py`** — `encode_vector`, the `cross_attention` branch. Replace
   `nn.MultiheadAttention` with an explicit implementation that adds a learnable
   per-head sink logit to the attention denominator (so a head's weights need not sum
   to 1). Take care with the existing all-pad-row guard.
2. **Same site** — `RMSNorm` on query and key immediately before the attention matmul
   (V4's stated reason for being able to drop QK-Clip).
3. Keep `nn.MultiheadAttention` behind a flag as the control arm.

### Ablation

| Arm | Sink | q/k RMSNorm |
|---|---|---|
| B0 (control) | off | off |
| B1 | on | off |
| B2 | on | on |

### Acceptance
- Primary metric: **absent-class false-positive rate** (existing metric in
  `experiment_log.md`) must drop, not just AP.
- Report per-class AP; sink logits are inspectable — log their learned values.

---

## Phase 5 — Vector-encoder positional encoding for chained strokes

**Idea source:** transfer report §3.2 (own finding; the DeepSeek-OCR-derived original
proposal was **withdrawn** — see §3.2.1 there).

**Verified defect.** `src/models/vector_encoder.py` has **no positional encoding**
(`input_projection` + `type_embedding` only), yet:
- 42.6% of consecutive stroke pairs share an endpoint → document order is a chain.
- `sample_strokes` preserves that order (`index_select(0, sorted(keep))`), so the signal
  is present in the input and simply never surfaced to the model.

**Hypothesis.** A stair/ramp is a run of short segments whose identity is the chain, not
any single segment's absolute position. The image branch cannot resolve the run at stride
8; the vector branch has exact geometry in the right order but is not told the order.

### Changes

Two arms — run both, they answer different questions:

1. **C1: learned sequence position embedding** — `pos_embed(1, n_max, model_dim)` added
   after `input_projection`, mirroring `image_pos_embed`. Conventional baseline;
   ~262k params at `n_max=1024`.
2. **C2: relative-geometry attention bias** — bias derived from endpoint distance between
   token pairs, so connectivity (shared endpoints) is expressed geometrically rather than
   by index. Fewer parameters and matching the actual structure; the chain is geometric,
   not positional.
3. **C0: control** — current encoder unchanged.

### Acceptance
- **Must report class-stratified AP on stair / ramp / escalator_stair.** These are the
  classes the hypothesis is about; an overall-mAP-only read cannot detect success or
  failure here (see the `min_overlap` lesson in `experiment_log.md`, where a +5.9 AP50
  5-epoch gain shrank to +0.8 at 30 epochs).
- Reject if no stair/ramp improvement.

---

## Phase 6 — Re-run the stride-4 head question on the clean manifest

Not from the DeepSeek survey — this is the repo's own open hypothesis, re-opened because
Phase 1 invalidates the original measurement.

The thin-head result (+2.6–3.1 AP50:95, stair 0.104 → 0.304) was measured on the leaking
manifest. Re-verify it on `splits_v3.json` before relying on it. If it survives, it
remains the best-verified quality lever and should be combined with Phases 3–5.

---

## Phase 7 — Multi-scale training  ⭐ HIGHEST PRIORITY AFTER PHASE 1

**Idea source:** `docs/research/glm_paddleocr_transfer_report_20260914.md` §5.1
(PP-OCRv6, arXiv 2606.13108v1).

**Verified defect.** `PairedTrainTransform` (`src/data/transforms.py`) applies flip,
90° rotation, and photometric jitter. There is **no scale augmentation** — confirmed
empirically: 200 applications of the train transform to a thin 40×5 box on a 1000×1000
canvas produce exactly **2 distinct box sizes** (original + 90°-rotated) and a min-side that
is **always 2.56 px**.

**Hypothesis.** Our measured bottleneck is objects whose apparent size falls below one
stride-8 cell (stair median min-side ≈5.1 px). Scale diversity makes a given stair sometimes
above and sometimes below one cell, which is the supervision that teaches sub-cell cues.
PP-OCRv6 trains across 7 scales and holds >81% Hmean at 2.83× while its predecessor drops to
66%; its resolution CV is 5.19% vs 8.02%.

### Changes

1. **`src/data/transforms.py`** — add scale randomisation to `PairedTrainTransform`,
   applied to image and boxes together before the final square resize. Suggested range
   0.7×–1.4× (configurable), with the existing `validate_half_open_boxes` checks retained
   after scaling.
2. Keep the scale factor in the sample metadata so the applied scale is auditable.
3. Make the range a CLI arg so the control arm is `--scale-range 1.0 1.0` (no-op).

### Ablation

| Arm | Scale range |
|---|---|
| D0 (control) | fixed 1.0 (current behaviour) |
| D1 | 0.85–1.15 |
| D2 | 0.70–1.40 |

### Acceptance
- **Class-stratified AP on stair / ramp / escalator_stair**, plus overall.
- Watch for a regression on large classes (wall, floor_plan_area) — aggressive scaling can
  hurt large-object localisation. Report per-class, not just mAP.
- Reject D2 if it beats D1 overall but loses on large classes; prefer the narrower range.

---

## Phase 8 — Auxiliary deep supervision at intermediate strides

**Idea source:** transfer report §5.2 (PP-OCRv6).

**Verified gap.** The hybrid stride-4 head exists (`head_stride=4`;
`_resolve_head_features` bilinearly upsamples the stride-8 trunk to a 128×128 grid), but
**only the final output is supervised**. There are no auxiliary heads.

**Mechanism.** PP-OCRv6 adds independent aux heads at **P2/P3/P4** (strides 4/8/16) with the
same targets, weighted **0.4 / 0.3 / 0.2**, and **removes them at inference** — ablation
+0.53 Hmean at zero inference cost. Their stated rationale: without aux supervision,
gradients must traverse the whole FPN to reach the backbone.

### Changes

1. **`src/models/detector.py`** — optional aux CenterNet heads at stride 8 and stride 16
   (in addition to the existing stride-4 output head).
2. **`src/training/losses.py`** — sum aux losses with configurable weights.
3. **Inference:** aux heads must be skipped. Requires the Phase-2 protocol item
   (`training_only` flag) to be implemented first, so compared models have identical
   inference cost.
4. **Checkpoint compatibility:** adding heads changes `model_config_fingerprint`. Existing
   checkpoints must stay loadable — make the heads opt-in via config with a default of off.

### Ablation

| Arm | Aux heads |
|---|---|
| E0 (control) | none |
| E1 | stride-8 only (w=0.4) |
| E2 | stride-8 + stride-16 (0.4 / 0.3) |

### Acceptance
- Report **inference latency** alongside AP to confirm the heads are genuinely free at
  inference.
- Reject if latency rises or AP does not improve.

---

## Phase 9 — SVG-based targeted synthesis for stair/ramp (speculative)

**Idea source:** transfer report §5.3 (hard-case mining loop, PaddleOCR-VL v1 §3.3), applied
where we hold an advantage they lacked: exact ground-truth geometry as vector primitives.

**Proposal.** Procedurally generate floor plans enriched in stair/ramp geometry by
recombining and perturbing existing stroke primitives, deriving box labels from the same
primitives — zero annotation cost.

**Why it is plausible.** `stair` is the repo's most decisive single-class gain when it works
(thin head 0.104 → 0.304, +0.201, 214 GT, not rare), and it is thin (min-side ≈5.1 px,
aspect 7.0; ramp 20.0).

**Honest caveats.**
- Synthetic plans may not match the real distribution.
- The `escalator_stair` case (172 px median side — large but zig-zag) suggests part of the
  problem is **inconsistent annotation, not resolution**. If so, synthesis will not help
  that class.
- This adds signal rather than reshaping existing signal, which is why it is worth trying —
  but it is the highest-variance item here.

### Acceptance
- Gate on **real `val_select`**, never on synthetic val.
- Time-box. If the first generator version does not move real stair AP, stop.

---

## Phase 10 — Dice term alongside focal on the heatmap (low confidence)

**Idea source:** transfer report §5.4 (PP-OCRv6 — their largest loss-side ablation gain,
+1.15, came from pairing Focal with Dice; we have only the Focal half).

**Verified gap:** `src/training/losses.py` has `focal_loss`, `l1_loss_masked`,
`centernet_loss`. **No Dice.**

**Honest caveat, and why this is last.** Their detection maps are dense shrink/threshold/
binary regions; ours are sparse Gaussian peaks, and 33.6% of our boxes already have
gaussian radius 0 at stride 8 — a known pathology. The transfer is not clean and this may
well be a null. Run it as a single A/B only after Phases 7–8 are resolved.

---

## Phase 11 — Three-expert consensus label verification (diagnostic)

**Idea source:** transfer report §5.5 (PaddleOCR-VL v1.6 §unreliable-supervision mining).

**Proposal.** Two independent detectors as experts (our CenterNet + a Faster R-CNN
baseline): if ≥1 agrees with the label, keep it; if all disagree but the experts agree with
each other, take the consensus; otherwise flag.

**Complements Phase 1.** Dedup removes *duplicated images*; this catches *contradictory
labels on identical images* — the verified `0263-0005` vs `0326-0005` case (same pixels, one
labelled `escalator` and the other `door_double` at the **same bbox**).

**Status:** correctness check, not a bet. No ablation needed; it either finds mislabels or
it does not. Schedule alongside Phase 1.

---

## Sequencing and dependencies

```
Phase 1 (dedup) ────┬─► Phase 6 (re-verify thin head)
                    ├─► Phase 11 (label consensus)   [no dependency]
                    │
Phase 2 (hygiene) ──┼─► (prerequisite for Phase 8's droppable heads)
                    │
Phase 3 (V3 bias) ──┤   [independent]
Phase 4 (sink) ─────┤   [independent]
Phase 5 (vec pos) ──┤   [independent]
Phase 7 (multi-scale)┤  [independent, highest value]
Phase 9 (synthesis) ─┤  [independent, speculative]
Phase 8 (aux heads) ─┘  [needs Phase 2]
Phase 10 (Dice)         [last, low confidence]
```

**Order:** 1 → 2 → 11 → 6 → **7** → 3 → 4 → 5 → 8 → 9 → 10.

Rationale: 1 is blocking and 11 rides along with it (both are data-correctness). 2 is free.
6 re-establishes which existing lever is real. **7 is placed early because it is the
strongest verified gap in either survey** — a confirmed missing augmentation that targets the
measured bottleneck, with no parameter change. 3/4/5 then in cost order. 8 needs 2's
droppable-head protocol. 9 is speculative and gated. 10 is last and may be a null.

**Not to be combined.** Every phase needs its own control arm. Stacking before measuring
individually would repeat the `dual_pathway_late` mistake documented in
`experiment_log.md`, where a confounded comparison could not be attributed.

---

## Compute budget

Each 30-epoch run is ~692 s/epoch on the local RTX 5070 Ti → **~5.8 h per arm**.

| Phase | Arms | GPU-hours | Notes |
|---|---:|---:|---|
| 1 | 0 | 0 | hashing + one eval |
| 2 | 0 | 0 | code only |
| 11 | 0 | ~2 | inference-only label audit |
| 6 | 2 | ~12 | thin vs control on v3 |
| **7** | 3 | ~17 | D0/D1/D2 — multi-scale |
| 3 | 3 | ~17 | A0/A1/A2 |
| 4 | 3 | ~17 | B0/B1/B2 |
| 5 | 3 | ~17 | C0/C1/C2 |
| 8 | 3 | ~17 | E0/E1/E2 (aux heads) |
| 9 | 2 | ~12 | synthesis on/off (speculative, gated) |
| 10 | 2 | ~12 | Dice on/off (low confidence) |
| **Total** | **24** | **~123 h** | ≈5 days wall-clock if serialized |

**Pruning guidance.** This is a menu, not a commitment. If Phase 7 (multi-scale) or
Phase 3 (V3 bias) produces a clear, replicated gain, adopt it and stop before the
low-confidence tail (Phases 9–10). The earlier phases are ordered by evidence strength, so
stopping early is the expected outcome, not a shortfall.

Local RTX 5070 Ti is preferred over Colab for this: 24 sequential arms with per-arm
controls is exactly the workload Colab's ephemeral VMs handle badly (session loss,
assignment exhaustion, no persistence across arms — see the 2026-09-13 incident). At ~5
days serialized it also fits a local overnight/weekend schedule.

---

## Risks

| Risk | Mitigation |
|---|---|
| Phase 1 shrinks val/test enough to raise variance | Report role sizes; if a role drops below ~800 images, note the widened CI and prefer larger `val_calibration` |
| Dedup changes class balance → rare classes vanish from val | Preserve rare-class coverage during group assignment; assert no rare class loses its last val instance |
| Phase 3's bias term destabilises early training | γ small (V3 used 0.001); warm up after 500 steps; log `b_c` trajectory |
| Phase 4's explicit attention is slower than `nn.MultiheadAttention` | Profile; sink+norm are cheap, but validate the all-pad-row guard under both |
| Phase 5's gain is invisible in aggregate mAP | Pre-registered stair/ramp-stratified metric (Phase 2 item 4) |
| Any phase's gain is actually the §2 leak re-appearing | Impossible after Phase 1 — this is why Phase 1 is blocking |

---

## Provenance

**Idea sources:**
- `docs/research/deepseek_transfer_report_20260914.md` — 14 DeepSeek papers (2026-09-13/14);
  source of Phases 3, 4, 5 and the §2 data-integrity defect
- `docs/research/glm_paddleocr_transfer_report_20260914.md` — 6 GLM/PaddleOCR papers
  (2026-09-14); source of Phases 7, 8, 9, 10, 11

**Evidence:**
- Authoritative results record: `docs/experiments/experiment_log.md`
- Frozen v2 manifest: `data/FloorPlanCAD_original/splits_v2.json`, fp
  `203f63588e906637f4dbcf6d1a001e67d0e688016c9b0c799c3ea89b438e8e9a`
- Measured bottleneck evidence: `experiment_log.md`, per-class diagnosis of the V2 control
  (stair 0.104, ramp 0.122, escalator_stair 0.036; median min-side ≈ 5.1 px)
- Stroke audit (one primitive per path): `docs/experiments/stroke_audit.md`, 2026-08-24
- No-scale-augmentation defect: `src/data/transforms.py` (`PairedTrainTransform`) +
  empirical test (200 samples → 2 distinct box sizes, min-side always 2.56 px)
- No-Dice-loss defect: `src/training/losses.py`
- No-aux-heads defect: `src/models/detector.py` (`_resolve_head_features`)

**Change log:**

| Date | Change |
|---|---|
| 2026-09-14 | Initial plan: Phases 1–6 from the DeepSeek survey |
| 2026-09-14 | Added Phases 7–11 from the GLM/PaddleOCR survey; re-ranked so Phase 7 (multi-scale, the strongest verified gap) runs early; budget 11 → 24 arms |
| 2026-09-14 | Teacher-distillation phase dropped at user direction (exact GT exists; teacher boxes are noise) |
