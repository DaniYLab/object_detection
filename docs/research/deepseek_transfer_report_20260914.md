# DeepSeek Literature Survey → FloorPlanCAD Detector: Transfer Report

**Date written:** 2026-09-14
**Papers surveyed:** 14 (`docs/research/deepseek/`, arXiv 2401–2606 + V4.1 tech report)
**Corpus size:** ~24,400 lines of extracted text (`.colab/ds_txt/`)
**Target model:** `dual_pathway_v2_thin` — 6.79M params, dual-pathway (image + SVG stroke
vector), stride-8 CNN trunk + hybrid stride-4 head, query-conditioned CenterNet, 35 classes
**Target metric:** full-role `val_select` AP50 **0.4950** / AP50:95 **0.3091** (seed 1337,
schema-v2 manifest `203f6358…`, epoch-30 checkpoint)
**External target:** Faster R-CNN R101 AP50 **0.602** (FloorPlanCAD paper, ICCV 2021)

---

## 0. TL;DR

Three results, in order of importance.

1. **A data-integrity defect was found that invalidates every AP number in this repo.**
   11.1% of `val_select` images (113/1016) are **byte-identical** to an image in
   `train_new`; 18.9% of `test_set` (1040/5502) is. The split is at image-id level, but
   FloorPlanCAD re-uses the same PNG under different ids. This was *not* what I set out to
   find — it surfaced while testing a dedup idea from DeepSeek-LLM. See §2.

2. **The vector encoder is missing the stroke order.** 42.6% of consecutive stroke pairs
   share an endpoint, so the SVG document order is a chain that encodes object structure —
   and `VectorEncoder` has no positional encoding to see it. This is a defect in our own
   model, found by testing a DeepSeek-OCR idea and then *disproving* the idea against an
   audit this repo had already run. See §3.2.

3. **Of 14 DeepSeek papers, three ideas are worth GPU time.** None of them fixes the
   measured bottleneck (sub-stride thin objects) by itself. Two are cheap model/training
   edits, one is free protocol hygiene. See §3 for the ranked list, §4 for what is
   explicitly rejected and why, and §5 for the ordered plan.

**Not planned:** teacher-data distillation. We have exact ground truth, so the teacher's
estimated boxes add noise rather than signal. Retained as an audit trail in §8 only.

---

## 1. Why survey these papers at all — and the honest scope limit

The papers are LLM systems papers. The target is a 6.79M-param detector trained on 8,129
images for 30 epochs. Most LLM machinery does not transfer, and the three independent
review passes converged on that conclusion without being told to.

The single most useful thing the survey produced was not a mechanism but a **diagnostic
habit**: DeepSeek-LLM's 89.8%-vs-22.2% cross-dump deduplication result prompted me to
check our own corpus for near-duplicates. That check found §2.

**Stated plainly, so it is not lost:** the measured bottleneck is stair/ramp median
min-side ≈ 5.1 px, below the stride-8 cell. That is a resolution/representation problem.
Nothing in these 14 papers addresses it directly. The gap to AP50 0.602 will be closed by
representation and label quality, not by any technique below.

---

## 2. Data integrity finding (highest-value result) — discovered 2026-09-13/14

### 2.1 The defect

`data/FloorPlanCAD_original/splits_v2.json` (fingerprint `203f63588e906637…`, schema v2,
frozen 2026-08-27) assigns images to roles by **image id**, e.g.
`train_set_1/0263-0005`. But FloorPlanCAD contains many **byte-identical PNGs under
different ids** — `train_set_1/0263-0005` and `train_set_1/0326-0005` are the same file.

Verified by SHA-256 of the raw PNG bytes:

| Measure | Value |
|---|---:|
| Exact-duplicate PNG groups | **1,163** |
| Groups straddling ≥2 roles | **806** |
| Images sitting in a duplicate group | 3,474 / 15,663 |
| Total image slots | 15,663 |
| Unique image contents | **13,352** |
| Redundant slots | **2,311** |

### 2.2 Leakage into each role

Two different questions, two different numbers — both are correct, do not conflate them.

**Narrow (twin specifically inside `train_new`) — this is the number that inflates val:**

| Role | Images with an exact twin in `train_new` |
|---|---:|
| `val_select` | **113 / 1016 = 11.12%** |
| `val_calibration` | 120 / 1016 = 11.81% |
| `test_set` | **1,040 / 5502 = 18.90%** |

**Broad (shares content with *any* other role, including `val_cal`↔`val_sel`):**

| Role | Images sharing content with another role |
|---|---:|
| `train_new` | 1,135 / 8129 = 13.96% |
| `val_select` | 186 / 1016 = 18.31% |
| `val_calibration` | 193 / 1016 = 19.00% |
| `test_set` | 1,186 / 5502 = 21.56% |

Split-pair group counts: `test_set↔train_new` 615, `test_set↔val_select` 109,
`test_set↔val_calibration` 102, `train_new↔val_select` 100,
`train_new↔val_calibration` 98, `val_calibration↔val_select` 35.

### 2.3 Consequence

- The headline **0.4950 / 0.3091** is measured on a val set that is ~11% memorised.
- The historical held-out test numbers (`B floorplan_base` 0.3896/0.2170, and the
  quarantined V/V2 test rows) carry the same defect at ~19%.
- Part of the apparent 0.495 → 0.602 gap to Faster R-CNN may be a split artifact rather
  than a capability gap.
- **Every architecture comparison in `docs/experiments/experiment_log.md` is affected**,
  including the replicated ones (conditioning +5.37 AP50, vector +9.39 AP50, thin-head
  +2.6–3.1 AP50:95). Replicated effects are unlikely to be *created* by an 11% leak, but
  their magnitudes are unverified.

### 2.4 A second, independent problem: inconsistent labels

In sampled duplicate pairs the PNGs are identical but `_meta.json` differs:

| Pair | Identical PNG | Identical class+bbox set |
|---|---|---|
| `0263-0005` vs `0326-0005` | ✅ | ❌ — one labels `escalator`, the other `door_double`, **at the same bbox** |
| `0369-0029` vs `0371-0029` | ✅ | ✅ |
| `0024-0032` vs `0990-0032` | ✅ | ✅ |

So the corpus contains the same drawing with contradictory annotations. This is worth
auditing separately from the split fix.

### 2.5 A wrong turn worth recording

My first measurement said same-prefix images were pixel-identical (`IoU = 1.000`, 400/400
pairs). That was an **artifact of an inverted threshold** — these PNGs are white-ink on a
black ground (`min 0, max 179, mean 8.1`), and I thresholded `< 0.5` as "ink", selecting
every pixel. Corrected (`ink = pixel > 0.15`), same-prefix mean ink-IoU is **0.048**.

Lesson, and the reason the fix below keys on content hash: **the filename prefix is not
the duplicate key.** `NNNN` groups a drawing family but different `NNNN` can be the same
drawing, and same `NNNN` is usually *not* the same drawing. Only the PNG SHA-256 is.

### 2.6 Recommended fix

Group by PNG SHA-256, assign **every** member of a group wholly to one role (largest-group
-first, respecting target fractions), rebuild and re-freeze the manifest with a new
fingerprint, then re-baseline existing checkpoints on it. No training required. Until
this is done, treat all schema-v2 AP numbers as **upper bounds** and never quote a delta
across a re-split without re-running the control.

---

## 3. Transferable ideas, ranked

### 3.1 ★ Batch-level class balancing via a bias term (DeepSeek-V3, arXiv 2412.19437v2)

**Mechanism.** V3 replaces auxiliary load-balancing *loss* with a per-expert bias `b_i`
added only to the top-K routing score; after each step `b_i -= γ` if overloaded, `+= γ` if
underloaded (γ = 0.001 for the first 14.3T tokens, 0.0 for the final 500B). The gating
value multiplied into the output still comes from the raw affinity, so **no gradient
interference**. Ablation, aux-loss-free vs aux-loss-based: HumanEval 40.2→46.3, MBPP
59.2→61.2, MATH 37.2→39.6. Separately: batch-wise balancing ≈ aux-loss-free (val loss
2.253 vs 2.258 @1B), and both beat the tighter per-sequence constraint — V3's stated
reason is that the looser scope **lets units specialize**.

**Why it fits us.** Our `--sampler balanced --balance-power 0.5` plus
`--neg-queries-per-pos 1` is exactly the hard per-sequence constraint V3 measured as
harmful. Our 35 classes are heavily long-tailed (counter 1234 → window_bay 2).

**Concrete action.** (a) Move frequency balancing from per-image to per-batch so one floor
plan need not be class-representative. (b) Maintain a per-class bias `b_c`, updated by the
sign rule, applied to the center-logit threshold rather than to the loss.
**Cost:** ~10 lines. **Confidence:** medium-high. **Do first.**

### 3.2 Stroke-sequence ordering: the vector encoder has no positional encoding (revised 2026-09-14)

> **This section replaces an earlier draft.** The draft proposed "sub-primitive
> tokenization" (splitting each stroke into 4 sub-tokens, per DeepSeek-OCR's Slow
> Perception ruler). **That proposal was wrong and is withdrawn** — see §3.2.1. What
> follows is the corrected finding, which the data actually supports.

#### 3.2.1 Why the original proposal was withdrawn

`docs/experiments/stroke_audit.md` (2026-08-24, over all 15,663 drawings) establishes:
`M` count == `L` + `A` count on every split (train: 7,133,394 M vs 6,279,041 L +
854,353 A). **Every SVG path element already contains exactly one primitive** (`M→L` or
`M→A`); only `M`, `L`, `A` commands exist, no Bezier/polyline. So `strokes.py`'s
one-token-per-primitive layout is already the finest granularity available — there are no
sub-primitives to emit. I verified the token shape directly (802 strokes × 12 dims for
`train_set_1/0000-0002`).

I also verified the truncation premise: on `train_new`, strokes/drawing median 118, mean
234, p99 1,711, max 4,452. At `n_max=1024`, **96.6% of drawings are fully covered** and
3.4% truncated. So "thin runs collapse to a handful of tokens" is false — a stair is
represented by many tokens, not few.

**Lesson recorded:** the idea was plausible from the paper but contradicted by an audit
this repo had already run. Read the existing audit before importing an external fix.

#### 3.2.2 The real defect: document order carries information the model cannot see

Two measurements on the data:

| Measurement | Result |
|---|---|
| Distinct stroke endpoints (`0000-0002`, 802 strokes) | 988; **550 (55.7%) shared by >1 stroke** |
| Consecutive stroke pairs sharing an endpoint | **341 / 801 = 42.6%** |

So the SVG document order is **chained**: consecutive primitives genuinely connect, and
that chain encodes object structure (a wall run, a symbol outline) that the token
coordinates alone do not express. `sample_strokes` preserves this order — both the
weighted and the fallback path return `index_select(0, sorted(keep))`, so subsampling
never permutes.

But `src/models/vector_encoder.py` has **no positional encoding**: it is
`input_projection(12→256) + type_embedding(line/arc)`, then self-attention blocks
(`VectorEncoderBlock`), then a norm. Nothing tells the encoder that token *i* follows
token *i−1*. The coordinates are absolute, so the model can learn *where* each primitive
is, but not *what it is chained to*.

**Why this matters for the measured bottleneck.** A stair or ramp is a run of short
segments whose identity comes from the chain, not from any single segment's absolute
position. The image branch sees the run at 8 px resolution and cannot resolve it; the
vector branch has the exact geometry in the right order but is never told the order. This
is the one place where the §1 scope limit does not apply — it is a representation defect
in our own model, not an LLM technique.

#### 3.2.3 Concrete proposal

Add explicit positional information to the vector encoder, faithful to the chained
document order. Two options, both small:

1. **Learned sequence position embedding** (mirrors `image_pos_embed` in
   `detector.py`): `pos_embed = nn.Parameter(zeros(1, n_max, model_dim))`, added after
   `input_projection`. ~1024 × 256 ≈ 262k params at n_max=1024.
2. **Relative-geometry bias in attention** — cheaper and better matched to the data:
   since 42.6% of consecutive pairs share an endpoint, an attention bias derived from
   the endpoint distance between token pairs encodes connectivity without a position
   table, and is permutation-faithful rather than index-based.

Option 2 is more principled here (the chain is geometric, not index-based) and adds fewer
parameters; option 1 is the conventional baseline and worth running as the control.

**Cost:** small, localized to `vector_encoder.py`. **Confidence:** medium — the defect is
verified, the fix's magnitude is not. **Measure by class-stratified AP on stair/ramp**
(the same discipline §3.2.1 argues for), not overall mAP.

**Sequencing:** this touches the model, so it must be measured against a re-split
baseline (§5 item 1), or the effect will be indistinguishable from the 11% leak.

### 3.3 ★ Attention sink + q/k RMSNorm in the cross-attention fusion (DeepSeek-V4, arXiv 2606.19348v1)

**Mechanism.** V4 adds learnable per-head **sink logits** to the attention denominator so a
head's attention need not sum to 1 (may be near 0), and applies **RMSNorm on every query
head and KV entry** immediately before core attention — stated as the reason they could
drop QK-Clip. Also: partial RoPE + a sliding-window branch alongside compressed entries,
so local detail is never lost to compression.

**Why it fits us.** Our absent-class negative query must express "no object of this class
here" through the *same* softmax that spreads mass over spatial locations. A sink logit
gives it an explicit null attractor. Absent-class false positives are precisely what AP50
punishes, and the repo already measures this (B: 50.24% of absent queries emit ≥1
prediction at threshold 0.05). Separately, our SVG features are 12-dim projections with
heterogeneous scale; saturated logits are a plausible silent failure.

**Cost:** a handful of parameters + one norm layer. **Confidence:** medium. Low risk.

### 3.4 Free protocol improvements (adopt regardless)

- **Droppable-at-inference heads** (V3): the MTP module is discarded at inference so
  compared models have identical inference cost. Adopt for any new head, or 30-epoch
  comparisons are not like-for-like.
- **Moderate-difficulty band selection** (Prover-V1.5 keeps only ~4.5k theorems where SFT
  success is moderate; R1 makes the same argument for its RM). Directly applicable to
  `src/models/reasoning_selector.py`, whose `confidence_threshold=0.7` and
  `spatial_threshold_px=8` are currently hand-tuned constants with no stated derivation.
- **Sample masking in packed batches** (V3): confirm concatenated samples are mutually
  invisible in the loss.
- **Translation-invariant geometry augmentation** (DeepSeek-OCR): translate drawing and
  boxes together. Note their version *centres* the ground truth — wrong for a localisation
  task; use plain translation.

---

## 4. Explicitly rejected, with reasons

| Idea | Source | Why not |
|---|---|---|
| Optical compression of the SVG stream | DeepSeek-OCR | Analogy is real but **inverted**: their ~10× budget is lossless at *character* granularity; our information unit is geometry at ~5 px, ~1–2 cells at stride 4. Compression discards exactly the sub-grid detail we already fail on. |
| MLA / KV-cache compression | V2, V3, V4, V4.1 | CenterNet has no KV cache and no autoregressive decode. The 93.3% / 5.76× headlines mean nothing here. |
| MoE routing (fine-grained + shared experts) | DeepSeekMoE, V2, V3 | Precondition is 100B–2T tokens. At 8k images you cannot grow specialised experts; the routing machinery would consume the stride-4 head's budget. |
| GRPO / PPO / RL loop | DeepSeekMath, R1, Math-V2, Prover | Detection is already dense-supervised against perfect GT boxes. There is no reward to discover, no reward model, no sampling budget. Math-V2's verifier premise is absent — our verifier *is* the ground truth. |
| FP8 training, DualPipe, mHC, CSA/HCA, CSA2/CED, FP4 KV | V3, V4, V4.1 | Need Hopper/multi-GPU scale. V3's own FP8 error bound (<0.25%) requires 1×128 tile + 128×128 block scaling *plus* 128-element FP32 promotion on CUDA cores. On 6.8M params under bf16 our binding constraint is data, not FLOPs. |
| Sparse attention / DSA / lightning indexer | V3.2 | Our longest attention is 1,024 stroke tokens at dim ≈256 (~0.3 GFLOP, well under 1% of runtime). Optimises a cost we do not pay. |
| Scaling-law exponents | DeepSeek-LLM | Fitted over 1e17–3e20 FLOPs. We are **data-limited**, so the paper's "spend on the model" prescription *inverts* here. Do not use these numbers to justify widening. |
| TTA / ensembling as a route to 0.602 | DeepSeekMath | The Maj@K-vs-Pass@K result: RL improved Maj@K but **not** Pass@K — gains came from sharpening the distribution, not adding capability. Read as: TTA gives small real gains then plateaus. |
| "More data of the same kind" | DeepSeekMath (Table 8/9) | arXiv-only corpora gave **no improvement or degradation** on every math benchmark. Adding more of the same will not close a 10-point gap. |
| Truncate-and-resume, RMaxTS | Prover-V1.5 | Require an autoregressive sequence decoder; our head is a dense per-pixel predictor. Revisit only if moving to a DETR-style query head. |
| MTP in object space | V3 | Correct rationale for a data-bound regime, but the object-ordering problem is unsolved and the gain at 6.8M params is unproven. Speculative. |

---

## 5. Ranked action list

| # | Action | Source | Cost | Confidence | Blocks others? |
|---|---|---|---|---|---|
| 1 | **Dedup manifest by PNG hash, re-freeze, re-baseline** | own finding (§2) | Low (no training) | High (it is a defect, not a bet) | **Yes — everything else** |
| 2 | Batch-level class balance + bias term | V3 §3.1 | ~10 lines | Medium-high | No |
| 3 | Attention sink + q/k RMSNorm in fusion | V4 §3.3 | Low | Medium | No |
| 4 | **Vector-encoder positional encoding for chained strokes** | own finding (§3.2) | Low | Medium | After 1 |
| 5 | Protocol: droppable heads, moderate-band thresholds, sample masking | V3, Prover §3.6 | Free | — | No |

**Recommended order:** 1 first (it invalidates the baseline every other item would be
measured against), then 5 (free), then 2 and 3 (cheap, independent), then 4.

**Dropped from the plan at user direction (2026-09-14):** teacher-data distillation
(former §3.4) and the auxiliary reasoning-prefix head (former §3.5). The user's reasoning,
which I agree with: **we already have exact ground truth, so the teacher's estimated boxes
add noise rather than signal.** Only the reasoning *text* would be of any use, and per the
verification in §3.4 it is thin (34% of rows carry zero boxes; mean 5.5 boxes/image against
a GT mean of 4.8) and self-described as estimates. §3.4 and §3.5 are retained below in
"Superseded sections" for audit trail only, not as planned work.

---

## 6. Provenance ledger

**Survey date:** 2026-09-13 (extraction) → 2026-09-14 (write-up)
**Papers:** 14 PDFs in `docs/research/deepseek/`; text extracted to `.colab/ds_txt/`
(`pdftotext`, ~24,400 lines). Read in full across three parallel review passes, each
asked to be explicitly skeptical about LLM→small-detector transfer.

| arXiv / doc | Short name | Year |
|---|---|---|
| 2401.02954v1 | DeepSeek LLM (scaling laws) | 2024-01 |
| 2401.06066v1 | DeepSeekMoE | 2024-01 |
| 2401.14196v2 | DeepSeek-Coder | 2024-01 |
| 2402.03300v3 | DeepSeekMath (GRPO) | 2024-02 |
| 2405.04434v5 | DeepSeek-V2 (MLA) | 2024-05 |
| 2406.11931v1 | DeepSeek-Coder-V2 | 2024-06 |
| 2408.08152v1 | DeepSeek-Prover-V1.5 | 2024-08 |
| 2412.19437v2 | DeepSeek-V3 | 2024-12 |
| 2501.12948v2 | DeepSeek-R1 | 2025-01 |
| 2510.18234v1 | DeepSeek-OCR | 2025-10 |
| 2511.22570v1 | DeepSeekMath-V2 | 2025-11 |
| 2512.02556v1 | DeepSeek-V3.2 (DSA) | 2025-12 |
| 2606.19348v1 | DeepSeek-V4 | 2026-04 |
| `DeepSeek_V41_Tech_Report.pdf` | DeepSeek-V4.1-Flash (KV compression) | 2026 (no arXiv id) |

**Evidence sources for §2** (all reproducible):
- `data/FloorPlanCAD_original/splits_v2.json` — fp `203f63588e906637f4dbcf6d1a001e67d0e688016c9b0c799c3ea89b438e8e9a`
- PNG SHA-256 over `data/FloorPlanCAD_original/{train_set_1,train_set_2,test_set}/*.png`
- `_meta.json` `source.image_sha256`, `fingerprint`, `instances[].bbox_px`

**Related documents:**
- `docs/experiments/experiment_log.md` — authoritative results record (the only one)
- `docs/archive/fabricated_20260905/` — archived docs with invented numbers; do not cite
- `docs/research/ground_truth.md` is **archived and unreliable** (see the archive dir)

**Caveats on this report's own numbers:**
- §2 counts are exact (SHA-256), not sampled — except the annotation-consistency table
  (§2.4), which is 3 hand-checked pairs. A full audit of duplicate-pair label agreement
  has not been run.
- §3 confidence ratings are judgement calls from three reading passes, not measurements.
  Only §3.2 has a plausible mechanism-level link to the measured bottleneck; the rest are
  cheaper bets.
- The 0.4950/0.3091 baseline quoted throughout is itself subject to the §2 defect. After
  re-splitting, **re-measure before comparing anything to it.**
- §S1 (teacher distillation, §8): the teacher's *box* quality against GT is **not
  measured** — only counts and feature/image-id alignment are verified. Since this item is
  dropped from the plan, the measurement is no longer on the critical path; it is recorded
  only so the option can be re-opened with evidence if ever needed.

---

## 7. Change log

| Date | Change |
|---|---|
| 2026-09-13 | Extracted 14 PDFs to text; ran three parallel review passes; discovered the duplicate-image defect while testing DeepSeek-LLM's dedup idea |
| 2026-09-14 | Verified defect counts by SHA-256; verified teacher-data shapes and set-equality; wrote this report |
| 2026-09-14 | **Revised §3.2.** Original draft proposed "sub-primitive stroke tokenization" (from DeepSeek-OCR / Slow Perception). Withdrawn: `stroke_audit.md` (2026-08-24, all 15,663 drawings) proves every SVG path holds exactly one primitive (`M` count == `L`+`A` count), so no sub-primitives exist to emit, and 96.6% of drawings are fully covered at `n_max=1024`. Replaced with the verified defect: `VectorEncoder` lacks positional encoding while document order is 42.6% chained. |
| 2026-09-14 | **Dropped teacher-data distillation** (former §3.4/§3.5) at user direction: exact ground truth exists, so teacher boxes are noise. Moved to §8 as audit trail. |
| 2026-09-14 | Re-ranked §5: dedup (1) → protocol hygiene (2) → V3 bias term (3) → attention sink (4) → vector positional encoding (5). |



---

## 8. Superseded sections (audit trail — NOT planned work)

Retained so the reasoning is traceable. Both were dropped on 2026-09-14: we have
exact ground truth, so teacher boxes add noise. See §5 for the decision.

### S1. Filter the 15,663 teacher samples, then use as SFT targets (DeepSeek-R1, arXiv 2501.12948v2)

**Mechanism.** R1's own §F.1 is the relevant result: a 32B model **distilled** from R1
scores AIME 72.6, while a same-size model trained with the full RL pipeline scores 47.0;
a 1.5B distilled model beats GPT-4o (28.9 vs 9.3). The paper's conclusion: small models
"require enormous computational power and may not even achieve the performance of
distillation." Also: R1's data pipeline is a **rejection-sampling filter** — keep only
correct final answers, machine-parseable, no repetition, second verification pass.

**Why it fits us.** We have `data/Teacher/teacher_distill.jsonl` (15,663 samples,
149 MB, model `deepseek-v4-flash-vision-exp`). The filter is mandatory, not optional: the
sample I read is explicitly uncertain ("assuming 1000x1000 pixels", "around y=300 to 550",
"Let me estimate the coordinates") — the teacher's boxes are acknowledged estimates.

**Verified state of the teacher data (checked 2026-09-14, do not assume more than this):**

| Artifact | Shape / content | Note |
|---|---|---|
| `teacher_distill.jsonl` | 15,663 records, `teacher_model = deepseek-v4-flash-vision-exp` | 15,507 mention bbox |
| `teacher_targets.npz` | 15,663 rows; `teacher_embedding` (15663, 256); `teacher_step_maps` (15663, 4, 16, 16) | split into `train_pool` 10,161 / `analysis_only` 5,502 |
| `num_teacher_boxes` | mean **5.5**, median 5, **5,268 / 15,663 images have zero boxes** | GT for comparison: mean 4.8, median 2 |
| `train_new_reasoning_features.npz` | (8129, 256) embeddings + `step_maps` (8129, 4, 16, 16) | **matches `train_new` exactly** — verified set-equality, 0 missing, 0 extra |

The 5,502 `analysis_only` rows are the test-set images; only the 10,161 `train_pool` rows
are usable for training. Zero-box rows (34%) carry no box supervision.

**Concrete action.** Parse each usable sample → match the teacher's boxes against GT at an
IoU threshold → keep only passes, with a second verification pass. Then distil as soft
targets. Note R1's warning (§Table 3) that the cold-start SFT stage *dropped* AIME
77.9 → 59.0 before RL recovered it: **expect an initial regression**, so keep
`dual_pathway_v2_thin` as the fallback and evaluate on `val_select` at every stage.
**Cost:** time-boxed data experiment. **Confidence:** medium-low — the box-verification
pass is what decides whether this is usable at all; **do that measurement first**, before
building any distillation machinery.

### S2. Auxiliary intermediate-state prediction (DeepSeek-Prover-V1.5, arXiv 2408.08152v1)

**Mechanism.** Insert the verifier's intermediate tactic state into the input as
`/- tactic state: … -/`, and compute SFT loss **only on tokens after the marker**; the
prefix is prompt-only. Same trick underlies DeepSeek-Coder's fill-in-the-middle at rate
0.5 (100% FIM maximised infilling but degraded plain completion).

**Why it fits us.** This is the only design found that gives the teacher reasoning text a
real supervision path rather than a decorative one: concatenate a reasoning prefix, compute
loss only on object predictions following the marker.
**Cost:** adds a head — must be droppable at inference (§3.6). **Confidence:** medium-low.
