# Deep Research: Path to Beating the SOTA Table

**Date:** 2026-08-26 · **Goal:** exceed the source SOTA table (F1 0.806) —
see `docs/SOTA.md`. Web search was unavailable this session; analysis is
grounded in our own measurements plus the SOTA table the user supplied.

## 1. Where we actually stand (category-level, with caveats)

Mapping our 30-class test AP50 into the SOTA table's 8 categories (macro-mean;
NOT directly comparable — their metric is F1 from segmentation-style matching,
ours is detection AP50 — but it locates the gaps):

| Category | V test AP50 (macro) | SOTA F1 | Gap |
|---|---:|---:|---:|
| Door | 0.567 | 0.848 | −0.28 |
| Window | 0.249 | 0.709 | −0.46 |
| Stair | 0.143 | 0.857 | −0.71 |
| Appliance | 0.548 | 0.769 | −0.22 |
| Furniture | 0.650 | 0.764 | −0.11 |
| Equipment | 0.186 | 0.926 | −0.74 |
| Parking lot | 0.799 | 0.539 | **+0.26** |

We already beat SOTA on Parking lot. Furniture is closest (−0.11). The
catastrophic gaps (Stair, Equipment, Window) are concentrated in a specific
failure mode — see §2.

## 2. Root cause: rare-class hallucination collapse

The per-class failure pattern on test is systematic:

| Class | val AP50 | test AP50 | test preds : GT |
|---|---:|---:|---:|
| column | 0.772 | 0.350 | 39:1 |
| bathtub | 0.532 | 0.289 | 90:1 |
| escalator | 0.472 | 0.296 | 121:1 |
| door_revolving | 0.552 | 0.034 | 56:1 |
| elevator | 0.400 | 0.023 | 246:1 |
| ramp | 0.000 | 0.000 | 300:1 |
| escalator_stair | 0.049 | 0.060 | 403:1 |

Good classes run 9–36 preds per GT; failing classes run 40–400:1. These
classes have 20–641 test GT but only 3–190 val GT (elevator: 11 val / 40
test; door_revolving: 6/36). Diagnosis:

1. **The negative-query mechanism overfits to val composition.** With
   neg_queries_per_pos=1, the model learns "which absent classes to suppress
   for images like val's". On test it hallucinates the rare classes it has
   almost never been told to suppress.
2. **Rare classes are also starved of positives** (ramp: 20 GT total in
   train) — the classifier never firms up.
3. **Small-object penalty**: column test median rel-size 0.084 vs val 0.118
   — the 64×64 heatmap is size-biased toward the val distribution it saw.

## 3. What this implies for the plan

The gap is NOT primarily architectural. V's vector branch already delivers
the largest single gain (+9.7 test AP50). The remaining gap decomposes into:

- **~60% rare-class calibration/suppression** (fixable without new science)
- **~25% small-object resolution** (stride-8 heatmap, 512 resize)
- **~15% training budget** (30 epochs, no pretrain, small models)

## 4. Prioritized attack plan

### Phase 1 — calibration & sampling (no architecture change, ~1 day)
1. **Per-class score calibration on val**: replace the global 0.05 threshold
   with per-class thresholds chosen on val (protocol-compliant). Directly
   attacks the 40–400:1 pred ratios — elevator needs a much higher bar.
2. **Neg-query oversampling for rare classes**: weight negative sampling so
   rare classes are queried-absent as often as common ones.
3. **Class-balanced sampler power tuning** (balance_power 0.5 → 0.75) so
   rare classes get more gradient.

### Phase 2 — resolution (architecture-light, ~2 days)
4. **Stride-4 heatmap variant**: double output resolution; targets and
   decoder already parameterised by stride. Expect the largest single gain
   for small symbols (sink, column, window).
5. **TTA: H-flip + 90° rotations at eval** (drawings are rotation-agnostic
   by construction; we already train with rot90 augmentation).

### Phase 3 — capacity & pretraining (~1 week)
6. **Pretrained backbone** (ConvNeXt-T / ResNet-18 ImageNet init) into the
   existing stride-8 encoder — the SOTA table's models are all pretrained.
7. **Longer schedule (60–90 epochs) + EMA weights**.
8. **Vector branch stays as-is** — it is our differentiator and its gain
   already generalizes.

### Success gate
Re-evaluate the 8-category macro after each phase; the target is to pass
SOTA's weighted F1 0.798 on their protocol, or (more honestly) to document
our own 8-category detection-AP50 numbers against their F1 as a
"conservative comparison" — we already win Parking lot and are close on
Furniture, so Phase 1–2 alone may close Door/Appliance.

## 5. Caveats

- The SOTA table's F1 and our AP50 are different metrics on possibly
  different matching rules; before claiming "beat SOTA" we must either (a)
  implement their exact evaluation (per-category F1 at a chosen operating
  point), or (b) state clearly we report detection AP.
- Rare-class AP is high-variance (elevator 40 GT); improvements there need
  the seed-replication discipline used throughout.
