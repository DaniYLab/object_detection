# Deep-Dive: Why the Vector Branch Helps Where It Helps

**Date:** 2026-08-25 · **Data:** val reports seed 1337 (B vs V) + SVG structural
analysis over train metadata. Companion to `experiment_log.md`.

## The headline distribution

V improves 20/25 classes with GT>20 (mean gain **+0.102 AP50**), degrades 3,
neutral 2. Predictions drop from 227,770 (B) to 135,986 (V) — the vector branch
massively suppresses false positives (hallucinated boxes), which is where most
of the +10 AP50 comes from.

## Per-class anatomy

| Class | B→V AP50 | Δ | Strokes/inst (median) | Stroke length (mean) | Reading |
|---|---|---:|---:|---:|---|
| door_revolving | 0.027→0.552 | +0.525 | 21 | — | clean circular symbol; arc params are its identity |
| toilet | 0.304→0.770 | +0.467 | 39 | 0.033 | dense long strokes; arc-heavy |
| column | 0.458→0.772 | +0.314 | 32 | 0.023 | arcs define the symbol |
| table | 0.575→0.801 | +0.225 | 27 | 0.009 | many short strokes = strong combinatorial signature |
| counter / room_label / parking / oven / window / plant / sink... | | +0.13–0.22 | | | |
| bed | 0.478→0.417 | −0.060 | 8 | 0.036 | few long strokes; weak per-token signature |
| sink | 0.376→0.253 | −0.122 | 8 | 0.016 | few short strokes + fine detail lost to subsampling |
| floor_plan_area | 0.409→0.265 | −0.144 | 17 | 0.013 | sparse scattered boundary strokes |

**Emerging rule:** classes whose instances are made of *many* strokes, or whose
identity lives in *arc parameters*, gain the most. Classes made of *few,
short* strokes lose — their vector signature is weak, and the random
subsampling (4.5% of drawings exceed n_max=1024) plus cross-attention dilution
hurts more than the geometry helps.

The largest gain (door_revolving, toilet, column) confirms the mechanism: these
are **circle/arc-defined symbols**. Raster at 512px destroys arc curvature
(64×64 heatmap cells); the vector token carries center/radius/angles exactly.
This is precisely the information the raster pipeline discards.

## False-positive suppression

V cuts prediction count by 40% at the same decoder threshold. The vector branch
gives the model a "does any geometry exist here" prior — on empty regions the
image branch alone hallucinates; with strokes as K/V the cross-attention has
nothing to attend to and the heatmap stays cold.

## Truncation reality check

Only 4.5% of drawings exceed n_max=1024 (median is 133 strokes), so
subsampling truncation is **not** the main degradation driver for sink/bed —
their weakness is intrinsic (few, short, low-information tokens), not
truncation. Heaviest drawings (~4,400 strokes) are hatch/dot patterns
dominated by walls and toilets; their loss is tolerable.

## Actionable follow-ups (ranked)

1. **Per-stroke importance weighting at truncation** — replace uniform random
   subsampling with length-weighted sampling (keep long strokes; they carry
   identity). Cheap, may recover sink/bed partially.
2. **Instance-level vector query augmentation** — the current fusion is
   class-agnostic (whole drawing). A second fusion stage conditioned on the
   queried class embedding (FiLM on vector tokens before cross-attention)
   would let "toilet" queries attend to arc-heavy strokes specifically.
3. **Arc-privilege head** — the gains concentrate in arc-defined symbols; a
   small auxiliary loss predicting arc presence per cell could sharpen this
   further. Likely the cheapest per-AP-point gain.
4. **Larger n_max with linear attention** — 4.5% truncation is small but the
   p99 tail (10K+ strokes) is where floor_plan_area lives; linear/ Performer
   attention over strokes would raise the cap to 4K cheaply.
5. **Prediction calibration** — 40% fewer predictions at fixed threshold 0.05
   suggests V's score distribution shifted; re-tuning the threshold on val
   (protocol-compliant, before test) may add AP for free.

## Follow-up results (2026-08-25)

**#5 Threshold sweep — resolved, no change needed.** Swept 0.01–0.50 over
saved val predictions (`scripts/dev/sweep_threshold.py`,
`outputs/threshold_sweep_dual_pathway_seed1337.json`). AP50:95 is flat at
0.2737 for every threshold ≤ 0.05 and decreases monotonically above (0.10 →
0.2735, 0.50 → 0.2411). The decoder's 0.05 is already optimal: 66% of raw
scores sit below 0.10 yet removing them costs only 0.0002 AP — the model is
well calibrated and low-score predictions are inert, not harmful.

**#1 Length-weighted truncation — implemented.** `sample_strokes` now samples
strokes with probability ∝ endpoint diagonal (radius as proxy for
near-full-circle arcs), floored at 2% so short strokes stay in the lottery.
Affects only the 4.5% of drawings exceeding n_max=1024, so it is a targeted
fix for the heavy-tail drawings where floor_plan_area degrades. Covered by
`tests/test_dual_pathway.py` (34 tests). Requires a retrain to take effect —
queued behind the current experiment plan.

**Remaining:** #2 class-conditioned vector fusion (architecture change,
train + replicate) and #3 arc-privilege head are the next candidates if the
vector line continues.

## V2 results (2026-08-26) — #2 + #1 combined

`dual_pathway_v2` (class-conditioned vector FiLM + length-weighted
truncation) seed 1337 full-val: **AP50=0.5160, AP50:95=0.2742** vs V1
(0.5032/0.2737) = **+1.28 AP50**.

Per-class against V1 (seed 1337):

- The two worst V1 regressions recovered: `floor_plan_area` 0.265→0.348
  (+0.083, length-weighted truncation keeping its long boundary strokes) and
  `elevator` 0.400→0.549 (+0.149). `door_revolving` improved further
  0.552→0.702 (+0.150).
- Six mid-tier classes regressed slightly (stair −0.066, oven −0.054,
  refrigerator −0.048, table −0.034, shower −0.027, plant −0.022) — the
  class-conditioned FiLM helps classes with distinctive geometry but adds a
  mild optimization cost elsewhere.
- Net effect positive (+1.28 AP50) but modest; V2−V1 mixes the fusion change
  and the truncation change, and is single-seed. If the vector line
  continues, replicating V2 on seed 2024 and/or ablating the two changes
  separately would separate the contributions.

## Caveats

- All analysis is seed-1337 val; per-class deltas vary by seed (door_revolving
  has only 6 GT instances — its +0.525 is fragile).
- stroke-length stats use endpoint distance only; arc length is underestimated.
- No literature comparison was possible (web search unavailable in this
  session); claims are grounded in our own measurements.
