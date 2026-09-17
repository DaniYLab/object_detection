# ArchCAD failure localisation and validation power — 2026-09-17

Five read-only probes, run after the budget / capacity / data experiments all landed in
the same validation band. None of them trains anything; they ask what the input features
and the validation set can support at all, before more GPU hours are spent.

## Probe 1 — the features do carry instance signal, but no single threshold recovers it

Pairwise decision "these two primitives belong to the same instance" as a function of
feature-space distance, over thing primitives on 128 val images. 1,098 instances,
148,971 same-instance pairs out of 1,041,162 pairs (14.31%).

| radius | precision | recall | F1 | permutation baseline F1 |
|---:|---:|---:|---:|---:|
| 0.020 | 0.6853 | 0.0749 | 0.1254 | 0.0369 |
| 0.050 | 0.7586 | 0.1789 | 0.2644 | 0.0686 |
| 0.100 | 0.8497 | 0.4421 | **0.5371** | 0.1556 |

The best single threshold reaches F1 0.537 against a baseline of 0.156, so the 12
dimensions are not information-free. But a *scale-blind* rule caps at recall 0.44, and
instances here range from a single primitive to dozens. This is the first quantitative
statement of the scale problem.

## Probe 2 — no query collapse

| model | kept queries / image | non-empty | mean pairwise IoU | pairs with IoU > 0.9 |
|---|---:|---:|---:|---:|
| trained | 30.7 | 15.5 | 0.0164 | 0.39% |
| random init | 25.7 | 15.6 | 0.1427 | 3.86% |

Trained queries are *more* diverse than random initialization. Mode collapse is ruled out
as the cause of the 71% localization miss, and the query budget is not binding either: the
decoder emits about 15 non-empty masks for roughly 11 GT segments per image.

## Probe 3 — the misses are about same-class competition, not size

Coverage of thing GT segments by any raw query mask at IoU > 0.5 (class-agnostic), 1,103
segments:

| primitives per instance | covered | | same-class instances in image | covered |
|---|---:|---|---|---:|
| 0–4 | 27.4% | | 1 (alone) | **42.7%** |
| 10–14 | 32.4% | | 2 | 39.6% |
| 25+ | **49.5%** | | 3 | 29.3% |
| | | | 4+ | **25.4%** |

Size does not explain the misses — small instances are not disproportionately missed and
the largest are covered best. Same-class competition does: coverage falls monotonically
from 42.7% to 25.4% as the class repeats, and **91% of all misses (712 of 779) sit in
images where the class appears more than once**.

The nearest *different* instance of the same class is much closer for missed instances:
median feature distance **0.285** versus **0.540** for covered ones, and 27.5% of missed
instances have such a neighbour within 0.1 against 9.1% of covered ones. Worst classes are
the densely repeated ones: `hole` 9.0%, `other_door` 9.5%, `sink` 9.5%, `elevator` 13.3%,
`single_door` 16.7%, `parking_space` 17.7%.

## Probe 4 — the gate: a scale-aware rule has headroom for about half the problem

For each instance an oracle may choose one radius. `d_in` is the largest minimum-spanning-
tree edge over the instance's own primitives (the scale that keeps it connected); `d_out`
is the smallest distance to a primitive of a different instance **of the same class**. An
instance is separable if `d_in < d_out`.

| | instances | share | model covered |
|---|---:|---:|---:|
| separable | 635 | 57.6% | 39.5% |
| not separable | 468 | 42.4% | 15.6% |

Cross-tabulated over all 1,103 instances:

- separable **and** covered — 22.8%
- separable **but** missed — **34.8%** → headroom a scale-aware method could reach
- not separable **and** covered — 6.6%
- not separable **and** missed — **35.8%** → beyond any proximity rule

`d_in / d_out`: median 0.725, and 47.4% of instances have ratio above 1. Only 10.6% of
instances have no different same-class instance in their image.

## Probe 5 — the 128-image validation subset is the binding limit on resolution

One fixed checkpoint (the frozen `primitive_256_s1337`) evaluated on a 512-image
validation set and on four disjoint 128-image folds of it:

| set | images | thing GT | pooled PQ | thing recall |
|---|---:|---:|---:|---:|
| fold 0 | 128 | 1,103 | 0.119358 | 0.096102 |
| fold 1 | 128 | 931 | 0.135035 | 0.121375 |
| fold 2 | 128 | 1,172 | 0.124999 | 0.091297 |
| fold 3 | 128 | 926 | 0.120739 | 0.082073 |
| all | 512 | 4,132 | 0.124883 | 0.097289 |

Spread across the four 128-image folds: pooled PQ range **0.015677** (sd 0.007087), thing
recall range **0.039301** (sd 0.016815). That is the sampling noise of a 128-image
estimate — essentially the same as the `0.016429` epoch-to-epoch figure used as the noise
floor all day.

Two consequences:

1. The floor used in every decision rule today is dominated by *which drawings happen to
   be in the validation subset*, not by training randomness. A 512-image subset should
   halve it.
2. "No resolvable effect" is the correct reading of several results, not "no effect".
   `scale_dim64_1000` versus the control was `−0.014315`, inside this floor. Thing recall
   is worse still: a 0.0393 range on values near 0.09 means recall-based gates, including
   the H1 overfit gate, were always going to be noisy on this subset.

## What this means for the next step

The gate splits the missed instances into two halves that need *different* fixes, and the
validation subset cannot currently resolve either.

1. **Scale and relational features** address the 34.8% that are separable but missed: the
   input needs to carry local packing scale (kNN distance statistics), size relative to
   that scale, and structural cues such as junction degree, which `src/drawing_graph/`
   already computes. Cheap: a dataset feature change plus a training run.
2. **Appearance is required** for the 35.8% that no proximity rule can separate. That is
   the raster branch of plan step 5, and this measurement is the first evidence that it is
   necessary rather than merely next in the plan.
3. **Enlarge the validation subset to 512 before the next comparison**, so that the
   effects being chased become measurable at all. The frozen reference value on 512
   images is pooled PQ `0.124883`, thing recall `0.097289`.

## Artifacts

- `outputs/panoptic/_runs_20260917/probe_feature_separability.py` (probes 1–2)
- `outputs/panoptic/_runs_20260917/probe_missed_instances.py` (probe 3)
- `outputs/panoptic/_runs_20260917/probe_scale_aware_gate.py` (probe 4)
- `outputs/panoptic/_runs_20260917/probe_val_power.py` (probe 5)
