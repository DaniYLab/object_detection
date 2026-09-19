# ArchCAD: learned grouping does not capture the oracle-grouping prize — 2026-09-20

The decomposition left one large lever unclaimed: with today's semantic quality, replacing the
fixed geometric grouping with a perfect one would move pooled PQ from 0.305800 to 0.480167.
This is the attempt to capture part of it, and the measurements it produced.

## What was tried

A small classifier over **candidate edge** features, in the style the successful probes used:
features are centre distance, minimum endpoint distance, endpoint coincidence, primitive
lengths and types, and the predicted class; the label is whether two primitives share a
ground-truth instance. Edges are built from the frozen semantic head's own predictions so
training and inference see the same distribution, and the decision threshold is fitted on a
training holdout rather than on validation.

Two candidate sets were tried. The first used the k nearest same-class neighbours; the second
used the fixed rule's own candidates — same-class proximity within 0.1 plus coincident
endpoints — so the classifier could only prune and could not lose for lack of candidates.

| configuration | pooled PQ | thing PQ | thing RQ |
|---|---:|---:|---:|
| semantic head + fixed geometric grouping | **0.304004** | 0.303521 | 0.345526 |
| semantic head + learned pruning, kNN candidates | 0.293159 | 0.289416 | 0.327603 |
| semantic head + learned pruning, fixed rule's candidates | 0.291465 | 0.287381 | 0.325884 |
| semantic head + oracle grouping | 0.480167 | 0.539573 | 0.539573 |

The learned rule improved edge quality — precision 0.5814 at recall 0.8718, F1 0.6976, against
a candidate set that is only 33.6% positive — and still scored **lower** PQ. **Edge F1 is not
what the metric rewards**, and pruning by a learned score does not capture the prize.

## What the fixed rule gets wrong

Per ground-truth thing instance on the validation set, with the trained semantics
(`probe_grouping_errors.py`, which does load the checkpoint):

| outcome | instances | share |
|---|---:|---:|
| recovered | 1,597 | 38.6% |
| **merged** — a component spans more than one instance | **2,426** | **58.7%** |
| fragmented — primitives land in several components | 36 | 0.9% |
| missed whole | 73 | 1.8% |

The failure is **over-merging, not fragmentation**. A rule that only prunes edges is aimed at
the right error, and it made things worse anyway.

The merges are mostly pairs of adjacent same-class symbols: 699 merged components swallow
1,960 instances, 437 of those components hold exactly two instances and 157 hold three, and
the classes are the repeated-symbol ones — `concrete_column` 215 components, `single_door` 78,
`double_door` 76, `parking_space` 61, `staircase` 53, `pile` 52, `hole` 40.

## Two intermediate findings retracted

Two probes in this stretch did not call `load_state_dict` and therefore evaluated a randomly
initialised model. Their conclusions were wrong and are withdrawn:

- "the model predicts `sink` for 80% of primitives" — wrong; with the trained weights the top
  over-predictions are `bed` 4.19x, `chair` 2.91x, `wall` 2.88x, and `wall` takes 38.5% of all
  predictions, which is the wall-as-attractor pattern already measured;
- "95% of merges are `elevator` components swallowing seven instances each" — wrong; the
  corrected anatomy is the repeated-symbol pattern above.

Both were caught because the numbers contradicted the checkpoint's reported 56.6% accuracy,
and that contradiction was checked rather than explained away. The corrected probes are the
ones cited here; the artifacts are corrected in place.

## Where this leaves the grouping lever

The prize is real — oracle grouping at today's semantics is worth +0.174 — but it has now
survived four attempts to approach it:

| approach | result |
|---|---|
| the fixed geometric rule | 0.304004 |
| a learned rule that prunes the fixed rule's candidates | 0.291465, worse |
| scale-aware input features | no resolvable effect (H7) |
| appearance / raster | no channel exists in the data (probe 11) |

What the misses share is that the boundary between two adjacent same-class symbols is not
determined by the drawing's geometry. That is the same conclusion the very first probes
reached from the other side: 42.4% of instances are not separable by any single proximity
radius, and the two things that could break the tie — appearance and scale — are both closed.

The untested options are narrower than they look:

1. **Additive grouping.** Every design here could only remove candidate edges. A rule that
   links primitives *beyond* the proximity radius — long-range instance connectivity — has
   not been tested, and the merge anatomy gives it no obvious target either.
2. **Joint training.** All of this work is post hoc: the semantics are frozen and the grouping
   is fitted on top. A model trained with grouping in the objective could in principle find
   boundaries that a post-hoc rule cannot, but the capacity sweep says more parameters do not
   help and the metric is not differentiable, so the mechanism is unclear.

## Artifacts

- `outputs/panoptic/_runs_20260917/probe_learned_grouping.py`
- `outputs/panoptic/_runs_20260917/probe_grouping_errors.py`
- `outputs/panoptic/_runs_20260917/probe_merge_anatomy.py` (corrected)
- `outputs/panoptic/_runs_20260917/probe_class_overprediction.py` (corrected)
- `outputs/panoptic/_runs_20260917/verify_checkpoint_predictions.py` (the check that caught it)
