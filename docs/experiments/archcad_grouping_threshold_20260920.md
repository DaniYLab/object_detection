# ArchCAD: the learned grouping rule was fitted on the wrong objective — preregistered 2026-09-20

## Question

Restating both decomposition bounds for the current pipeline made the grouping lever the largest
unclaimed one, and larger than it was when it was last attacked
(`archcad_learned_grouping_20260920.md`):

| configuration, 512 validation images | pooled PQ | thing PQ |
|---|---|---|
| the adopted pipeline (560-epoch semantics, per-class radii) | 0.343407 | 0.346588 |
| oracle grouping, same semantics | 0.582606 | 0.644438 |
| oracle semantics, adopted grouper | 0.841391 | 0.782920 |

The failed attempt trained an edge classifier that reached precision 0.5814, recall 0.8718 and
F1 0.6976, and still scored **lower** PQ than the rule it was pruning (0.291465 against
0.304004). Its threshold was chosen to maximise edge F1 on a training holdout. That is a
different objective from the metric: edge F1 rewards keeping edges, while the measured failure
is over-merging — 58.7% of ground-truth instances merged, 0.9% fragmented — so the operating
point the metric wants is far more aggressive than the one F1 picks.

This is not a hypothetical mismatch. Fitting the per-class radii on PQ rather than on a
recovered-minus-merged proxy is what produced the grouping gain adopted on 2026-09-20, and the
proxy was shown to rank the best variant nearly worst.

H18: the same edge classifier, with its decision threshold fitted on train PQ, beats the adopted
per-class rule by more than the resolution.

## Design

One factor changes from the failed attempt: the threshold. Held fixed: the edge features (centre
distance, minimum endpoint distance, endpoint coincidence, both lengths, both types, predicted
class), the candidate set (now the **adopted per-class** rule's own edges, so pruning can only
remove links the current rule makes, and the classifier can never lose for lack of candidates),
the classifier architecture and its training recipe (10 epochs, AdamW at 2e-3, positive-class
weighting), and the frozen evaluator.

- Checkpoint: `outputs/panoptic/semantic_full_b32_560ep/last.pt` (the 0.343485 pipeline).
- Fit set: 2,048 training drawings (seed 1337). Threshold grid: 0.30 to 0.95 in steps of 0.05.
- The threshold is chosen by **pooled PQ on the fit set**; validation (512 images, seed 1338) is
  scored once, at the chosen threshold.
- The adopted rule is scored on the same fit and validation sets in the same run, so the
  comparison is like-for-like within one process.

```powershell
python outputs/panoptic/_runs_20260917/h18_pq_threshold_grouping.py
```

## Decision rule fixed before execution

Resolution: 0.0078. Baseline: **0.343407** pooled PQ for the adopted rule on these 512
validation images, so the gate is **0.351207**.

- **The objective was the problem** — val PQ `> 0.351207`. Then a learned grouping rule is worth
  building properly (more fit data, a better classifier, per-class thresholds), and it is the
  route into the +0.239 grouping prize.
- **The information was the problem** — val PQ `<= 0.351207`. Then the edge classifier cannot
  see the boundary between adjacent same-class symbols from these features, and the conclusion
  in `archcad_learned_grouping_20260920.md` stands: the next grouping mechanism has to be
  additive (a rule that can link beyond the radius, or predict instance centres) rather than a
  pruning of the existing candidates.

The threshold grid is coarse (0.05). If the best threshold sits at a grid edge the fit is
reported as such and re-run finer before any conclusion is drawn, because a grid-edge optimum
means the search did not bracket the optimum.

## Results

Run completed. The adopted rule and the learned rule were scored in the same process on the same
sets: the adopted rule reproduces its recorded validation number exactly (train PQ 0.354347 on
the 2,048 fit drawings, val PQ **0.343407** on the 512 validation images), which is the
consistency check that the reconstruction matches the pipeline.

Candidate set: 6,733,511 edges over 2,048 training drawings, 31.61% of them positive (both
primitives in the same ground-truth instance), and 2,037,324 over the 512 validation drawings.
The classifier converged normally (loss 1.04212 → 0.74273 over ten epochs).

The threshold sweep on **train PQ**, which is the whole point of the experiment:

| threshold | 0.30 | 0.40 | 0.45 | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train PQ | 0.345525 | 0.334840 | 0.329840 | 0.323130 | 0.315687 | 0.306644 | 0.296795 | 0.284129 | 0.255748 | 0.217882 | 0.162327 | 0.064824 | 0.045099 |

| configuration | train PQ | val PQ |
|---|---|---|
| adopted per-class rule | 0.354347 | 0.343407 |
| learned pruning at the PQ-fitted threshold 0.30 | 0.345525 | 0.335758 |

The rule does **not** clear the gate (0.351207). It scores 0.335758, below the rule it prunes.

Two things must be said about the threshold before the verdict is read.

1. **The optimum is at the grid edge**, so the search never bracketed it. That is recorded as a
   deviation, as the preregistration requires. But extending the grid downward cannot change the
   verdict: the sweep is monotone — every increase in pruning lowers PQ — and a threshold below
   0.30 converges to keeping every candidate, which *is* the adopted rule. In other words the
   best this family can do is reproduce the baseline, and it pays for that with the least
   pruning in the grid, where it is already 0.0088 below it on train.
2. **The objective was not the problem.** This was the hypothesis, and it is refuted: with the
   threshold fitted on PQ, the operating point moved to the least-pruning end of the grid and
   still lost. At that point the classifier keeps 48.57% precision and 97.44% recall of the
   candidate edges — it is removing edges that the metric needs.

So the preregistered second branch applies: **the information was the problem**, not the
objective. A classifier over pairwise features — centre distance, minimum endpoint distance,
endpoint coincidence, both lengths, both types and the predicted class — cannot tell which
candidate edges are inside one symbol and which cross between two adjacent same-class symbols,
because two adjacent symbols of the same class share exactly the geometry those features
describe (a shared boundary, coincident endpoints, comparable length).

This closes the pruning direction for good: the per-class radii were already fitted on PQ, so
any learned reweighting of the same candidate set with these features loses. The next grouping
mechanism has to be **additive** — it must be able to link primitives beyond the radius, or
predict where each instance lies, rather than decide which of the current rule's links to keep.

## Artifacts

- `outputs/panoptic/_runs_20260917/h18_pq_threshold_grouping.py`
- `outputs/panoptic/h18_pq_threshold_grouping.json`
- `outputs/panoptic/_runs_20260917/check_h18.py` (vectorised candidate builder against a Python
  reference: 1,066 edges, 0 missing, 0 extra)
