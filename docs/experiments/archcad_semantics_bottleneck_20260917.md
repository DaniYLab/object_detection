# ArchCAD: the bottleneck is per-primitive semantics, not instance grouping — 2026-09-17

Five more read-only probes, run while the full-split arm trains. They overturn the
previous reading of the failure and identify where PQ is actually lost.

## Probe 6 — probe 4's gate was not a singleton artifact

Only 65 of 1,103 val thing instances are single-primitive, so the earlier "57.6%
separable" figure was not dominated by trivially separable singletons. Among the 1,038
multi-primitive instances — the ones that need real grouping — 570 are separable by an
oracle radius and 59.8% of those are missed, versus 84.4% missed for the 468 that are not.
The separable/inseparable distinction is real.

## Probe 7 — the missed masks are over-merges, not fragments

For every missed instance, decomposed against its best query mask:

| | median | share of misses |
|---|---:|---:|
| GT recall of the mask (how much of the instance it contains) | **0.849** | 27.3% below 0.5 |
| precision of the mask (how much of it belongs to the instance) | **0.252** | 84.3% below 0.5 |

Of 779 missed instances, 66.0% are "mask too large" (recall high, precision low), 9.0% are
fragments, 18.4% are both low. For singletons the mask contains the primitive with median
recall **1.000** and precision **0.134** — it always finds it, and buries it in ~6x too much
weight.

The model is not failing to find the right primitives. Its masks are not instance-shaped.

## Probe 8 — no decision rule can repair those masks

Comparing the mask probability of primitives inside the instance against the ones the same
query over-includes:

| | median probability |
|---|---:|
| primitives inside the instance | 0.704 |
| primitives wrongly included | **0.748** |

The wrongly included primitives are scored *higher*. At every threshold the wrong mass is
kept at a higher rate than the right mass (0.6: 81.7% vs 64.2%; 0.8: 38.7% vs 35.5%). Only
31.5% of queries even rank inside above outside on average. Across missed instances the
best query includes 38,210 extra primitives against 6,272 correct ones, and 72.1% of the
mask weight is excess.

So this is not a binarisation or threshold problem, and tuning the decoder cannot fix it:
the mask head's scores carry almost no instance information.

## Probe 9 — geometry separates instances far better than the model does

Pairwise "same instance" as connected components over thing primitives:

| rule | precision | recall | F1 |
|---|---:|---:|---:|
| centre proximity < 0.1 | 0.6694 | 0.9555 | 0.7471 |
| endpoint connectivity, tolerance 0.01 | 0.8598 | 0.7503 | **0.7688** |
| both | 0.6385 | 0.9965 | 0.7275 |

Connectivity is more precise, proximity more complete, and the union reaches 99.65%
recall while over-merging. Errors are false merges, which is the same failure mode as the
model's, but at a fraction of the severity.

## Probe 10 — the geometry-only ceiling, and where PQ actually goes

Oracle per-primitive semantics plus a learning-free proximity + endpoint-connectivity
grouping, evaluated with the frozen evaluator on the same 128 validation images:

| configuration | pooled PQ | thing PQ | thing SQ | thing RQ |
|---|---:|---:|---:|---:|
| model semantics + model instances | 0.119358 | 0.093355 | 0.801884 | 0.116420 |
| model semantics + geometric grouping | **0.164508** | 0.158683 | 0.879214 | 0.180483 |
| oracle semantics + geometric grouping | **0.830832** | 0.771045 | 0.969945 | 0.794937 |

And the model's **per-primitive semantic accuracy is 14.8%** (4,205 of 28,453 primitives).

Read as a decomposition: replacing the learned mask head with the geometric grouper gains
+38% PQ without any training; replacing the model's semantics with the oracle gains +405%.
**Roughly 85% of the lost PQ is per-primitive semantic error, not instance grouping.**

### Caveat on the reference, and its resolution

The grouper's two thresholds (proximity 0.1, connectivity 0.01) were chosen by sweeping on
this validation subset, so the table above is optimistic. Re-fitting them on 256 **training**
images (`src/data/archcad_geometric_grouping.py::fit_thresholds`) selects proximity 0.020
and connectivity 0.010, which reaches pairwise F1 0.9107 on train, and gives a lower and
attributable validation reference:

| configuration, train-fitted constants | pooled PQ | thing PQ |
|---|---:|---:|
| model semantics + model instances | 0.119358 | 0.093355 |
| model semantics + geometric grouping | 0.164135 | 0.159962 |
| oracle semantics + geometric grouping | **0.751385** | 0.683065 |

Proximity alone at 0.1, with connectivity disabled, gives 0.749670 — so the reference sits
near 0.75 whether or not connectivity is used, and the earlier 0.831 was 0.080 of tuning
optimism.

Decomposition against the honest reference: the gap from ceiling to model is 0.632026, of
which **0.587250 (92.9%) is semantic error** and 0.044776 (7.1%) is the mask head. The
conclusion is unchanged, which is the point of having fitted the constants on train.

## What this says about the architecture

The current model derives per-primitive semantics as a byproduct of the query-mask vote:
`semantic = argmax_c (class_probability_c · mask).sum over queries`. When the mask head
produces blobs that are not instance-aware — measured above — that vote is close to
arbitrary, which is where the 14.8% comes from. Meanwhile the underlying task, "which
symbol is this primitive", is a plain per-primitive classification problem, and it is known
to be learnable: the 8-image overfit test reached semantic macro F1 0.93, and the
irreducible ambiguity floor measured on the input is 0.66% of training primitives.

The architecture therefore entangles two problems with very different difficulty, and the
harder one (set prediction over hundreds of small repetitive symbols) is being asked to
carry the easier one (per-primitive classification). A DETR-style set predictor is designed
for images with few, large, visually distinct objects; an ArchCAD drawing has hundreds of
small symbols with massive intra-class repetition.

## Artifacts

- `outputs/panoptic/_runs_20260917/probe_miss_decomposition.py` (probes 6–7)
- `outputs/panoptic/_runs_20260917/probe_mask_overinclusion.py` (probe 8)
- `outputs/panoptic/_runs_20260917/probe_connectivity.py` (probe 9)
- `outputs/panoptic/_runs_20260917/probe_geometry_ceiling.py` (probe 10)
- `outputs/panoptic/_runs_20260917/probe_semantics_vs_grouping.py` (probe 10, hybrid cell)
