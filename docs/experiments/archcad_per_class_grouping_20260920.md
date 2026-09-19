# ArchCAD: per-class grouping radii, the first win on the grouping lever — 2026-09-20

`archcad_learned_grouping_20260920.md` closed with the oracle-grouping prize (+0.174 at
today's semantic quality) having survived four approaches: the fixed rule, a learned pruning
classifier, scale-aware input features and appearance. This records the fifth attempt, which
worked, and the two refuted hypotheses it passed through.

## Refuted: dropping proximity, and my own selection criterion

**Connectivity only.** The merge anatomy showed the rule over-merges adjacent same-class
symbols, and two adjacent columns share no endpoint, so proximity looked like the culprit.
Scored on validation, connectivity-only grouping is far worse, because most instances need
proximity to be connected at all:

| variant | pooled PQ on validation |
|---|---:|
| proximity 0.1 + connectivity 0.01 (default) | **0.305800** |
| connectivity 0.01 only | 0.221738 |
| connectivity 0.005 only | 0.199347 |
| connectivity 0.02 only | 0.245759 |

**A recovered-minus-merged count is not a proxy for PQ.** Variants were first ranked on the
training split by "instances recovered minus instances merged", which the anatomy had used.
That criterion ranked `proximity 0.02 + connectivity 0.01` (criterion `+0.071`) above
`proximity 0.05 + connectivity 0.01` (criterion `−0.036`); scored on validation the first
gives 0.282876 and the second **0.307945**. A rule that recovers more instances can still
produce many extra components that match nothing, and every one of those is a false positive
the criterion never counted. The criterion is withdrawn; PQ itself is cheap enough to use
directly.

## The hypothesis that held: one radius cannot serve every class

Probe 1 measured this from the input side — a single distance threshold caps at recall 0.44
because instances range from one primitive to dozens — and it is consistent with the merge
anatomy, where `concrete_column` merges in 215 components while `hole` and `staircase` need to
be joined at all.

So the proximity radius was fitted **per class** by coordinate descent on 1,024 training
images, using the frozen evaluator's own PQ as the objective, then scored on validation with
no further adjustment.

| | pooled PQ | thing PQ |
|---|---:|---:|
| uniform proximity 0.1, the current default | 0.305800 | 0.300241 |
| **per-class radii fitted on train** | **0.327670** | **0.329621** |
| difference | **+0.021871** | +0.029380 |

`+0.021871` is 2.8 times the 0.0078 resolution of the validation set. The fit-set PQ moved
from 0.308225 to 0.334994, so the validation gain is 82% of the fit gain — some optimism from
fitting eight constants on 1,024 images, but the effect survives it.

The fitted radii are interpretable and match the diagnosis: `concrete_column` 0.05 and `pile`
0.02 tighten, which is what separates adjacent columns and piles and is where the merges were;
`single_door` 0.05 tightens while `staircase`, `hole` and `parking_space` widen to 0.20;
`wall` and `double_door` stay at 0.10. `pile` alone contributed +0.0198 on the fit set.

```text
fitted per-class proximity radii, connectivity 0.01 throughout
  single_door       0.050      hole             0.200
  double_door       0.100      wall             0.100
  staircase         0.200      concrete_column  0.050
  parking_space     0.200      pile             0.020
```

## Why this costs nothing to adopt

The semantic head is trained with per-primitive cross-entropy, which never touches the
grouping: instances are built only at decode time. So better grouping constants improve the
pipeline's output without retraining, and they apply to every checkpoint already on disk.

## Status of the lever

| approach | pooled PQ |
|---|---:|
| fixed uniform rule | 0.305800 |
| learned pruning of candidate edges | 0.291465 |
| connectivity only | 0.221738 |
| **per-class radii fitted on train** | **0.327670** |
| oracle grouping at the same semantics | 0.480167 |

Roughly an eighth of the prize has been taken, and the direction that took it — per-class
structure in the grouping rule — is the one the input-side probes pointed at all along. A
two-pass fit over every class present is running; per-class connectivity tolerances are the
natural extension after it.

## Artifacts

- `outputs/panoptic/_runs_20260917/probe_grouping_variants.py` (the refutation)
- `outputs/panoptic/_runs_20260917/fit_per_class_radii.py` (the eight-class fit)
- `outputs/panoptic/_runs_20260917/fit_per_class_radii_full.py` (two passes, all classes)
