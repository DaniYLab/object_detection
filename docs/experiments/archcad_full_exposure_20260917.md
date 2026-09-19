# ArchCAD: full-split exposure, 56 epochs — preregistered 2026-09-17

## Question

`archcad_exposure_20260917.md` showed exposure is exhausted at 4,096 images: quadrupling the
passes moved validation pooled PQ by −0.002272 and widened the train−validation gap from
+0.006397 to +0.028988. The same experiment found one asymmetry that has not been tested:
the full split has not saturated.

| run | images | steps | val pooled PQ | train − val |
|---|---:|---:|---:|---:|
| `semantic_4096` | 4,096 | 14,336 | 0.206352 | +0.006397 |
| `semantic_4096_112ep` | 4,096 | 57,344 | 0.204081 | +0.028988 |
| `semantic_full` | 27,272 | 95,452 | **0.246187** | **+0.007280** |

`semantic_full` has nearly seven times the data of the arm that saturated and a gap four
times smaller, and its validation curve was still climbing at its last evaluation
(0.21324 → 0.21876 → 0.22984 → 0.24619 at epochs 7/14/21/28).

H13: the full split still has exposure headroom, so doubling its passes improves validation.

## Design

| arm | architecture | images | epochs | steps | val images |
|---|---|---:|---:|---:|---:|
| reference | transformer, dim 64, 2 layers | 27,272 | 28 | 95,452 | 512 |
| **H13** | identical | 27,272 | **56** | **190,904** | 512 |

Everything else unchanged: batch 8, length-grouped, AdamW at LR 0.0003 with weight decay
0.01 and gradient clip 1, FP32, `--deterministic`, model seed 1337, selection seed 1337,
validation selection seed 1338, geometry-only 12-dimensional input, grouper constants
proximity 0.020 / connectivity 0.010 fitted on train, frozen evaluator, `--eval-every 7`.

Cost: about six hours, which is why the 4,096-image arm was tested first and why this is a
doubling rather than the quadrupling that arm used.

```powershell
python scripts/research/train_archcad_semantic.py --output-dir outputs/panoptic/semantic_full_56ep --train-images -1 --val-images 512 --epochs 56 --eval-every 7 --batch-size 8 --eval-batch-size 8 --batch-order length-grouped --seed 1337 --selection-seed 1337 --dim 64 --heads 4 --encoder-layers 2 --deterministic
```

Endpoint: the final epoch. One seed; diagnostic, not a replicated finding. No test split read.

## Decision rule fixed before execution

Resolution: 0.0078 in pooled PQ. Reference: `semantic_full` at 0.246187.

- **Exposure still helps at the full split** — final validation pooled PQ `> 0.253987`.
  Then the pipeline is data-scale limited rather than structurally limited, and the next
  work is scale: more passes, and the batch-size question the batching measurements already
  opened, before anything is redesigned.
- **Saturated here too** — final `<= 0.253987` while the train−validation gap stays within a
  few points of the reference's +0.007280. Then every axis of this pipeline has a measured
  limit at every scale tested, the objective rather than the scale is what remains, and the
  next build is a model trained jointly over per-primitive embeddings and their grouping,
  scored end to end through the grouper — the shape the task actually has.
- **Overfitting** — final `<= 0.253987` with the gap widening well beyond the reference's.
  Then the full split cannot absorb more passes either, and the same conclusion follows.

Whatever happens, the run reports validation and train curves at eight points, so the shape
of the saturation is recorded and not only its endpoint.

## Results

Completed in 59 minutes rather than six hours. |Epoch | Val loss | Val pooled PQ | Val accuracy | Train pooled PQ |
|---:|---:|---:|---:|---:|
| 7 | — | 0.21324 | — | 0.1953 |
| 28 (reference) | 1.8979 | 0.24619 | 0.4844 | 0.2535 |
| 56 | 1.6024 | **0.273106** | **0.525435** | 0.279264 |

`0.273106 > 0.253987`, and by `+0.019119` over the gate.

**Verdict: exposure still helps at the full split.** The train−validation gap stayed put —
`+0.007280` at 28 epochs against `+0.006157` at 56 — so this is not overfitting; the run is
still converting passes into validation quality. Per the preregistered rule the pipeline is
**data-scale limited rather than structurally limited**, and the next work is scale before
anything is redesigned.

This is the new best result: **pooled PQ 0.273106**, thing PQ 0.267180, per-primitive
accuracy 52.54%.

## The value of better grouping, at the new best

One cell of the semantic-versus-grouping decomposition was still unmeasured, and it decides
how much a learned grouping is worth. Filled in here for this checkpoint, using the same
grouping constants the runs used, through the frozen evaluator:

| configuration | pooled PQ | thing PQ | thing RQ |
|---|---:|---:|---:|
| model semantics + geometric grouping (this run) | 0.273106 | 0.267180 | 0.312223 |
| **model semantics + oracle grouping** | **0.480167** | 0.539573 | 0.539573 |
| oracle semantics + geometric grouping | 0.807876 | 0.748993 | 0.794937 |
| oracle semantics + oracle grouping | 1.0 | 1.0 | 1.0 |

The oracle-grouping cell is a construction: one predicted instance per ground-truth segment,
carrying that segment's mask and the model's majority predicted class over its primitives.

**Perfect grouping at today's semantic quality scores 0.480167, which is +0.207062 over what
the pipeline achieves.** That is 39% of the distance from the head to the oracle-semantics
ceiling of 0.807876. Grouping is therefore a large lever, comparable to semantic quality
rather than a rounding error, which is the strongest argument yet for training the assignment
jointly instead of using a fixed geometric rule.

## Deviation: the runs did not use the preregistered grouping constants

The preregistration specified proximity 0.020 and connectivity 0.010, fitted on the training
split. Every semantic-head run, including this one, actually used **proximity 0.1 and
connectivity 0.01** — the trainer's argparse defaults — because the launch commands did not
pass `--proximity`. Measured on the `semantic_full` checkpoint, the same validation set:

| constants | pooled PQ |
|---|---:|
| 0.1 / 0.01, what the runs used | 0.246187 |
| 0.020 / 0.010, what was preregistered | **0.231259** |

So the reported levels are about `0.0149` optimistic relative to the preregistered
configuration, and that difference is resolvable against the 0.0078 floor. Two consequences:

- **Comparisons between arms are unaffected**: all five semantic-head runs used 0.1, so the
  H9, H12 and H13 verdicts stand as measured.
- **The absolute levels are not the preregistered ones.** Where this document and
  `archcad_semantic_head_20260917.md` quote 0.246187 for the 28-epoch full-split run, the
  preregistered-constant value is 0.231259. The 0.1 value is not wrong, but it is a
  validation-tuned constant from the probe sweeps rather than the train-fitted one, and the
  config's `grouping.fitted_on` note describes provenance the value does not have.

The default is left unchanged, because changing it now would silently break comparability
with the five runs already on record; the provenance note in the trainer is corrected instead
and the choice must be stated explicitly in future preregistrations.

## Artifacts

- `outputs/panoptic/semantic_full_56ep`
- Decomposition: `outputs/panoptic/_runs_20260917/probe_grouping_value.py`
- Constants audit: `outputs/panoptic/_runs_20260917/check_grouping_constants.py`

