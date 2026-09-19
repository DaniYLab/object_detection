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

Pending at preregistration. Record outcomes and any deviations after completion.
