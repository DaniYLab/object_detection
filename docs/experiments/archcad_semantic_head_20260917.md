# ArchCAD per-primitive semantic head — preregistered 2026-09-17

## Question

`archcad_semantics_bottleneck_20260917.md` decomposed the PQ gap against a train-fitted
geometry-only reference and found that **92.9% of it is per-primitive semantic error**,
not instance grouping. The query/mask model produces semantics as a byproduct of a mask
vote, `argmax_c (class_probability_c · mask).sum`, and its mask head was measured to emit
blobs that are not instance-aware, so that vote is close to arbitrary.

H9: predicting the per-primitive label directly, with no queries and no masks, beats that
indirect route at matched training regime.

## References, measured before this run on 512 validation images

Grouping constants proximity 0.020 / connectivity 0.010, fitted on 256 training images
(`src/data/archcad_geometric_grouping.py::fit_thresholds`, pairwise F1 0.9107 on train).

| pipeline | pooled PQ | thing PQ |
|---|---:|---:|
| query/mask model end to end (4,096 images x 28 epochs) | 0.148511 | 0.136764 |
| **its semantics + geometric grouping** (the arm to beat) | **0.169716** | 0.163329 |
| oracle semantics + geometric grouping | 0.807876 | 0.748993 |
| its per-primitive semantic accuracy | 31.56% | |

The same table for the 256-image / 20-epoch baseline: end to end 0.124883, its semantics
plus grouping 0.173977, accuracy 21.97%.

## Design

`ArchCADPrimitiveSemantic` — encoder over primitive tokens plus a 30-way linear head,
trained with length-weighted cross-entropy over non-ignore primitives. Instances come from
the geometric grouper, so the pipeline is: predicted semantics -> components -> frozen
evaluator.

Identical to the matched query/mask arm in every respect that is not the architecture:
4,096 training images (selection seed 1337), 28 epochs = 14,336 steps, batch 8,
length-grouped, AdamW at LR 0.0003 with weight decay 0.01 and gradient clip 1, FP32,
`--deterministic`, model seed 1337, 512 validation images (selection seed 1338), frozen
decoder and evaluator, geometry-only 12-dimensional input.

```powershell
python scripts/research/train_archcad_semantic.py --output-dir outputs/panoptic/semantic_4096 --train-images 4096 --val-images 512 --epochs 28 --eval-every 7 --batch-size 8 --eval-batch-size 8 --batch-order length-grouped --seed 1337 --selection-seed 1337 --dim 64 --heads 4 --encoder-layers 2 --deterministic
```

Endpoint: the final epoch. Intermediate evaluations are for curve shape only. One seed;
diagnostic, not a replicated finding. No held-out test split is read.

## Decision rule fixed before execution

Resolution on the 512-image validation set: **0.0078**.

Let `pq` be the final-epoch validation pooled PQ of the semantic head through the grouper,
and `accuracy` its per-primitive accuracy.

- **Confirmed** — `pq > 0.169716 + 0.0078 = 0.177516`. Semantics are the lever and the
  entanglement was the problem. Next: scale the semantic head (capacity, richer features,
  more data) and replace the hand-made grouper with a learned one, since the grouping is
  then the smaller term but still 7% of the gap.
- **Not better than the indirect route** — `pq <= 0.177516`. Then per-primitive
  classification from 12-dimensional geometry is itself the hard part, and the next
  question is what features semantics needs; the appearance branch (raster sampling)
  becomes the candidate again, now scoped to *semantics* rather than to grouping.
- `accuracy` is reported against the 31.56% reference but is not the gate, because the
  metric that matters is PQ and the two can disagree.

Whatever the outcome, the train-set metrics, the accuracy and the distance from the 0.808
oracle ceiling are reported, so a partial result still localises what remains.

## Results

Completed in 26 minutes. Final epoch, 4,096 training images, 28 epochs = 14,336 steps,
512 validation images, grouper constants fitted on train.

| Epoch | Val loss | Val pooled PQ | Val recall | Train pooled PQ |
|---:|---:|---:|---:|---:|
| 0 | 3.7237 | 0.00010 | 0.0002 | 0.0005 |
| 7 | 2.0412 | 0.20821 | 0.1985 | 0.2060 |
| 14 | 1.9903 | 0.20652 | 0.1994 | 0.2059 |
| 21 | 1.9427 | 0.20655 | 0.2026 | 0.2141 |
| 28 | 1.8979 | **0.20635** | 0.2047 | 0.2127 |

| reference | pooled PQ | per-primitive accuracy |
|---|---:|---:|
| semantic head (this run) | **0.206352** | **37.59%** |
| query/mask model end to end, same regime | 0.148511 | 31.56% |
| that model's semantics through the same grouper (the arm to beat, gate 0.177516) | 0.169716 | — |
| oracle semantics through the same grouper | 0.807876 | 100% |

**Verdict: CONFIRMED.** `0.206352 > 0.177516`, by `+0.028836` over the gate and `+0.036636`
over the arm to beat — 4.7 times the 0.0078 resolution of this validation set. Accuracy
also improves on the reference, 37.59% against 31.56%.

The head reaches this with 69,854 parameters against the query/mask model's 149,343, no
queries, no mask head and no Hungarian matching. Dropping the entanglement of instance
prediction and classification was worth more than any of the levers that were moved
earlier in the day.

Two features of the curve matter for what comes next:

- it plateaus by epoch 7 and stays flat to epoch 28, with train pooled PQ 0.2127 against
  validation 0.2064 — it is **not** overfitting at 4,096 images, so more data and more
  capacity are both still available;
- the panoptic components are SQ 0.825 and RQ 0.250, against 0.808/1.000 from the oracle
  ceiling. The remaining loss is recall of instances, driven by semantic error: perfect
  semantics would score 0.808 through the very same grouper.

## Where this leaves the program

Two independent wins now exist and have not been combined:

| direction | pooled PQ (512 val) |
|---|---:|
| architecture: semantic head, 4,096 images | **0.206352** |
| data: full 27,272-image split, query/mask model | **0.219300** |

At epoch 21 of the full-split run the two were level (0.20509 against 0.20635), and this
document was first written on that reading; at its endpoint the full-split query/mask model
pulls ahead, so the honest statement is that the head wins **at matched data** (0.20635
against 0.14851 at 4,096 images) and that the full split is worth more than the architecture
change on its own. Neither has been tested together.

The obvious next run is therefore the head on the full split. After that, the same question
the probes already framed: semantics sit at 37.6% accuracy against an oracle that scores
0.808 through the very same grouper, so the ceiling is far away and the remaining work is
semantic quality — capacity, features, and only then a learned grouper.

