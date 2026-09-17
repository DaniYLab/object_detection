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

Pending at preregistration. Record outcomes and any deviations after completion.
