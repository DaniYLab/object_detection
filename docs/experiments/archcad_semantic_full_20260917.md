# ArchCAD semantic head on the full training split — preregistered 2026-09-17

## Question

Two independent improvements landed today and neither has been tested with the other:

| direction | pooled PQ (512 val) | note |
|---|---:|---|
| data: full 27,272-image split, query/mask model | **0.219300** | largest gain of the day |
| architecture: per-primitive semantic head, 4,096 images | **0.206352** | 69,854 params, no queries, no masks, no Hungarian |

The head wins at matched data (0.20635 against 0.14851 at 4,096 images), the full split wins
against the 4,096-image subset for the query/mask model (+0.070790), and the full-split
query/mask model currently leads outright.

H10: the two gains are additive, so the head on the full split exceeds both.

## Design

`ArchCADPrimitiveSemantic` (encoder over primitive tokens plus a 30-way head, length-weighted
cross-entropy) on every eligible training image, with instances from the geometric grouper.

| arm | images | architecture | val images | status |
|---|---:|---|---:|---|
| — | 4,096 | semantic head | 512 | 0.206352, done |
| — | 27,272 | query/mask | 512 | 0.219300, done |
| **H10** | all eligible (27,272 expected) | semantic head | 512 | this run |

Everything else is held at the values used by both earlier arms: 28 epochs, batch 8,
length-grouped, AdamW at LR 0.0003 with weight decay 0.01 and gradient clip 1, FP32,
`--deterministic`, model seed 1337, selection seed 1337, validation selection seed 1338,
grouper constants proximity 0.020 / connectivity 0.010 fitted on 256 training images,
geometry-only 12-dimensional input, dim 64 with 4 heads and 2 encoder layers.

Steps: `ceil(27,272 / 8) × 28 = 95,480`, comparable to the full-split query/mask arm's 95,452.

```powershell
python scripts/research/train_archcad_semantic.py --output-dir outputs/panoptic/semantic_full --train-images -1 --val-images 512 --epochs 28 --eval-every 7 --batch-size 8 --eval-batch-size 8 --batch-order length-grouped --seed 1337 --selection-seed 1337 --dim 64 --heads 4 --encoder-layers 2 --deterministic
```

Endpoint: the final epoch. Intermediate evaluations are for curve shape only. One seed;
diagnostic, not a replicated finding. No held-out test split is read.

## Decision rule fixed before execution

Resolution on the 512-image validation set: **0.0078**. References: semantic head at 4,096
images `0.206352`, query/mask model at the full split `0.219300`.

Let `pq` be the final-epoch validation pooled PQ, and `accuracy` the per-primitive accuracy
(the head's 4,096-image value was 37.59%).

- **Gains are additive** — `pq > 0.219300 + 0.0078 = 0.227100`. Both directions are worth
  pursuing and the next step is capacity and features for semantics, which is where the
  remaining headroom is: 37.6% accuracy against an oracle that scores 0.808 through the same
  grouper.
- **Architecture gain does not survive data scale** — `pq <= 0.206352`. Then the head's
  advantage was a small-data effect and the query/mask architecture is competitive given
  enough data; the two architectures should be compared at matched data on the *other* axis
  (capacity, features) before either is scaled further.
- **In between** — report the curves, treat the two directions as partially redundant, and
  do not claim additivity.

`accuracy` is reported against the 37.59% reference. The train-set metrics and the distance
to the 0.808 oracle ceiling are reported whatever happens, so a partial result still
localises what remains.

## Results

Pending at preregistration. Record outcomes and any deviations after completion.
