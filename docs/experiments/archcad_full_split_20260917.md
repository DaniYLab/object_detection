# ArchCAD full training split — preregistered 2026-09-17

## Question

H6 showed data is a live lever and that its effect grows with the step budget: at 3,584
steps, 4,096 images beat 256 by `+0.025173`; at 14,336 steps the same comparison gives
`+0.053686`. Validation also improved over the control for the first time
(`E − control = +0.033217`).

But the experiment used 4,096 of the **28,771** training images available in the public-40K
train split — 14%. The fixed 256-image subset that every earlier result was measured on is
0.9%. The dose-response curve has never been sampled above 4,096.

H8: validation continues to improve when the full eligible training split is used, at
matched per-drawing exposure.

## Design

| Arm | Train images | Epochs | Steps | Val images |
|---|---:|---:|---:|---:|
| A′ | 4,096 | 28 | 14,336 | 512 |
| F | all eligible (up to 28,771) | 28 | `ceil(N/8) × 28` | 512 |

Matched **epochs**, so every drawing receives 28 passes in both arms; the arms differ in
how many drawings exist. That is the same exposure-matched comparison the 2×2 showed to be
necessary, and the full-split arm is a nested superset of the 4,096-image arm because
selection is a prefix of one seeded shuffle.

The eligible count `N` is not known in advance and is reported by the run: eligibility is
1..1024 primitives and 1..64 segments, and the corpus has a long primitive tail (mean 350
per image, maximum 81,472) plus 261-segment drawings, so `N < 28,771`. `--train-images -1`
now means "every eligible image", and `config.training.train_images_resolved` records the
result.

Arm A′ is already produced by the scale-features experiment
(`outputs/panoptic/val512_frozen_4096`, val pooled PQ 0.14851, recall 0.2335) and is reused
rather than re-run.

## Frozen protocol

Unchanged from the batched-step protocol: `ArchCADPrimitiveBatched`, `dim=64`, 64 queries,
geometry-only four-point features (12 dimensions, **not** the rich variant, which showed no
resolvable effect), batched B=8 length-grouped, AdamW LR 0.0003 with weight decay 0.01 and
gradient clip 1, frozen loss (2/7/5, no-object 0.05), frozen decoder and evaluator,
`--deterministic`, model seed 1337, selection seed 1337, validation selection seed 1338.

Endpoint: the final epoch. Intermediate evaluations are for curve shape only. One seed;
diagnostic, not a replicated finding. No held-out test split is read.

## Command

```powershell
python scripts/research/train_archcad_primitive_batched.py --output-dir outputs/panoptic/scaleData_full --train-images -1 --val-images 512 --epochs 28 --eval-every 7 --batch-size 8 --eval-batch-size 8 --batch-order length-grouped --seed 1337 --selection-seed 1337 --deterministic
```

## Decision rule fixed before execution

Resolution on the 512-image validation set: **0.0078**, half the measured 128-image fold
spread. Let `delta = final_val_pooled_PQ(F) − 0.14851`.

- **Data still live** — `delta > +0.0078`. Then the curve is still climbing at 7x the
  largest set tested, and the next step is to keep scaling exposure (more epochs at the
  full split) rather than changing the representation.
- **Data saturates** — `|delta| <= 0.0078`. Then the data lever is exhausted between
  4,096 and 28,771 images at 28 passes, and the agenda becomes representational:
  the appearance branch (raster feature sampling), which the scale-features result already
  points to from the other direction.
- Anything else is **INCONCLUSIVE** and is recorded as such.

Train-set metrics and the train/val gap are reported either way, since a data-scale arm
that only overfits less is also informative.

## Results

Pending at preregistration. Record outcomes and any deviations after completion.
