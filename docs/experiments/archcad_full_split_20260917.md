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

Completed in 92 minutes, of which roughly 25 were preparation (parsing and decoding 27,272
drawings). The eligible count is **27,272** of the 28,771 training images — 1,499 are
excluded by the 1..1024 primitive and 1..64 segment caps, which is the first time that
number has been measured. 3,409 steps per epoch, 95,452 steps total.

| Epoch | Val loss | Val pooled PQ | Val recall | Train pooled PQ |
|---:|---:|---:|---:|---:|
| 0 | 14.8207 | 0.00020 | 0.0002 | 0.0001 |
| 7 | 6.2734 | 0.15917 | 0.2522 | 0.1502 |
| 14 | 5.8919 | 0.19959 | 0.3105 | 0.1908 |
| 21 | 5.6830 | 0.20509 | 0.3187 | 0.1989 |
| 28 | 5.4262 | **0.21930** | **0.3330** | 0.2188 |

`delta = 0.21930 − 0.14851 = +0.070790`, far above the 0.0078 resolution.

**Verdict: DATA STILL LIVE.** H8 is supported, and the dose-response is still climbing
steeply rather than bending: 256 → 4,096 images was worth +0.025 at an equal step budget,
and 4,096 → 27,272 is worth +0.0708 here.

Train pooled PQ 0.2188 against validation 0.21930 — there is no generalisation gap left at
this data scale. The model is not overfitting; it is still underfitting, so both more data
and more steps remain available.

### What this changes

The data lever answered the question the 2×2 raised, and it did so decisively: this is the
largest single improvement measured for the query/mask architecture all day, obtained by
using the data that was already on disk. The earlier sessions' 256-image subset was 0.9% of
the training split and every result before today was measured there.

It also resets what "the model" means for the next comparison. The query/mask architecture
at 27,272 images scores 0.21930, which is above the per-primitive semantic head at 4,096
images (0.20635). The head still wins at matched data — 0.20635 against 0.14851 at 4,096 —
so the two are independent wins that have not been combined.

### Deviations

- Preparation took about 25 minutes, longer than the estimate, and is included in the
  elapsed time. No protocol parameter was changed.
- The run shared the machine with read-only probes and a concurrent semantic-head run. That
  affects wall clock only.

### Artifacts

- `outputs/panoptic/scaleData_full` with `progress.json`
- Resolved counts in `config.training.train_images_resolved` / `val_images_resolved`

