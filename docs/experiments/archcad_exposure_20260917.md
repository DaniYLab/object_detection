# ArchCAD: is exposure still buying anything? — preregistered 2026-09-17

## Question

Every other axis is now closed by measurement: appearance has no channel in the data,
per-primitive shape alone gives 19.78% untrained and 26.31% trained, context is worth
+11.28 points and scaling capacity on top of it overfits, and the geometric grouper plus
component classification cannot be used without ground-truth semantics.

What remains is exposure. At the full training split the semantic head reaches
0.246187 with **train pooled PQ 0.2535 against validation 0.2462** — no generalisation gap
at all, so it is still underfitting — and the data curve in `archcad_full_split_20260917.md`
was still climbing steeply at 27,272 images.

H12: more passes over the same data still improve validation.

## Design

Two arms, both affordable and neither requiring new code.

| arm | architecture | train images | epochs | steps | purpose |
|---|---|---:|---:|---:|---|
| A | transformer, dim 64, 2 layers | 4,096 | 112 | 57,344 | 4x steps at fixed data |
| B | no-context MLP, width 256 | 27,272 | 28 | 95,452 | 6.7x data for the cheap model |

Arm A is the decision arm: it isolates exposure at data and protocol held fixed, and its
reference is the existing `semantic_4096` run at 0.206352. Arm B asks whether the
no-context architecture's 26.31% at 4,096 images improves with the full split, which
matters for choosing how much model the eventual joint build needs.

Everything else follows the semantic-head protocol unchanged: batch 8, length-grouped,
AdamW at LR 0.0003 with weight decay 0.01 and gradient clip 1, FP32, `--deterministic`,
model seed 1337, selection seed 1337, 512 validation images (seed 1338), geometry-only
12-dimensional input, grouper constants proximity 0.020 / connectivity 0.010 fitted on
train, frozen evaluator.

```powershell
python scripts/research/train_archcad_semantic.py --output-dir outputs/panoptic/semantic_4096_112ep --train-images 4096 --val-images 512 --epochs 112 --eval-every 14 --batch-size 8 --eval-batch-size 8 --batch-order length-grouped --seed 1337 --selection-seed 1337 --dim 64 --heads 4 --encoder-layers 2 --deterministic
python scripts/research/train_archcad_semantic.py --output-dir outputs/panoptic/nocontext_full --train-images -1 --val-images 512 --epochs 28 --eval-every 7 --batch-size 8 --eval-batch-size 8 --batch-order length-grouped --seed 1337 --selection-seed 1337 --architecture no_context --width 256 --deterministic
```

A 112-epoch full-split run would be roughly twelve hours and is deliberately **not**
launched: it is only worth its cost if arm A shows exposure still helps.

Endpoint: the final epoch. One seed; diagnostic, not a replicated finding. No test split read.

## Decision rule fixed before execution

Resolution on the 512-image validation set: 0.0078. Arm A reference: 0.206352.

- **Exposure still helps** — arm A's final validation pooled PQ `> 0.214152`. Then the
  head is nowhere near its ceiling at this data scale, and the next run is the full split
  with proportionally more passes, accepting the multi-hour cost.
- **Saturated** — arm A `<= 0.214152` while its train pooled PQ stays within a few points
  of its validation value. Then more passes buy nothing either, every axis of this pipeline
  has a measured limit, and the only route left is to change the objective rather than the
  scale: a model trained jointly over per-primitive embeddings and their grouping, scored
  through the grouper, which is the shape the task actually has.
- **Overfitting** — arm A `<= 0.214152` with train pooled PQ far above validation. Then
  4,096 images cannot absorb 57,344 steps, and the missing ingredient is data rather than
  passes; the full split at 112 epochs becomes the measured next step even at its cost.

Arm B is reported, not gated: it says whether a model with no context at all also benefits
from 6.7x data, which bounds how much of the transformer's advantage is architecture rather
than scale.

## Results

Pending at preregistration. Record outcomes and any deviations after completion.
