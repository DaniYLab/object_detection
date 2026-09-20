# ArchCAD: longer training at batch 32, and a refit of the grouping radii — preregistered 2026-09-20

## Question

Two levers were live when the grouping work finished, and this is the first run that uses
both at once.

**Exposure for semantics.** The batch-32 curve was still adding quality at its last evaluated
point (`0.30001 → 0.30397 → 0.30580` over epochs 224/252/280) and the batch-8 curve was
adding more per doubling (`+0.0127` in its last interval), so the model is not saturated.
Batch 32 is the cheaper configuration per minute, measured: it reached the same validation
quality as the 112-epoch batch-8 run in 20% less wall clock.

**Grouping radii.** `PER_CLASS_PROXIMITY` was fitted against the predictions of the
`semantic_full_b32_280ep` checkpoint. Better semantics change which primitives carry which
predicted class, so the radii must be refitted whenever the checkpoint changes, and a longer
run is exactly such a change.

H16: a longer run improves validation, and refitting the radii on the new checkpoint adds
further.

## Design

| stage | what | cost |
|---|---|---|
| 1 | train the semantic head, full split, batch 32, 560 epochs | ~100 min |
| 2 | refit per-class proximity radii on the new checkpoint, train split | ~100 min |
| 3 | score on 512 validation images against the current best | minutes |

Stage 1 doubles the epochs of the current best run. Everything else is held: batch 32,
length-grouped, AdamW at LR 0.0003 with weight decay 0.01 and gradient clip 1, FP32,
`--deterministic`, model seed 1337, selection seed 1337, validation 512 images (seed 1338),
geometry-only 12-dimensional input, `--eval-every 28`, `--eval-train-images 2048`.

Stage 2 repeats the two-pass coordinate descent over every class present, with connectivity
fixed at 0.01, which the previous fit showed is not worth tuning per class
(`archcad_per_class_grouping_20260920.md`). The fit objective is the frozen evaluator's pooled
PQ on 1,024 **training** images; validation is touched only in stage 3.

```powershell
python scripts/research/train_archcad_semantic.py --output-dir outputs/panoptic/semantic_full_b32_560ep --train-images -1 --val-images 512 --epochs 560 --eval-every 28 --eval-train-images 2048 --batch-size 32 --eval-batch-size 32 --batch-order length-grouped --seed 1337 --selection-seed 1337 --dim 64 --heads 4 --encoder-layers 2 --deterministic
```

## Decision rule fixed before execution

Resolution: 0.0078. Current best: **0.331524** pooled PQ on 512 validation images, which is
the `b32_280ep` checkpoint scored with the radii fitted against it.

- **Exposure still pays** — stage 3 `> 0.339324`. Then the scale lever continues and the next
  run doubles again; the grouping rule is left as it is.
- **Exposure has flattened** — stage 3 `<= 0.339324` while the train−validation gap stays
  near the current `+0.001543`. Then semantics are limited by something other than exposure at
  this model size, and with appearance, per-primitive shape and capacity all closed, the
  remaining options are a different input representation or a joint model — both of which
  would be new builds rather than more of the same.
- **Overfitting** — stage 3 `<= 0.339324` with the gap widening well beyond `+0.001543`. Then
  the full split cannot absorb 560 epochs at this batch size either, and the same conclusion
  follows as for the flattened case.

Stage 2 is reported either way: the refitted radii and their validation effect say whether the
grouping gain survives a change of checkpoint, which is a claim the previous fit could not
test.

## Results

Pending at preregistration. Record outcomes and any deviations after completion.
