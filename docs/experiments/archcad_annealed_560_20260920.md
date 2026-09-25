# ArchCAD: the promoted annealing run at 560 epochs — preregistered 2026-09-20

## Question

The H17 screen measured that annealing is a live lever
(`archcad_annealing_capacity_20260920.md`). At 140 epochs, cosine annealing to zero scored
0.337864 against the constant-rate control's 0.310439 on the same 512 validation images, a gain
of +0.027425 or 3.5x the resolution, with the largest per-interval gain of the run landing in the
final interval (+0.033034 against the control's +0.009401).

The promotion rule written into H17 says the winning change goes to a 560-epoch run, and only a
result above the current best counts as an improvement.

H18: the same run at 560 epochs with cosine annealing beats 0.343485 pooled PQ.

## Design

One factor changes from the current best run: the learning-rate schedule. Everything else is
identical to `semantic_full_b32_560ep` — full training split (27,272 eligible drawings), batch 32,
length-grouped batches, AdamW at 3e-4 with weight decay 0.01 and gradient clip 1, FP32,
`--deterministic`, model seed 1337, selection seed 1337, validation 512 images (seed 1338),
geometry-only 12-dimensional input, `--eval-every 28`, `--eval-train-images 2048`, and the
adopted per-class grouping for the reported numbers.

The schedule is cosine over the whole budget, from 3e-4 to 0, applied per step (the new
`--lr-schedule cosine`, with `--lr-final 0`), so the endpoint is a settled point rather than a
sample from a moving one.

```powershell
python scripts/research/train_archcad_semantic.py --output-dir outputs/panoptic/semantic_b32_cosine560 --train-images -1 --val-images 512 --epochs 560 --eval-every 28 --eval-train-images 2048 --batch-size 32 --eval-batch-size 32 --batch-order length-grouped --seed 1337 --selection-seed 1337 --dim 64 --heads 4 --encoder-layers 2 --lr-schedule cosine --lr-final 0 --deterministic
```

Cost: ~90 minutes of training at the observed rate, run alone so it is not slowed by concurrency.

## Decision rule fixed before execution

Resolution: 0.0078. Current best: **0.343485** pooled PQ on the 512 validation images (the
560-epoch constant-rate checkpoint scored with the radii fitted against it), so the gate is
**0.351285**.

- **Annealing holds at scale** — the run's adopted-protocol validation PQ `> 0.351285`. Then
  annealing replaces the constant schedule everywhere, the grouping radii are refitted on the new
  checkpoint (which the H16 result says will move pooled PQ by ~0.0001, so it is a formality), and
  the exposure question reopens: with a schedule that settles, longer runs may pay again where
  the constant schedule had flattened.
- **The screen did not transfer** — `<= 0.351285`. Then the 140-epoch gain was a short-budget
  effect, the constant-rate 560-epoch endpoint stands, and the semantic side of the pipeline has
  no live lever left at this model size — which makes the joint build the next step rather than
  more training.

Reported either way: the full 28-epoch curve, so the gain can be attributed to intervals rather
than to an endpoint, and the per-primitive accuracy, so it is visible whether the gain is
semantics or grouping.

## Results

Pending at preregistration. Record outcomes and any deviations after completion.
