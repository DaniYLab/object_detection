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

Completed, ~100 minutes. The run clears the gate decisively:

| epoch | 28 | 140 | 280 | 392 | 448 | 504 | 532 | 560 |
|---|---|---|---|---|---|---|---|---|
| val pooled PQ | 0.238664 | 0.297976 | 0.328674 | 0.345477 | 0.361328 | 0.377160 | 0.385246 | **0.391101** |
| val thing PQ | 0.235612 | 0.291648 | 0.326062 | 0.345952 | 0.362141 | 0.379570 | 0.388323 | 0.396052 |
| per-primitive accuracy | 0.4290 | 0.5268 | 0.5594 | 0.5805 | 0.5993 | 0.6183 | 0.6290 | 0.6299 |

**0.391101** pooled PQ against the 0.351285 gate — cleared by `+0.039816`, 5.1x the resolution —
and `+0.047616` over the previous best 0.343485. Per-primitive accuracy rises from 0.5944 to
0.6299, so the gain is semantic, not a grouping artefact.

Two checks were run on the finished checkpoint, outside the training process
(`score_run_checkpoint.py`): the adopted-protocol number reproduces exactly (0.391101 pooled,
0.396052 thing, 0.6299 accuracy), and the radii fitted on the *previous* checkpoint score
0.392015 on this one — the transfer result holds again, within 0.001, on a checkpoint produced by
a different schedule.

**The curve has not flattened.** Per-interval gains over the last four intervals are `+0.005479`,
`+0.010353`, `+0.008086`, `+0.005855`; the run ends still climbing, at a rate far above the
0.0078 resolution per 28 epochs. This is the preregistered "annealing holds at scale" branch, and
it reopens the exposure question exactly as that branch anticipated: the constant-rate schedule
had flattened by 420 epochs, but a schedule that settles is still paying at 560.

Two consequences, and the first is the next experiment:

1. **The schedule length is now the binding constraint, not the epoch count.** Cosine-to-zero over
   560 epochs means the endpoint is fully annealed; asking for more requires a longer schedule,
   not more epochs of this one. A 1,120-epoch cosine is preregistered in
   `archcad_annealed_1120_20260920.md` with a gate of 0.398901.
2. The screen's 140-epoch result (0.337864) now reads differently from how it was promoted: it was
   not that annealing reaches the old 560-epoch quality in a quarter of the time, it is that
   annealing changes where the curve goes. Four times the epochs of the same schedule add
   `+0.053237`, which is more than any lever has produced since the per-class radii.

Deviations: none beyond those already recorded for the screen. The run was launched alone, so its
timings are not comparable to the concurrent-screen numbers.
