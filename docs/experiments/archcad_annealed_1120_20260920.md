# ArchCAD: a longer annealing schedule — preregistered 2026-09-20

## Question

The 560-epoch cosine run ended at 0.391101 pooled PQ, still climbing: the last four intervals
gained `+0.005479`, `+0.010353`, `+0.008086`, `+0.005855`, all above the 0.0078 resolution, and
per-primitive accuracy reached 0.6299 (`archcad_annealed_560_20260920.md`). The constant-rate
schedule had flattened by epoch 420; the annealed one has not flattened by 560.

With cosine-to-zero over 560 epochs, the endpoint is fully annealed, so more epochs of *that*
schedule do not exist. What can be extended is the schedule: a cosine over 1,120 epochs keeps the
learning rate higher for longer and anneals more slowly at the end.

H20: a 1,120-epoch cosine schedule beats 0.391101 pooled PQ by more than the resolution.

## Design

One factor changes: `--epochs 1120` with the same cosine schedule, so the rate at any epoch is
`0.5·(1 + cos(π·step/total))·3e-4` over twice as many steps. Everything else is identical to the
560-epoch annealed run: full split (27,272 eligible drawings), batch 32, length-grouped, AdamW at
3e-4 with weight decay 0.01 and gradient clip 1, FP32, `--deterministic`, model seed 1337,
selection seed 1337, validation 512 images (seed 1338), geometry-only 12-dimensional input,
`--eval-every 28`, `--eval-train-images 2048`, adopted per-class grouping.

```powershell
python scripts/research/train_archcad_semantic.py --output-dir outputs/panoptic/semantic_b32_cosine1120 --train-images -1 --val-images 512 --epochs 1120 --eval-every 28 --eval-train-images 2048 --batch-size 32 --eval-batch-size 32 --batch-order length-grouped --seed 1337 --selection-seed 1337 --dim 64 --heads 4 --encoder-layers 2 --lr-schedule cosine --lr-final 0 --deterministic
```

Cost: ~180 minutes of training, run alone.

The comparison is deliberately not epoch-matched. A 1,120-epoch cosine and a 560-epoch cosine
reach the same learning rate only at their endpoints, so this asks whether a longer anneal buys a
better endpoint, which is the question the rising curve raises. An epoch-matched comparison
would need a schedule that is not cosine, and is not what is being tested here.

## Decision rule fixed before execution

Resolution: 0.0078. Current best: **0.391101** pooled PQ on the 512 validation images, so the gate
is **0.398901**.

- **Longer anneal still pays** — `> 0.398901`. Then the schedule length is a live lever, the next
  step doubles again, and the run at 560 epochs is recorded as an intermediate point of a longer
  schedule rather than as a configuration.
- **The anneal has flattened** — `<= 0.398901` while the train−validation gap stays near its
  560-epoch value. Then this schedule is at its optimum, and with the semantic levers closed at
  this model size the next step is the joint build rather than more training.
- **Overfitting** — `<= 0.398901` with a widening gap. Same conclusion, recorded separately.

Reported either way: the interval gains against the 560-epoch run's own intervals, so the answer
says whether the extra epochs bought anything or only moved where the anneal happens.

## Results

Pending at preregistration. Record outcomes and any deviations after completion.
