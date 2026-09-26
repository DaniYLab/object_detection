# ArchCAD: does capacity help under the adopted schedule? — preregistered 2026-09-20

## Question

The H17 screen's capacity arm never produced a measurement: the larger model under two-way
concurrency was several times slower per epoch than the screen assumed, and it was stopped when
the annealing arm won (`archcad_annealing_capacity_20260920.md`). The schedule lever has since
been promoted and extended — 0.391101 at 560 epochs, 0.405821 at 1,120 — but its return per
doubling fell from `+0.053237` to `+0.014720` and the last interval is negative, so a third
doubling is expected to buy less than the resolution
(`archcad_annealed_1120_20260920.md`).

That leaves capacity as the only untested axis on the semantic side. It matters more now than it
did in the screen: the earlier "capacity closed" result came from the query/mask pilot, not from
this head, and a schedule that settles changes the comparison, because a bigger model's usual
penalty — failing to converge in the budget — is exactly what annealing addresses.

H21: under the adopted cosine schedule, a larger head beats the current head at equal epochs.

## Design

One factor changes from the winning 140-epoch screen arm (which scored 0.337864): the model size,
from `--dim 64 --heads 4 --encoder-layers 4`-equivalent (69,854 parameters) to
`--dim 128 --heads 8 --encoder-layers 4` (~1.1M parameters). Everything else is held: 140 epochs,
cosine to zero, full split, batch 32, length-grouped, AdamW at 3e-4, weight decay 0.01, gradient
clip 1, FP32, `--deterministic`, seed 1337, validation 512 images, `--eval-every 28`,
`--eval-train-images 2048`, adopted per-class grouping.

```powershell
python scripts/research/train_archcad_semantic.py --output-dir outputs/panoptic/h21_capacity_cosine140 --train-images -1 --val-images 512 --epochs 140 --eval-every 28 --eval-train-images 2048 --batch-size 32 --eval-batch-size 32 --batch-order length-grouped --seed 1337 --selection-seed 1337 --dim 128 --heads 8 --encoder-layers 4 --lr-schedule cosine --lr-final 0 --deterministic
```

Cost: ~60–90 minutes, run alone.

A screen at 140 epochs can miss a capacity effect that only appears with a longer anneal — the
same caveat the H17 screen carried. It is accepted for the same reason: the alternative is a
three-hour run on an axis with no prior measurement for this head.

## Decision rule fixed before execution

Resolution: 0.0078. Reference: **0.337864**, the 140-epoch cosine arm at the current size, so the
gate is **0.345664**.

- **Capacity is live** — `> 0.345664`. Then the size increase is promoted to the full schedule and
  the two winning changes are combined, with the schedule extended again only after that.
- **Capacity is closed for this head** — `<= 0.345664`. Then no semantic lever remains at this
  model size: budget, data, scale features, appearance, schedule and capacity are all measured,
  and the next step is a different architecture (the joint query/mask model) rather than more of
  this one.

Reported either way: the per-interval curve, so the comparison is not an endpoint, and the
parameter count actually trained.

## Results

Pending at preregistration. Record outcomes and any deviations after completion.
