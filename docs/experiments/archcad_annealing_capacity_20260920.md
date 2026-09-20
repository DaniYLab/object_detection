# ArchCAD: does the semantic head need annealing or capacity? — preregistered 2026-09-20

## Question

The 560-epoch run ended on two symptoms that the H16 decision rule was not written to
interpret (`archcad_longer_training_20260920.md`). Its last third oscillates by 0.013361 pooled
PQ between adjacent checkpoints while the trend across it is +0.005325, below the 0.0078
resolution; and its train−validation gap widened from `+0.001543` to `+0.007922` as validation
stalled. Both are what a constant learning rate looks like from the outside: the model keeps
moving around a basin instead of settling into it.

The head is also small and was never sized for this task. It has 69,854 parameters at
`dim 64`, 4 heads, 2 layers. The capacity sweep of 2026-09-17 was run on the query/mask pilot,
a different model with a different output contract, so "capacity closed" does not cover this
head.

H17: annealing the learning rate over the budget, or enlarging the head, or both, beats the
constant-rate 140-epoch point by more than the resolution.

## Design

A 140-epoch screen — a quarter of the full run — on the full training split, with the frozen
protocol of the 560-epoch run: batch 32, length-grouped, AdamW, weight decay 0.01, gradient
clip 1, FP32, `--deterministic`, model seed 1337, selection seed 1337, validation 512 images
(seed 1338), geometry-only 12-dimensional input, adopted per-class grouping, `--eval-every 28`,
`--eval-train-images 2048`.

| arm | change from the control | cost |
|---|---|---|
| A control | none — the 560-epoch run's epoch-140 checkpoint | already exists |
| B annealed | `--lr-schedule cosine` to 0 over 140 epochs | ~22 min |
| C capacity | `--dim 128 --heads 8 --encoder-layers 4`, constant rate | ~22 min |
| D both | C with B's schedule | ~22 min |

The control needs no run. The 560-epoch run's first 280 epochs are bit-identical to the
280-epoch run's (10 of 10 evaluated epochs equal in loss, train PQ and validation PQ), so its
epoch-140 weights are exactly what a 140-epoch constant-rate run produces. Its adopted-protocol
score is measured before the arms start:

```powershell
python outputs/panoptic/_runs_20260917/h17_control.py
```

Arms B, C and D run concurrently. Concurrency was verified bit-exact earlier (independent
processes, 2.3x aggregate throughput at K=8) and the GPU sits at 4–8% utilisation in the
single-process configuration, so the arms do not share a bottleneck.

## Decision rule fixed before execution

Resolution: 0.0078. Control at 140 epochs: **0.310439** pooled PQ on 512 validation images
(thing PQ 0.310896, per-primitive accuracy 0.5385), measured 2026-09-20.

- **A lever is live** — an arm scores `> 0.318239`. Then the winning change is promoted to a
  560-epoch run, preregistered separately, and only a result above the current best `0.343485`
  is treated as an improvement.
- **Neither moves this head** — no arm clears `0.318239`. Then the residual error is not an
  optimisation or size artefact at this budget, and with appearance closed by measurement
  (monochrome raster, constant styling fields) the next build is the joint query/mask model
  rather than more of the same.

A screen at 140 epochs can miss a lever that only pays later; that risk is accepted because the
alternative is three 90-minute runs, and because the symptoms being treated — endpoint
oscillation and a widening gap — are visible well before 140 epochs in the existing records
(the gap is already `+0.003436` at 140).

## Results

Pending at preregistration. Record outcomes and any deviations after completion.
