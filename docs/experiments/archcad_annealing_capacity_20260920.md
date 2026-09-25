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

### The annealing arm wins

Control and arm, both scored under the adopted protocol on the same 512 validation images
(control is the 560-epoch run's own checkpoints, which are bit-identical to a constant-rate
140-epoch run):

| epoch | 28 | 56 | 84 | 112 | 140 |
|---|---|---|---|---|---|
| control, constant rate | 0.238057 | 0.266697 | 0.287463 | 0.301038 | 0.310439 |
| arm B, cosine to zero | 0.239988 | 0.269423 | 0.282346 | 0.304830 | **0.337864** |

Per-interval gains for the arm are `+0.029435`, `+0.012923`, `+0.022484`, `+0.033034`: the last
interval is the largest of the run, and the control's is `+0.009401`. That is the annealing
signature — as the rate goes to zero the model stops orbiting and settles — and it is the same
pathology the 560-epoch constant-rate run showed from the other side, where the endpoint was the
top of a 0.013361 oscillation.

At 140 epochs the arm is **+0.027425** above the control, 3.5x the resolution. Per-primitive
accuracy rises with it (0.5578 against 0.5385). The arm also ends close to the constant-rate
560-epoch endpoint (0.343407) at a quarter of the epochs, which is worth noting for budgeting:
annealing buys in 140 epochs what the constant schedule needed 560 for.

**Rule application: a lever is live.** The arm clears the 0.318239 gate, so the winning change is
promoted to a 560-epoch run, preregistered in `archcad_annealed_560_20260920.md`, where only a
result above the current best 0.343485 counts as an improvement.

### The capacity arm was stopped, and that is a deviation

Arm C (`--dim 128 --heads 8 --encoder-layers 4`) ran for about 50 minutes without completing its
first evaluation interval: the larger model under two-way concurrency was several times slower
per epoch than the screen assumed, and it had not produced a single measurement. It was stopped
when the annealing arm won, so that the promoted run could have the machine to itself, and its
question — does capacity move this head at all — is left open and cheap to re-run later. Arm D
(capacity plus annealing) was never started.

Two smaller deviations are recorded for the same reason. The three arms were launched
concurrently as designed and the machine ran out of memory (three ~4 GB processes plus the
host's own 20 GB service): the cosine arm survived, the capacity and combined arms died at their
initial evaluation and were relaunched one at a time. And the run was interrupted once by a
session suspension, after which the cosine arm was restarted from scratch; its reproduced
epoch-28 and epoch-56 values match the pre-interruption run exactly (0.239988 and 0.269423), so
the restart did not change the trajectory.
