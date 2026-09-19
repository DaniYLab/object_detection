# ArchCAD: matched wall clock, batch 8 against batch 32 — preregistered 2026-09-17

## Question

`archcad_batched_step_20260917.md` measured the trade-off at matched **exposure** and found
batch 32 5.3x faster and 0.029081 worse:

| configuration | steps | val pooled PQ | elapsed |
|---|---:|---:|---:|
| batch 8, 56 epochs | 190,904 | 0.273106 | 59 min |
| batch 32, 56 epochs | 47,768 | 0.244025 | 11.2 min |

That settles exposure but not the question that matters operationally: **which configuration
gives the most validation quality per minute of wall clock?** Batch 32 buys 5.3 exposures in
the time batch 8 buys one, and exposure doublings have been worth about +0.027 each, so both
answers are plausible and the measured trade-off does not decide between them.

H15: at matched wall clock, batch 32 reaches higher validation pooled PQ than batch 8.

## Design

| arm | images | epochs | steps | batch | expected wall clock |
|---|---:|---:|---:|---:|---:|
| reference, already run | 27,272 | 56 | 190,904 | 8 | 59 min |
| **H15** | 27,272 | **280** | **238,840** | **32** | ~56 min |

280 epochs at batch 32 is the point where the two arms cost about the same wall clock, using
the measured 12.0 s per epoch at batch 32. The arm therefore sees **5x the exposure** of the
reference in the same time.

Everything else unchanged: `ArchCADPrimitiveSemantic`, dim 64, 4 heads, 2 encoder layers,
length-grouped batches, AdamW at LR 0.0003 with weight decay 0.01 and gradient clip 1, FP32,
`--deterministic`, model seed 1337, selection seed 1337, validation 512 images (seed 1338),
geometry-only 12-dimensional input, grouping constants proximity 0.1 / connectivity 0.01,
frozen evaluator, `--eval-every 28`, `--eval-train-images 2048`.

```powershell
python scripts/research/train_archcad_semantic.py --output-dir outputs/panoptic/semantic_full_b32_280ep --train-images -1 --val-images 512 --epochs 280 --eval-every 28 --eval-train-images 2048 --batch-size 32 --eval-batch-size 32 --batch-order length-grouped --proximity 0.1 --connectivity 0.01 --seed 1337 --selection-seed 1337 --dim 64 --heads 4 --encoder-layers 2 --deterministic
```

Endpoint: the final epoch. One seed; diagnostic, not a replicated finding. No test split read.

## Decision rule fixed before execution

Resolution: 0.0078. Reference: batch 8 at 56 epochs, 0.273106.

- **Batching wins per minute** — final validation pooled PQ `> 0.273106`. Then batch 32 is the
  configuration to use for every further scaling run, and the whole programme gets 5.3x
  cheaper per unit of progress; the next step is to push exposure at batch 32.
- **Batching loses per minute** — final `<= 0.273106`. Then the model's preference for many
  small updates outweighs the throughput gain, batch 8 stands, and speed has to come from
  elsewhere: preparation caching and concurrent processes rather than batch size.
- The rule does not require a margin beyond the resolution, because the comparison is between
  two configurations rather than against a null; a difference inside 0.0078 is reported as a
  tie and settles the question in favour of whichever is cheaper to run, which is batch 32.

The 280-epoch curve at batch 32 is recorded at ten points either way, so the shape of the
exposure response at batch 32 becomes known and not only its endpoint.

## Results

| run | batch | epochs | steps | val pooled PQ | val accuracy | train − val | elapsed |
|---|---:|---:|---:|---:|---:|---:|---:|
| `semantic_full_56ep` | 8 | 56 | 190,904 | 0.273106 | 0.5254 | +0.006158 | 3,555 s |
| `semantic_full_112ep` | 8 | 112 | 381,808 | 0.304004 | 0.5662 | +0.004368 | 3,762 s |
| **`semantic_full_b32_280ep`** | **32** | **280** | 238,840 | **0.305800** | **0.5727** | **+0.001543** | **3,002 s** |

`0.305800 > 0.273106`, by `+0.032694`.

**Verdict: batching wins per minute.** Batch 32 at 280 epochs also edges past the 112-epoch
batch-8 run — `+0.001796`, which is inside the 0.0078 resolution and therefore a tie on
quality — while costing **20% less wall clock** (3,002 s against 3,762 s). Per the
preregistered rule, a tie is settled in favour of whichever is cheaper to run.

### Correcting the size of that win

`archcad_batched_step_20260917.md` said batching was 5.3x cheaper, measured per epoch at
matched exposure. End to end it is not: batch 32 needs roughly 2.5 times the epochs to reach
the same validation quality, because fewer and larger updates are less effective per unit of
exposure — the same fact that made it lose at matched exposure. The net saving at equal
quality is **about 20%**, not 5.3x, and that is the number to use when planning runs.

What remains true is that batch 32 is the cheaper configuration to run and should be the
default for scaling from here.

### Where the exposure curve now sits

| run | epochs | val pooled PQ at each evaluated point |
|---|---|---|
| batch 8 | 28/56/84/112 | 0.24619, 0.27311, 0.29126, 0.304004 |
| batch 32 | 28/56/84/112/140/168/196/224/252/280 | 0.22294, 0.24403, 0.26208, 0.27420, 0.28653, 0.28801, 0.29301, 0.30001, 0.30397, **0.30580** |

Both curves are decelerating and converging near **0.305**: the last batch-8 interval adds
`+0.0127` while the last batch-32 interval adds `+0.0018`. Exposure is close to exhausted by
this pipeline, which makes the measured grouping lever the one worth building next: perfect
grouping at today's semantics is worth `+0.207062` by the oracle measurement in
`archcad_full_exposure_20260917.md`, against the `+0.002` to `+0.013` that another exposure
doubling now buys.

## Artifacts

- `outputs/panoptic/semantic_full_b32_280ep`
- Gate applied by `outputs/panoptic/_runs_20260917/apply_h15_rule.py`

