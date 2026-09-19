# ArchCAD: full-split exposure, 112 epochs — preregistered 2026-09-17

## Question

`archcad_full_exposure_20260917.md` doubled the passes at the full training split and
validation pooled PQ moved from 0.246187 to **0.273106**, with the train−validation gap
essentially unchanged (+0.007280 → +0.006157). Exposure is therefore still being converted
into validation quality at 27,272 images, unlike at 4,096 images where it saturated after 14
epochs. The same run also cost 59 minutes, not the six hours estimated, which makes the next
doubling cheap enough to simply run.

The batching measurements (`archcad_batched_step_20260917.md`) say the same work can be done
faster at a larger batch, but changing batch size changes the optimization, so this run keeps
batch 8 to stay comparable with the 28- and 56-epoch points on record.

H14: the full split is still far from its exposure ceiling, so doubling again to 112 epochs
improves validation.

## Design

| arm | images | epochs | steps | val images |
|---|---:|---:|---:|---:|
| reference | 27,272 | 28 | 95,452 | 512 |
| `semantic_full_56ep` | 27,272 | 56 | 190,904 | 512 |
| **H14** | 27,272 | **112** | **381,808** | 512 |

Architecture and protocol unchanged: `ArchCADPrimitiveSemantic`, dim 64, 4 heads, 2 encoder
layers, batch 8, length-grouped, AdamW at LR 0.0003 with weight decay 0.01 and gradient clip
1, FP32, `--deterministic`, model seed 1337, selection seed 1337, validation selection seed
1338, geometry-only 12-dimensional input, frozen evaluator, `--eval-every 14`.

Grouping constants are stated explicitly because the previous preregistration did not match
what ran: **proximity 0.1 / connectivity 0.01**, the same values as every semantic-head run on
record, so this arm stays comparable with them. The train-fitted 0.020/0.010 was measured to
score 0.231259 against 0.246187 on the same checkpoint and is deliberately *not* used here.

```powershell
python scripts/research/train_archcad_semantic.py --output-dir outputs/panoptic/semantic_full_112ep --train-images -1 --val-images 512 --epochs 112 --eval-every 14 --batch-size 8 --eval-batch-size 8 --batch-order length-grouped --proximity 0.1 --connectivity 0.01 --seed 1337 --selection-seed 1337 --dim 64 --heads 4 --encoder-layers 2 --deterministic
```

Cost: about two hours at the measured rate. Endpoint: the final epoch. One seed; diagnostic,
not a replicated finding. No test split read.

## Decision rule fixed before execution

Resolution: 0.0078. Reference: `semantic_full_56ep` at 0.273106.

- **Still climbing** — final validation pooled PQ `> 0.280906` with the train−validation gap
  still small. Then the scale lever is not close to exhausted and the next work is the
  batch-size and throughput question, so that scaling becomes cheap enough to push much
  further rather than a two-hour commitment per doubling.
- **Bending** — final `<= 0.280906` while the gap stays small. Then the curve is flattening
  and the remaining levers are the ones already measured as large: a learned assignment,
  worth `+0.207062` at today's semantic quality by the oracle-grouping measurement.
- **Overfitting** — final `<= 0.280906` with the gap widening well beyond `+0.006157`. Then
  passes are exhausted at the full split and the same conclusion follows.

Whatever happens the eight-point curve is recorded, so where the bend sits is measured rather
than inferred from the endpoint.

## Results

Pending at preregistration. Record outcomes and any deviations after completion.
