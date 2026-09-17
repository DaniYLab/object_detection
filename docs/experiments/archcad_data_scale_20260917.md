# ArchCAD data scale at matched optimizer steps — preregistered 2026-09-17

## Question

`scale_dim64_1000` returned PLATEAU: fifty times the budget on the fixed 256-image
subset fits the training set better (train pooled PQ 0.1449 → 0.6503) while validation
degrades. That verdict says the constraint is not optimization budget, and leaves two
untested levers: the geometry-only representation, and the amount of training data.

This experiment tests the data lever. The subset is 256 of 28,771 available training
images, so "is validation limited by how many drawings we train on?" is answerable
before touching the architecture.

H5: at a fixed number of optimizer steps, more training drawings improve final-epoch
validation quality.

## Design: nested subsets at matched steps

`select_samples` takes the first N eligible images from one seeded shuffle of the whole
split, so the arms are **nested** — the 256-image set is exactly the first 256 images of
the 1,024 set, which is the first 1,024 of the 4,096 set. Training-set size is therefore
the only variable that changes across arms.

Batch size 8 for every arm, so equal steps also means equal images consumed:

| Arm | Train images | Epochs | Steps/epoch | Total steps | Images seen |
|---|---:|---:|---:|---:|---:|
| A | 256 | 112 | 32 | 3,584 | 28,672 |
| B | 1,024 | 28 | 128 | 3,584 | 28,672 |
| C | 4,096 | 7 | 512 | 3,584 | 28,672 |

## Frozen protocol

- Model: `ArchCADPrimitiveBatched`, `dim=64`, `heads=4`, `encoder_layers=1`,
  `decoder_layers=2` — identical parameters to the frozen pilot, 149,343.
- Protocol: **batched step** (`archcad_batched_step_20260917.md`), B=8, length-grouped
  batch order, one optimizer step per batch. All three arms use it identically, so the
  comparison is internally valid. Its metrics were verified identical to the frozen
  evaluator on shared weights, which is what makes the arms comparable to each other on
  the reporting path.
- Input: geometry-only four-point features, unchanged; no GT label, layer or RGB prior.
- Data: existing SHA-group public-40K split, fingerprint `b0f01e46…5118c`. Validation is
  the same 128 images for every arm (selection seed 1338). Training selection seed 1337.
- Eligibility: 1..1024 primitives and 1..64 segments, no truncation; skipped candidates
  reported per arm.
- Optimizer: AdamW, LR 0.0003, weight decay 0.01, gradient clip 1, FP32, no scheduler,
  no augmentation.
- Loss: frozen per-image loss, mean over the batch; weights 2/7/5, no-object 0.05.
- Decoder and evaluator: frozen, untouched, default thresholds.
- Determinism: `--deterministic` on every arm.
- Endpoint: **the final epoch**, which is step 3,584 for all arms. Intermediate
  evaluations are for curve shape only; the best observed epoch is never reported as a
  result.
- One seed (1337). Diagnostic, not a replicated finding.
- No held-out test split is read.

## Commands

```powershell
python scripts/research/train_archcad_primitive_batched.py --output-dir outputs/panoptic/scaleData_256 --train-images 256 --val-images 128 --epochs 112 --eval-every 28 --batch-size 8 --eval-batch-size 8 --batch-order length-grouped --seed 1337 --selection-seed 1337 --deterministic
python scripts/research/train_archcad_primitive_batched.py --output-dir outputs/panoptic/scaleData_1024 --train-images 1024 --val-images 128 --epochs 28 --eval-every 7 --batch-size 8 --eval-batch-size 8 --batch-order length-grouped --seed 1337 --selection-seed 1337 --deterministic
python scripts/research/train_archcad_primitive_batched.py --output-dir outputs/panoptic/scaleData_4096 --train-images 4096 --val-images 128 --epochs 7 --eval-every 2 --batch-size 8 --eval-batch-size 8 --batch-order length-grouped --seed 1337 --selection-seed 1337 --deterministic
```

## Decision rule fixed before execution

Noise floor on this 128-image validation subset, measured earlier on the deterministic
100-epoch run: **0.016429** epoch-to-epoch pooled PQ. Deltas below it are not resolvable.

Let `best` be the largest training set that completes.

- **Data-limited** — final-epoch validation pooled PQ increases from arm A to arm B to
  arm C, and arm C exceeds arm A by more than the noise floor. Next step: scale training
  data further (toward the full 28,771-image split) before changing the representation.
- **Not data-limited** — all arms fall within the noise floor of each other. Then
  training-set size up to 4,096 is not the constraint, and the geometry-only
  representation becomes the primary suspect: proceed to raster feature sampling and
  fusion (plan step 5).
- **Inconclusive** — anything else, including a non-monotone or partially completed set.
  Record the curves and do not switch direction on this evidence alone.

All three arms are reported whatever the outcome, including the train-set metrics, so a
non-result is still informative.

## Results

Pending at preregistration. Record outcomes and any deviations after completion.
