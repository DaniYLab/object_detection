# ArchCAD optimization checks — preregistered 2026-09-17

## Hypotheses

H1: The current primitive-only model can memorize a small training set.
H2: The 20-epoch baseline is budget-limited, so a fixed 100-epoch run improves
validation thing recall on the identical selected data.

Keep representation, model (149,343 params), loss weights 2/7/5, query count 64,
optimizer AdamW (LR 0.0003, weight decay 0.01), gradient clipping 1.0 and decoder
thresholds unchanged. No scheduler or augmentation is added.

## Fixed experiments

1. Overfit: first 8 eligible train images under selection seed 1337, seed 1337
   initialization, 200 epochs / 1,600 updates. Evaluate those 8 train images and
   16 independent validation images at epochs 0/20/40/.../200. The overfit test
   concerns training performance; its validation is diagnostic only.
2. Budget: same 256 train / 128 val subset and seed 1337 as the preceding
   replication. Cold-train for 100 epochs / 25,600 updates. Evaluate all selected
   train/val images at epochs 0/20/40/60/80/100 and retain each checkpoint.

Thresholds for interpretation fixed before execution:

- Strong overfit: training thing recall >=0.95 and pooled PQ >=0.90 at epoch 200.
  Failure only means the current setup did not overfit within the fixed budget.
- Budget effect: final validation thing recall improves by >=0.03 absolute
  versus epoch 20, without pooled PQ regression. One seed is diagnostic, not a
  replicated architectural result. Inspect intermediate curves but use final
  epoch as endpoint, not the best observed validation epoch.
- Verify epoch-20 control against the saved seed-1337 result within numerical
  tolerance (1e-6 scalar metric error). Reject a controlled comparison if selected
  IDs/source hashes, core code hashes or early trajectory differ materially.

The runner adds only monitoring: evaluation has no gradient updates or random
sampling. No training code is edited while a run is active. No held-out test
predictions are produced. Reports use the local v1 metric profile whose paper
parity remains unverified.

## Commands

```powershell
python scripts/research/train_archcad_primitive_pilot.py --output-dir outputs/panoptic/overfit8_20260917 --train-images 8 --val-images 16 --epochs 200 --eval-every 20 --seed 1337 --selection-seed 1337
python scripts/research/train_archcad_primitive_pilot.py --output-dir outputs/panoptic/budget100_20260917_s1337 --train-images 256 --val-images 128 --epochs 100 --eval-every 20 --seed 1337 --selection-seed 1337
```

## Results

Pending at preregistration. Record outcomes and any deviations after completion.
