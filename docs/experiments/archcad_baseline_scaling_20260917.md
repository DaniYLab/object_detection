# ArchCAD baseline budget/capacity scale-up — preregistered 2026-09-17

## Question

The primitive-only control on the fixed 256 train / 128 val subset is trained for 20
epochs / 5,120 optimizer steps. That budget is now known to be far below what this
architecture needs to fit even eight images (see
`archcad_optimization_20260917.md`). Every downstream comparison — raster fusion,
conditioned reflex, representation changes — is currently measured against a control
that has not begun to fit its own training set.

H3: at fixed capacity (`dim=64`), extending the budget on the fixed subset improves
final-epoch validation quality rather than only overfitting.

H4: at a fixed longer budget, `dim=128` (1,107,231 parameters) improves final-epoch
validation quality over `dim=64` (149,343 parameters) under the identical budget.

## Frozen protocol

- Architecture: `ArchCADPrimitiveBaseline`; `dim=64` (`heads=4`, `encoder_layers=1`,
  `decoder_layers=2`) or `dim=128` (`heads=4`, `encoder_layers=2`, `decoder_layers=4`).
  No other architecture is introduced and no pretrained weights are used.
- Input: geometry-only four-point features per entity, unchanged; no GT label, layer or
  RGB prior.
- Data: the existing SHA-group public-40K train/val split, fingerprint
  `b0f01e46718cbdef041c5b43a61bebec372acf1852d67d1f640d5772a335118c`. Select 256 train /
  128 validation images with selection seed 1337 (val 1338). The selected image IDs and
  source hashes must be identical to the saved `primitive_256_s1337` run; verify by
  comparing `config.json` `train`/`val` entries before interpreting any result.
- Eligibility: 1..1024 primitives and 1..64 target segments, no truncation.
- Model seed 1337. One seed only: this is a diagnostic run to choose a direction, not a
  replicated result, and it must not be quoted as a three-seed finding.
- Budget: 1000 epochs = 256,000 optimizer steps per run, batch = 1 image.
- Optimizer: AdamW, fixed LR = 0.0003, weight decay 0.01, gradient clip 1, FP32.
  No scheduler, no augmentation.
- Loss: class CE + mask BCE + Dice, weights 2/7/5, no-object class weight 0.05.
- Decoder: frozen. Score threshold 0.1, mask threshold 0.5, `argmax(score × mask)`
  assignment, stuff merged. Not tuned here; `mask_threshold` was already shown to be
  inert on a converged model.
- Determinism: `--deterministic` on every run. Two independent 20-epoch runs already
  proved bit-identical, so this removes the control-comparison problem recorded in
  `archcad_optimization_20260917.md`.
- Evaluation: `--eval-every 100`, all selected train and validation images. Intermediate
  evaluations are for inspecting curve shape only; **the endpoint is the final epoch**.
  The best observed epoch is never reported as a result.
- Metrics: local v1 semantic F1/wF1, primitive-mask AP, pooled total/thing/stuff PQ/SQ/RQ.
  Published-evaluator parity remains unverified.
- No held-out test split is read, and no threshold or hyperparameter is changed in
  response to validation observations.

## Fixed experiments

```powershell
python scripts/research/train_archcad_primitive_pilot.py --output-dir outputs/panoptic/scale_dim64_20_control --train-images 256 --val-images 128 --epochs 20 --eval-every 20 --seed 1337 --selection-seed 1337 --deterministic
python scripts/research/train_archcad_primitive_pilot.py --output-dir outputs/panoptic/scale_dim64_1000 --train-images 256 --val-images 128 --epochs 1000 --eval-every 100 --seed 1337 --selection-seed 1337 --deterministic
python scripts/research/train_archcad_primitive_pilot.py --output-dir outputs/panoptic/scale_dim128_1000 --train-images 256 --val-images 128 --epochs 1000 --eval-every 100 --seed 1337 --selection-seed 1337 --dim 128 --heads 4 --encoder-layers 2 --decoder-layers 4 --deterministic
```

The 20-epoch run is not a new treatment: it re-runs `det_control_a` under the current
script revision. It serves as the same-revision control for H3 and as a second
cross-revision reproducibility check (it must match `det_control_a` exactly).

## Decision rule fixed before execution

Reference value: deterministic 20-epoch validation pooled PQ = **0.11164704552069724**
(`det_control_a`). Known noise on this 128-image validation subset, measured on the
deterministic 100-epoch run, is an epoch-to-epoch pooled-PQ range of **0.016429**
(0.102589–0.119018 over the five trained evaluations). Any smaller delta is not
resolvable here.

Let `delta = final_epoch_val_pooled_PQ − 0.111647` for the run under test.

- **Budget-limited** — `delta >= +0.02` for `scale_dim64_1000`: more budget helps, so
  scale further before changing the representation or adding fusion.
- **Plateau** — `delta < +0.02` **and** the last three evaluations of that run span
  less than 0.01 pooled PQ: the primitive-only configuration has stopped responding to
  budget, so the next step is raster feature sampling and fusion rather than more epochs.
- **Inconclusive** — anything else, including a delta inside the noise floor or a curve
  still climbing at the end. Record it as inconclusive and extend the budget at the same
  configuration rather than switching direction.

H4 is secondary and reported, not gated: if `scale_dim128_1000` beats `scale_dim64_1000`
in final-epoch validation pooled PQ by more than the noise floor, capacity is a live
variable for the next stage; otherwise the 1.1M-parameter variant is not carried forward.

## Results

Pending at preregistration. Record outcomes and any deviations after completion.
