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

Recorded after completion of `scale_dim64_1000`. The preregistered checks were run
first and both passed; the decision rule was then applied to the preregistered
endpoint.

### Preregistered checks: passed

- Selection is identical to the saved `primitive_256_s1337` run: 256 train / 128 val
  image IDs, source hashes, split fingerprint `b0f01e46…5118c` and eligibility skips
  all match, so the comparison is controlled.
- The 20-epoch control **reproduced `det_control_a` exactly** — validation loss
  `7.182919263839722` and all `semantic`/`instance`/`panoptic` metric blocks
  bit-identical — even though two script revisions (the determinism restart guard and
  the opt-in decoder parameter) landed in between. That is a second, cross-revision
  confirmation that both changes are inert.

### H3 — budget effect on the real subset: PLATEAU, not supported

`scale_dim64_1000`, deterministic, seed 1337, 1000 epochs = 256,000 steps:

| Epoch | Val loss | Val recall | Val pooled PQ | Train recall | Train pooled PQ |
|---:|---:|---:|---:|---:|---:|
| 0 | 14.8242 | 0.0009 | 0.00077 | 0.0000 | 0.0000 |
| 20 (control) | 7.1829 | 0.1106 | 0.11165 | 0.1517 | 0.1449 |
| 100 | 10.0303 | 0.1142 | 0.10905 | 0.4138 | 0.3115 |
| 200 | 14.9250 | 0.0734 | 0.10028 | 0.5185 | 0.4449 |
| 400 | 18.4734 | 0.0771 | 0.09713 | 0.6486 | 0.5414 |
| 600 | 20.6544 | 0.0861 | 0.10707 | 0.7019 | 0.5938 |
| 800 | 20.6798 | 0.0780 | 0.10276 | 0.7105 | 0.6190 |
| 1000 | 22.1765 | 0.0707 | **0.09733** | 0.7489 | **0.6503** |

`delta = 0.09733211824663637 − 0.11164704552069724 = −0.014315`, which is below the
`+0.02` threshold, and the last three evaluations span `0.007949` pooled PQ, under the
`0.01` plateau bound. **Verdict: PLATEAU.**

Reading: fifty times the budget makes the model fit its training set much better
(train pooled PQ 0.1449 → 0.6503, train recall 0.1517 → 0.7489, train instance AP50
0.6044) while validation degrades monotonically from epoch 200 (val loss 14.93 → 22.18,
val pooled PQ 0.11165 at the control → 0.09733). Validation peaked early, at epoch 100
(0.10905), and never recovered. The limitation on this subset is therefore **not**
optimization budget: it is generalization with 256 training images, given a
geometry-only four-point representation.

**Correction to an earlier claim in this repository's working notes.** The 1000-epoch
result falsifies the inference that the 20-epoch control was simply underfit and that
spending GPU hours on budget would raise validation. Both statements were recorded
before this run existed. The control sits close to the best validation performance
this configuration reaches; the frozen pilot's numbers are not "measured on an
unfit control" in the sense of being improvable by more epochs.

`scale_dim128_1000` is the H4 arm and is reported when it completes.

### What the verdict implies

Per the preregistration, PLATEAU means the next step is **not** more epochs. The two
untested levers, in the order the research plan already fixed, are:

1. **Representation** — add raster feature sampling and adaptive fusion (plan step 5).
   The model is geometry-only: 12 numbers per primitive with no image context.
2. **Data scale** — the fixed 256-image subset is 0.9% of the 28,771 available train
   images. Training on the full split was never attempted, and with the batched-step
   path (`archcad_batched_step_20260917.md`, ~5.9x throughput) a full-split epoch is
   now affordable, so this is the cheapest untested lever.

### Deviations

- The 20-epoch control re-runs `det_control_a` under a newer script revision; the
  bit-exact match above is what makes it usable as a same-revision control.
- `scale_dim64_1000` and `scale_dim128_1000` were executed while a separately
  monitored throughput benchmark ran on the same GPU. That benchmark changes wall
  clock only; its own runs were verified bit-identical to a sequential reference.
- One recorded run took longer than expected: 3883 s for 1000 epochs. No protocol
  parameter was changed in response.

### Artifacts

- `outputs/panoptic/scale_dim64_20_control`, `scale_dim64_1000`, `scale_dim128_1000`
- Decision applied by `outputs/panoptic/_runs_20260917/apply_decision_rule.py`
- Preregistered checks by `outputs/panoptic/_runs_20260917/check_scaling.py`

