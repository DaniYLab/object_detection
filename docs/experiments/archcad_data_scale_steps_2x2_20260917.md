# ArchCAD data scale × steps, 2×2 — preregistered 2026-09-17

## Question

`archcad_data_scale_20260917.md` matched its three arms on optimizer steps, which forced
the larger training sets to receive fewer passes: 112 passes over 256 drawings versus 7
over 4,096. Its verdict was INCONCLUSIVE, and the document records that data quantity is
entangled with repetition count in that design. Arm C (4,096 images) also stopped
overfitting without converging — its train and validation pooled PQ were the same number.

This completes the 2×2 so data scale and step count can be read separately.

H6: the 4,096-image arm at 3,584 steps was step-limited, so quadrupling its steps raises
final-epoch validation pooled PQ beyond the noise floor.

## Design

Two axes: training-set size {256, 4,096} and optimizer steps {3,584, 14,336}. The two
cells at 3,584 steps already exist from the previous experiment, so two new arms complete
the square:

| Arm | Images | Epochs | Steps | Images seen | Status |
|---|---:|---:|---:|---:|---|
| A | 256 | 112 | 3,584 | 28,672 | existing (`scaleData_256`) |
| C | 4,096 | 7 | 3,584 | 28,672 | existing (`scaleData_4096`) |
| D | 256 | 448 | 14,336 | 114,688 | **new** |
| E | 4,096 | 28 | 14,336 | 114,688 | **new** |

The 4,096-image set is the same selection as arm C (identical seed and prefix rule), so
across the whole square the training set only ever takes two values and each is fixed.

## Frozen protocol

Identical to `archcad_data_scale_20260917.md` in every respect: `ArchCADPrimitiveBatched`
(`dim=64`, `heads=4`, `encoder_layers=1`, `decoder_layers=2`, 149,343 parameters), batched
step protocol with B=8 and length-grouped order, geometry-only four-point features, the
same 128 validation images, eligibility 1..1024 primitives and 1..64 segments, AdamW at
LR 0.0003 with weight decay 0.01 and gradient clip 1, frozen loss (2/7/5, no-object 0.05),
frozen decoder and evaluator, `--deterministic`, seed 1337, selection seed 1337.

Endpoint: the final epoch (14,336 steps for both new arms). Intermediate evaluations are
for curve shape only. One seed; diagnostic, not a replicated finding. No test split read.

## Commands

```powershell
python scripts/research/train_archcad_primitive_batched.py --output-dir outputs/panoptic/scaleData_256x4 --train-images 256 --val-images 128 --epochs 448 --eval-every 112 --batch-size 8 --eval-batch-size 8 --batch-order length-grouped --seed 1337 --selection-seed 1337 --deterministic
python scripts/research/train_archcad_primitive_batched.py --output-dir outputs/panoptic/scaleData_4096x4 --train-images 4096 --val-images 128 --epochs 28 --eval-every 7 --batch-size 8 --eval-batch-size 8 --batch-order length-grouped --seed 1337 --selection-seed 1337 --deterministic
```

## Decision rule fixed before execution

Noise floor on this validation subset: **0.016429**. Reference for the control: the frozen
deterministic 20-epoch control at **0.11165**.

- **Step-limited, data is a live lever** — arm E ends more than the noise floor above arm
  C *and* more than the noise floor above the control. Then more data combined with enough
  steps does move validation, and the next step is to scale training data toward the full
  28,771-image split rather than to touch the representation.
- **Data is not the lever** — arm E lands within the noise floor of arm C. Doubling the
  data 16x and the steps 4x then still does not move validation past the control, and the
  geometry-only representation becomes the primary hypothesis: proceed to raster feature
  sampling and fusion (plan step 5).
- **Arm D (256 images, 14,336 steps)** is the control for the step axis. It is expected to
  overfit harder than arm A; if it instead improves, then steps alone matter and the
  earlier PLATEAU verdict on `scale_dim64_1000` needs reinterpretation.
- Anything else is **INCONCLUSIVE** and is recorded as such.

## Results

Both new arms completed. Final epoch for every cell, batched B=8, deterministic, seed
1337; cells A and C reuse the previous experiment.

| Cell | Images | Steps | Epochs | Val loss | Val pooled PQ | Val recall | Train pooled PQ |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 256 | 3,584 | 112 | 8.1191 | 0.09973 | 0.0825 | 0.2089 |
| C | 4,096 | 3,584 | 7 | 6.8765 | 0.12490 | 0.1224 | 0.1215 |
| D | 256 | 14,336 | 448 | 15.8673 | 0.09118 | 0.0598 | 0.4177 |
| E | 4,096 | 14,336 | 28 | 6.1785 | **0.14486** | **0.2285** | 0.1481 |

Effects, applied mechanically by `outputs/panoptic/_runs_20260917/apply_h6_rule.py`:

| effect | delta |
|---|---:|
| data, at 3,584 steps (C − A) | **+0.025173** |
| data, at 14,336 steps (E − D) | **+0.053686** |
| steps, at 256 images (D − A) | −0.008554 |
| steps, at 4,096 images (E − C) | **+0.019959** |

`E − C = +0.019959` and `E − control = +0.033217`, both above the 0.016429 floor.

**Verdict: STEP-LIMITED, DATA IS A LIVE LEVER.** H6 is supported.

### This reverses the previous experiment's reading

`archcad_data_scale_20260917.md` returned INCONCLUSIVE and reasoned that data scale might
not be the constraint. That reasoning was wrong in an identifiable way: those arms were
matched on steps, so the large-data arm saw each drawing only 7 times, and it had stopped
overfitting without converging. Given enough steps, the same 4,096-image set gives
0.14486 — the first configuration in this whole investigation to beat the frozen control
by more than the noise floor, with thing recall 0.2285 against 0.0973 for the control.

The interaction is the finding, not either main effect: extra steps *hurt* at 256 images
(−0.008554) and *help* at 4,096 (+0.019959). Steps and data are not separable levers, and
any statement of the form "more training does not help" is only meaningful together with
the data scale it was measured at. That is what the earlier PLATEAU verdict got wrong.

### What this implies

The model is still far from a data or optimization ceiling at 4,096 images. The
representation probes remain valid as a forecast of where that frontier ends, but they are
not the binding constraint yet. The cheapest next moves are to push the same direction
(4,096+ images with more passes, then the full 28,771-image split) while keeping the
matched-pair discipline.

### Deviations

- Cells D and E ran while read-only diagnostics and a concurrency benchmark shared the
  machine. Those change wall clock only and were themselves verified to be result-neutral.
- The rule was applied mechanically as preregistered, with no reinterpretation of the
  thresholds after seeing the numbers.

### Artifacts

- `outputs/panoptic/scaleData_{256,1024,4096,256x4,4096x4}`
- Rule applied by `outputs/panoptic/_runs_20260917/apply_h6_rule.py`

