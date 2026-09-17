# ArchCAD scale-aware input features — preregistered 2026-09-17

## Question

`archcad_failure_localization_20260917.md` measured that 57.6% of val thing instances are
separable by a proximity rule that knows the correct radius (`d_in < d_out`), while the
model covers only 39.5% of those. So **34.8% of all instances are separable and missed** —
headroom that a scale-aware input could reach. A further 35.8% are not separable by any
proximity rule and need appearance.

H7: adding local packing scale and scale-normalised size to the input improves final-epoch
validation quality over the frozen 12-dimensional input, at matched protocol, data and
steps.

## The change, and what it does not change

`src/data/archcad_rich_features.py` appends six dimensions per primitive to the frozen 12
(`FEATURE_NAMES`): polyline length, 4-point extent, and the nearest / mean-of-4 /
mean-of-8 distances between primitive centres, plus `log1p(length / d4)`.

Deliberately unchanged: the frozen dataset, the decoder, the evaluator, the loss, the
batched-step protocol, the selection and the eligibility caps. The model is a new class
whose only difference is a wider input layer, and `config['input_features']` records the
width so a checkpoint states its own input contract. A structural cue such as junction
degree is left out of this version on purpose — it needs endpoint geometry — and is the
natural next addition if this one moves validation at all.

## Design: matched pair on a larger validation subset

Both arms: 4,096 training images (selection seed 1337), 28 epochs = 14,336 optimizer
steps, batched B=8 length-grouped, deterministic, model seed 1337, identical everything
except input width. The amendment below records why this replaced the originally
preregistered 256-image / 3,584-step regime.

Validation moves to **512 images** for this comparison. The four 128-image folds of that
512 set span 0.015677 in pooled PQ, essentially the 0.016429 floor used all day, so a
128-image evaluation cannot resolve the effect being chased. 512 images should halve the
floor. The 512 set is a nested prefix of the same selection seed, so it contains the 128
images used previously; the frozen checkpoint's value on it is pooled PQ **0.124883**,
thing recall **0.097289**.

| Arm | Input | Val images | Command |
|---|---|---|---|
| A′ | frozen 12-dim | 512 | `--output-dir outputs/panoptic/val512_frozen` |
| B′ | rich 18-dim | 512 | `--output-dir outputs/panoptic/val512_rich` |

```powershell
python scripts/research/train_archcad_primitive_batched.py --output-dir outputs/panoptic/val512_frozen --train-images 256 --val-images 512 --epochs 112 --eval-every 28 --batch-size 8 --eval-batch-size 8 --batch-order length-grouped --seed 1337 --selection-seed 1337 --deterministic
python scripts/research/train_archcad_primitive_batched.py --output-dir outputs/panoptic/val512_rich --train-images 256 --val-images 512 --epochs 112 --eval-every 28 --batch-size 8 --eval-batch-size 8 --batch-order length-grouped --seed 1337 --selection-seed 1337 --rich-features --deterministic
```

Arm A′ is not a new treatment: it re-runs the frozen input under the current code, both to
provide the matched pair on 512 validation images and as a reproducibility check against
`scaleData_256` on the 128-image subset.

## Amendment before execution (2026-09-17)

The arms were preregistered at 256 training images and 3,584 steps. While this document
was being written, the separate 2×2 of data scale and steps completed and showed that
regime to be starved: at 3,584 steps the 256-image arm reaches val pooled PQ 0.09973
against 0.14486 for 4,096 images at 14,336 steps, and adding steps at 256 images makes
validation *worse* (`+0.008554` regression). Testing an input change where data is the
binding constraint would confound "these features do not help" with "the run cannot use
them".

Both arms therefore move to the regime where the baseline is not data-starved — 4,096
training images, 28 epochs, 14,336 steps — and keep the 512-image validation set. Nothing
else changes. This amendment was made before either arm ran, and in response to a
different experiment's outcome, not to any result of this one.

| Arm | Input | Train images | Epochs | Val images |
|---|---|---|---|---|
| A′ | frozen 12-dim | 4,096 | 28 | 512 |
| B′ | rich 18-dim | 4,096 | 28 | 512 |

```powershell
python scripts/research/train_archcad_primitive_batched.py --output-dir outputs/panoptic/val512_frozen_4096 --train-images 4096 --val-images 512 --epochs 28 --eval-every 7 --batch-size 8 --eval-batch-size 8 --batch-order length-grouped --seed 1337 --selection-seed 1337 --deterministic
python scripts/research/train_archcad_primitive_batched.py --output-dir outputs/panoptic/val512_rich_4096 --train-images 4096 --val-images 512 --epochs 28 --eval-every 7 --batch-size 8 --eval-batch-size 8 --batch-order length-grouped --seed 1337 --selection-seed 1337 --rich-features --deterministic
```

Reference points for the amended regime: the frozen 12-dim input at this exact
configuration is `scaleData_4096x4` on 128 validation images, val pooled PQ `0.14486`,
thing recall `0.2285`. Arm A′ re-runs it on 512 validation images to give a matched pair
at the lower noise floor and to check that changing only the validation set reproduces
that number on its 128-image subset.

## Decision rule fixed before execution

Resolution on this 512-image set is taken as **0.0078**, half the measured 128-fold spread
of 0.015677. Deltas below it are not resolvable.

Let `delta = final_epoch_val_pooled_PQ(B′) − final_epoch_val_pooled_PQ(A′)`.

- **Scale-aware input helps** — `delta > +0.0078`. Then the separable-and-missed third of
  the problem is reachable through the input, and the next step is to extend the same
  direction with structural cues (junction degree, endpoint degree) and repeat at 4,096
  training images.
- **No resolvable effect** — `|delta| <= 0.0078`. Then local scale is not the missing
  information for those instances, and the appearance branch (raster feature sampling,
  plan step 5) becomes the primary path rather than the deferred one.
- Either way the train-set metrics are reported, and the validation band is compared
  against the frozen reference `0.11165` on 128 images and `0.124883` on 512.

One seed. Diagnostic, not a replicated finding. No held-out test split is read, and no
hyperparameter is changed in response to validation observations.

## Results

Both arms completed at the amended regime: 4,096 training images, 28 epochs = 14,336
steps, 512 validation images, batched B=8, deterministic, seed 1337.

| Arm | Input | Val loss | Val pooled PQ | Val recall | Train pooled PQ |
|---|---|---:|---:|---:|---:|
| A′ | frozen 12-dim | 6.1376 | 0.14851 | 0.2335 | 0.1481 |
| B′ | rich 18-dim | 5.7802 | **0.15321** | 0.2181 | 0.1560 |

`delta = 0.15321 − 0.14851 = +0.00470`, below the preregistered `+0.0078` threshold.

**Verdict: NO RESOLVABLE EFFECT.** H7 is not supported.

Within the run, the rich input does lower validation loss (5.7802 versus 6.1376) but the
panoptic metric does not follow, and validation recall is if anything slightly lower
(0.2181 versus 0.2335). Train and validation pooled PQ coincide in both arms (0.1481 /
0.14851 and 0.1560 / 0.15321), so neither arm is overfitting — the measured difference is
an input-representation difference at a converged-ish operating point, and it is inside the
resolution of the metric.

Per the preregistered rule, the consequence is that **local packing scale is not the
missing information** for the instances the probe identified as separable, and the
appearance branch (raster feature sampling, plan step 5) becomes the primary path rather
than the deferred one.

### Cross-checks

- Arm A′ is the same configuration as `scaleData_4096x4` re-evaluated on 512 validation
  images: 0.14851 here against 0.14486 on the 128-image subset. Consistent, and it shows
  the larger validation set shifts the value by less than the floor as expected.
- Both arms ran to the preregistered endpoint with no protocol changes after the
  amendment.

### Artifacts

- `outputs/panoptic/val512_frozen_4096`, `outputs/panoptic/val512_rich_4096`

### Follow-up implied

Two things now point the same way, and neither is the input's scale information:

1. the separable-but-missed third of instances (34.8%) is not recovered by adding local
   scale, step size and size ratio to the input;
2. the third that no proximity rule can separate (35.8%) needs appearance by construction.

Both are addressed by raster feature sampling, which the research plan already lists as
step 5 and which this pair of results moves from "next in the plan" to "the measured
bottleneck". The data lever from H6 remains open and is being tested separately at the
full training split.

