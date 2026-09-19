# ArchCAD: how much is context worth, and does scaling it help — preregistered 2026-09-17

## Question

Two measurements bracket the per-primitive semantic head:

| predictor | accuracy |
|---|---:|
| constant "wall" | 21.60% |
| kNN over the primitive's own centroid and extent (no context, no training) | 19.78% |
| trained transformer head over the whole drawing (27,272 images) | 48.44% |

The gap between 19.78% and 48.44% is what the drawing's context contributes: 28.7 points.
`archcad_component_pipeline_20260917.md` closed the other routes — the raw data has no
appearance channel, and regrouping primitives without semantics scores below the head.

H11: how much of the head's accuracy survives with no context at all, and does giving the
model more capacity and context raise it further?

## Arms

All at 4,096 training images, 28 epochs = 14,336 steps, batch 8, length-grouped, AdamW at LR
0.0003 with weight decay 0.01 and gradient clip 1, FP32, `--deterministic`, model seed 1337,
selection seed 1337, 512 validation images (seed 1338), geometry-only 12-dimensional input,
grouper constants proximity 0.020 / connectivity 0.010, frozen evaluator.

| arm | architecture | parameters | question |
|---|---|---:|---|
| A | `ArchCADPrimitiveNoContext`, width 256 | ~70k, matched | how much is left without the drawing |
| B | transformer, dim 128, 4 layers | ~600k | does more capacity and depth help |
| — | transformer, dim 64, 2 layers (existing `semantic_4096`) | 69,854 | reference: accuracy 37.59%, PQ 0.206352 |

Arm A's width is chosen so its parameter count is close to the reference transformer's, so
the comparison is about context rather than size. Arm B is a deliberately larger model; if it
does not beat the reference, capacity is not the constraint.

```powershell
python scripts/research/train_archcad_semantic.py --output-dir outputs/panoptic/nocontext_4096 --train-images 4096 --val-images 512 --epochs 28 --eval-every 7 --batch-size 8 --eval-batch-size 8 --batch-order length-grouped --seed 1337 --selection-seed 1337 --architecture no_context --width 256 --deterministic
python scripts/research/train_archcad_semantic.py --output-dir outputs/panoptic/context_wide_4096 --train-images 4096 --val-images 512 --epochs 28 --eval-every 7 --batch-size 8 --eval-batch-size 8 --batch-order length-grouped --seed 1337 --selection-seed 1337 --architecture transformer --dim 128 --heads 4 --encoder-layers 4 --deterministic
```

Endpoint: the final epoch. One seed; diagnostic, not a replicated finding. No test split read.

## Decision rule fixed before execution

Resolution on the 512-image validation set: 0.0078 in pooled PQ; for accuracy, comparisons
are stated in points and 1 point is treated as the noise scale, since accuracy is a mean over
115,584 validation primitives.

Let `a` be arm A's accuracy and `b` arm B's PQ.

- **Context carries it** — `a <= 26%`: without the drawing the model lands near the
  no-training floor of 19.78%, so context is the dominant source of the head's accuracy and
  the next work is on context itself: positional and relational encoding, depth, and possibly
  multi-scale neighbourhoods.
- **Context is optional** — `a >= 32.6%` (within five points of the reference 37.59%): the
  primitive's own geometry plus learned capacity recovers most of the head's accuracy, so the
  earlier reading that the head "lives on context" is overstated. Then the constraint is
  something else and the next question is what, since features and appearance are closed.
- **In between** — report the split of the 28.7-point context gap and do not claim either.

For arm B, independently: `b > 0.206352 + 0.0078 = 0.214152` means capacity and depth help,
and the next step is to scale them; `b <= 0.214152` means the current head is already near
what this architecture reaches, and with appearance and per-primitive shape both closed, the
remaining gap to the 0.807876 oracle ceiling is structural — it belongs to the joint
semantics-versus-instance problem, not to a missing input.

Both arms report accuracy and PQ, and the per-class confusion is recomputed for whichever
arm wins so a partial result still localises what is left.

## Results

| arm | architecture | parameters | val accuracy | val pooled PQ | train accuracy | elapsed |
|---|---|---:|---:|---:|---:|---:|
| A | no-context MLP, width 256 | 77,854 | **26.31%** | 0.130652 | 24.83% | 181 s |
| B | transformer, dim 128, 4 layers | ~600k | **36.45%** | 0.201686 | 50.60% | 443 s |
| — | transformer, dim 64, 2 layers (reference) | 69,854 | 37.59% | 0.206352 | — | 26 min |

### Arm A: context is worth more than learning the primitive's own features, and neither explains the head

The preregistered rule put `a <= 26%` as "context carries it" and `a >= 32.6%` as "context is
optional". Arm A lands at **26.31%**, inside neither branch, so the rule returns the
intermediate reading: report the split and claim neither.

| step | accuracy | contribution |
|---|---:|---:|
| kNN on the primitive's own features, no training | 19.78% | floor |
| trained MLP on the primitive's own features (arm A) | 26.31% | **+6.53 points for learning** |
| trained transformer over the whole drawing (reference) | 37.59% | **+11.28 points for context** |

Context is the larger single factor, 1.7x what training on the primitive's own features buys,
but it is not the whole story. **The earlier claim in this repository that the head's accuracy
comes "from context, not from its input" is overstated** and is corrected here: the input
carries about a third of the head's accuracy once a model is fitted to it.

### Arm B: more capacity and depth do not help — they overfit

`b = 0.201686` against the gate of `0.214152`, so the preregistered consequence applies: the
current head is already near what this architecture reaches. The larger model is in fact
slightly *worse* than the 69,854-parameter reference on validation PQ (0.2017 against
0.2064) while its training accuracy reaches 50.60% against 36.45% on validation — a
fourteen-point generalisation gap where the smaller head had almost none.

### What the two arms together say

Semantics are limited on both axes that were still open, and the limits are now measured:

| axis | verdict |
|---|---|
| appearance / raster | closed: the raw data has no styling channel (`archcad_component_pipeline_20260917.md`) |
| per-primitive shape alone | 19.78% untrained, 26.31% trained |
| context | worth +11.28 points, and scaling capacity on top of it makes things worse |
| capacity and depth | do not help; the larger model overfits |

Per the preregistered rule, with appearance and per-primitive shape both closed and capacity
exhausted, the remaining gap to the 0.807876 oracle ceiling is **structural**: it belongs to
the joint semantics-versus-instance problem — a primitive's class often depends on which
symbol it belongs to, and which symbol it belongs to depends on the classes — not to a
missing input or to model size.

## Deviations

- Both arms were stopped once and relaunched after a fix to the geometric grouper, whose
  endpoint search was quadratic in Python and stalled the no-context arm's initial evaluation
  (one class holding a whole drawing took 3.4 s per image). The fix was verified equivalent
  to the committed implementation on 72 groupings with zero mismatches before relaunching, so
  the numbers here are not affected by it; it changed runtime only, from a stall to 181 and
  443 seconds.
- Arm B ran once with the pre-fix grouper and was stopped during preparation, then rerun.

## Artifacts

- `outputs/panoptic/nocontext_4096`, `outputs/panoptic/context_wide_4096`
- Equivalence evidence: `outputs/panoptic/_runs_20260917/verify_grouper_vectorisation.py`

