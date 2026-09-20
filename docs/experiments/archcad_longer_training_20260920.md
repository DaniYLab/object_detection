# ArchCAD: longer training at batch 32, and a refit of the grouping radii — preregistered 2026-09-20

## Question

Two levers were live when the grouping work finished, and this is the first run that uses
both at once.

**Exposure for semantics.** The batch-32 curve was still adding quality at its last evaluated
point (`0.30001 → 0.30397 → 0.30580` over epochs 224/252/280) and the batch-8 curve was
adding more per doubling (`+0.0127` in its last interval), so the model is not saturated.
Batch 32 is the cheaper configuration per minute, measured: it reached the same validation
quality as the 112-epoch batch-8 run in 20% less wall clock.

**Grouping radii.** `PER_CLASS_PROXIMITY` was fitted against the predictions of the
`semantic_full_b32_280ep` checkpoint. Better semantics change which primitives carry which
predicted class, so the radii must be refitted whenever the checkpoint changes, and a longer
run is exactly such a change.

H16: a longer run improves validation, and refitting the radii on the new checkpoint adds
further.

## Design

| stage | what | cost |
|---|---|---|
| 1 | train the semantic head, full split, batch 32, 560 epochs | ~100 min |
| 2 | refit per-class proximity radii on the new checkpoint, train split | ~100 min |
| 3 | score on 512 validation images against the current best | minutes |

Stage 1 doubles the epochs of the current best run. Everything else is held: batch 32,
length-grouped, AdamW at LR 0.0003 with weight decay 0.01 and gradient clip 1, FP32,
`--deterministic`, model seed 1337, selection seed 1337, validation 512 images (seed 1338),
geometry-only 12-dimensional input, `--eval-every 28`, `--eval-train-images 2048`.

Stage 2 repeats the two-pass coordinate descent over every class present, with connectivity
fixed at 0.01, which the previous fit showed is not worth tuning per class
(`archcad_per_class_grouping_20260920.md`). The fit objective is the frozen evaluator's pooled
PQ on 1,024 **training** images; validation is touched only in stage 3.

```powershell
python scripts/research/train_archcad_semantic.py --output-dir outputs/panoptic/semantic_full_b32_560ep --train-images -1 --val-images 512 --epochs 560 --eval-every 28 --eval-train-images 2048 --batch-size 32 --eval-batch-size 32 --batch-order length-grouped --seed 1337 --selection-seed 1337 --dim 64 --heads 4 --encoder-layers 2 --deterministic
```

## Decision rule fixed before execution

Resolution: 0.0078. Current best: **0.331524** pooled PQ on 512 validation images, which is
the `b32_280ep` checkpoint scored with the radii fitted against it.

- **Exposure still pays** — stage 3 `> 0.339324`. Then the scale lever continues and the next
  run doubles again; the grouping rule is left as it is.
- **Exposure has flattened** — stage 3 `<= 0.339324` while the train−validation gap stays
  near the current `+0.001543`. Then semantics are limited by something other than exposure at
  this model size, and with appearance, per-primitive shape and capacity all closed, the
  remaining options are a different input representation or a joint model — both of which
  would be new builds rather than more of the same.
- **Overfitting** — stage 3 `<= 0.339324` with the gap widening well beyond `+0.001543`. Then
  the full split cannot absorb 560 epochs at this batch size either, and the same conclusion
  follows as for the flattened case.

Stage 2 is reported either way: the refitted radii and their validation effect say whether the
grouping gain survives a change of checkpoint, which is a claim the previous fit could not
test.

## Results

### Stage 1 — 560 epochs

Completed, 90 min wall clock (4.5 min per 28 epochs). The run is a strict extension of the
280-epoch run, and that is checkable rather than assumed: every evaluated epoch from 28 to 280
is bit-identical between the two runs — train loss, train PQ and validation PQ, 10 of 10
evaluations equal. The two runs share one trajectory and differ only in where they stop. This
is the second independent confirmation of the determinism fix, after the seed-1337 control.

Train and validation PQ under the trainer's own protocol (uniform proximity 0.1, which is what
the trainer used before 2026-09-20):

| epoch | train PQ | val PQ | gap | val SQ | val accuracy |
|---|---|---|---|---|---|
| 28 | 0.220830 | 0.222942 | −0.002112 | 0.813960 | 0.4298 |
| 140 | 0.289963 | 0.286528 | +0.003436 | 0.836207 | 0.5385 |
| 280 | 0.307343 | 0.305800 | +0.001543 | 0.839337 | 0.5727 |
| 420 | 0.317148 | 0.316291 | +0.000858 | 0.837957 | 0.5898 |
| 560 | 0.324629 | 0.316707 | +0.007922 | 0.842929 | 0.5944 |

Validation gained `+0.010491` between 280 and 420 and `+0.000417` between 420 and 560. The last
interval is below the 0.0078 resolution: under the trainer's protocol the curve had already
flattened 140 epochs before the run ended, while the train−validation gap widened fivefold.

### Stage 2 — refit on the new checkpoint

Completed, 5,247 s. The refitted table differs from the adopted one in exactly two classes
(sink 0.10 → 0.05, steel_column 0.10 → 0.05); every other class kept its value. On the 512
validation images:

| weights | pooled PQ | thing PQ |
|---|---|---|
| uniform proximity 0.1 | 0.316707 | 0.310516 |
| the adopted table | 0.343407 | 0.346588 |
| refitted on this checkpoint | 0.343485 | 0.346674 |
| difference | +0.000078 | +0.000087 |

The refit is 1% of the resolution. The preregistration's premise — that "the radii must be
refitted whenever the checkpoint changes" — is not supported by this test: the adopted table
transfers to a new checkpoint essentially unchanged. The second reading is that the coordinate
descent sits on a flat optimum, which is the same conclusion `archcad_per_class_grouping_20260920.md`
reached from the other direction.

### Stage 3 — the preregistered decision

Stage 3 is **0.343485** against the gate of 0.339324 (previous best 0.331524 plus the 0.0078
resolution). The rule returns **exposure still pays**. New best: pooled PQ 0.343485, thing PQ
0.346674, per-primitive accuracy 0.5944, `+0.011961` over the previous best.

### Stage 4 — diagnostics added after the rule was applied, and not part of it

The verdict cleared the gate by 0.004161, less than one resolution, so the intermediate
checkpoints were scored under the adopted protocol to see whether the gain is a trend or an
endpoint. Every 28-epoch checkpoint, with the same checkpoint also scored using the radii fitted
on the 560-epoch endpoint:

| epoch | 280 | 308 | 336 | 364 | 392 | 420 | 448 | 476 | 504 | 532 | 560 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| adopted PQ | 0.331524 | 0.331997 | 0.327459 | 0.333419 | 0.338721 | 0.341145 | 0.331562 | 0.330046 | 0.337471 | 0.332624 | 0.343407 |
| fitted-on-560 PQ | 0.331202 | 0.331704 | 0.327360 | 0.333703 | 0.338749 | 0.341276 | 0.332045 | 0.330616 | 0.337888 | 0.333604 | 0.343485 |

The two columns agree within 0.001 at every one of the eleven epochs — the transfer result
again, now measured eleven times.

The mean over 280–364 is 0.331100 against 0.336425 over 392–560: the second half is better by
`+0.005325`, below the resolution, while the spread *within* the second half is 0.013361. The
doubling's headline `+0.011961` therefore rests on the endpoint being the highest point of a
noisy plateau, not on a trend that another doubling would continue.

Two cheap standard fixes for exactly that pathology were then measured, both scored under the
adopted protocol on the same 512 validation images:

| weights | pooled PQ |
|---|---|
| epoch 560 alone | 0.343407 |
| average of the last 2 / 3 / 4 / 7 checkpoints | 0.336993 / 0.339439 / 0.338742 / 0.340165 |
| average of the 3 best checkpoints chosen by train PQ | 0.344257 |

Weight averaging does not beat the endpoint by more than 0.00085, 11% of the resolution. The
one number that looked better — 0.345816 for the three best checkpoints chosen by *validation* —
does not survive honest selection on train, and is reported here only as the reason the
train-selected version was run.

### Appearance: closed by measurement, not by argument

The milestone table recorded appearance as closed because every entity's `rgb` is `[0,0,0]`.
That closes the annotation's colour channel, not the other two candidates, so both were
measured on this release.

**The rendered raster is monochrome.** Over 64 validation drawings: 16 quantised colours and
every one of them has `r = g = b`; 96.5% of pixels are background, ink is 3.55% on average
(range 0.85–8.44%); all drawings are 980×980. No colour channel exists.

**The annotation's styling fields carry no class signal.** Over 256 validation drawings and
74,688 entities: `line_width` is 0.25 for 74,129 of them (99.2%), with 0.18 for 462, 0.35 for
95 and 0.09 for 2; `linetype` is `Continuous` (68,335) or `CONTINUOUS` (6,308) — a casing
difference — plus 45 entities carrying layer names. Per class, the modal width is 0.25 at share
1.00 for every class with at least 200 primitives. Of 6,701 identical-geometry groups, 181 have
conflicting labels, and **0** of those 181 are separated by `line_width`, by `linetype`, or by
both together.

**The raster's only candidate channel is line thickness, and it is a rendering of the same
vectors.** The coverage-based width estimate over 14,497 primitives in 48 drawings has mean
2.263 px and sd 1.168; the between-class spread of class means is 0.530 px against a mean
within-class sd of 0.845 px. The estimator cannot separate stroke width from local ink density
(hatching and parallel lines inside the window both raise it), and whatever it does measure is
produced by the same entities the model already receives, so it cannot separate primitives whose
geometry is identical — which is exactly the ambiguity the audit recorded. The like-for-like
nearest-centroid comparison attempted for this doc was uninformative and is not reported as
evidence: on raw position-dominated tokens it scored 0.0393 for geometry and 0.0268 for width
against a 0.1914 majority baseline, i.e. the rule is worse than the baseline for both inputs and
says nothing about either.

Conclusion: no raster branch is justified on this release. The appearance channel is empty, and
the two places it could have paid — breaking identical-geometry ties and adding colour — do not
exist here.

### Verdict, and the departure from the rule's prescription

The preregistered rule fired "exposure still pays" and prescribes doubling again to 1,120
epochs. That prescription is **not** being followed, and the reason is the stage-4 evidence
above: the last 140 epochs of the doubling bought `+0.0004` under the trainer's protocol, the
adopted-protocol trend across the last 168 epochs is `+0.0053` (below resolution), the endpoint
gain is a plateau maximum, and weight averaging — the standard remedy — does not help. A further
doubling would cost ~3 hours to buy less than the resolution.

This is a deliberate departure, recorded rather than taken silently, and it does not change the
H16 verdict: exposure still pays at the gate. The doubling stays available as a fallback if the
alternative fails. The alternative targets the two symptoms the rule was not written to
interpret — endpoint oscillation and the widening gap — and is preregistered as H17 in
`archcad_annealing_capacity_20260920.md`: a 140-epoch screen of learning-rate annealing against
a larger head, gated on the control's 0.310439.
