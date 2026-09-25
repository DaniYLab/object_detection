# ArchCAD: an additive grouping mechanism — preregistered 2026-09-20

## Question

The grouping prize is the largest unclaimed lever in the pipeline: with the 560-epoch semantics,
oracle grouping scores 0.582606 against 0.343407 for the adopted rule on the same 512 validation
images (`archcad_learned_grouping_20260920.md`).

Two mechanisms have now failed against it. The fixed rule is exhausted — its radii are already
fitted on PQ, and refitting them on a new checkpoint moves pooled PQ by 0.000078
(`archcad_longer_training_20260920.md`). A learned edge classifier that prunes the fixed rule's
candidates fails whether its threshold is fitted on edge F1 (0.291465 against 0.304004) or on
PQ itself (0.335758 against 0.343407), and the PQ-fitted sweep is monotone, so that family's
best possible outcome is the rule it prunes (`archcad_grouping_threshold_20260920.md`).

What both share is that they can only *remove* links. The failure anatomy says the dominant error
is over-merging — 58.7% of ground-truth instances are swallowed by a component that spans more
than one instance, against 0.9% fragmented — and pairwise features cannot see the difference
between a link inside one symbol and a link between two adjacent same-class symbols, because
adjacent symbols of a class share exactly the geometry those features describe.

H19: predicting **where each instance is** separates adjacent same-class symbols where deciding
which links to keep cannot.

## Mechanism

For every primitive of a thing class, predict the offset from that primitive's centre to the
centre of its ground-truth instance, in normalised drawing coordinates. Instances are then
recovered by clustering: primitives of the same predicted class whose *predicted* centres agree
within a tolerance τ are one instance, and each stuff class stays one segment as it is today.

The mechanism is additive in the two ways the pruning rule could not be:

- two primitives far apart can share a predicted centre, so an instance fragmented by the
  radius can be rejoined;
- two adjacent symbols have different centres, so a merge can be undone even when the two
  symbols share endpoints and are closer than any radius that keeps their own primitives
  together.

τ is a single scalar fitted on **train PQ** by coordinate descent, the same objective that
produced the per-class radii.

## Design, in two steps, so the expensive one is only run if the cheap one passes

**Step 1 — the mechanism's ceiling (no training).** Feed the clustering rule the *ground-truth*
instance centres, fit τ on train PQ, and score validation. This measures the rule itself, apart
from any prediction error: if the clustering cannot recover the prize even with perfect centres,
the mechanism is wrong and step 2 is not worth running. Cost: minutes.

**Step 2 — is the offset learnable from the frozen representation?** Cache the 560-epoch
checkpoint's encoder features for a 4,096-drawing training subset and 512 validation drawings,
train a small MLP head on the offset target, and measure the quantity that decides the mechanism:
for same-class primitive pairs, how well do predicted-centre distances separate same-instance
pairs from different-instance pairs (AUC), against the same measure for raw centre distance —
which is what the adopted rule uses. Cost: minutes, no encoder training.

**Step 3, only if step 2 separates better than proximity — the full build.** Train the offset head
jointly with the semantic head (the offset loss added to the length-weighted cross entropy) on
the full split at the frozen protocol, 140 epochs, and score end to end through the frozen
evaluator with τ fitted on train PQ. Cost: ~25–40 min.

## Decision rules fixed before execution

Resolution: 0.0078. Adopted rule on the 512 validation images: **0.343407**, so the pipeline gate
is **0.351207**. Proximity baseline for step 2: AUC of raw centre distance on same-class pairs.

- **Step 1 fails** — the clustering rule with perfect centres scores `<= 0.351207`. Then the
  mechanism is wrong and it is dropped; the next idea is a different additive form.
- **Step 1 passes, step 2 fails** — predicted-centre AUC `<=` proximity AUC. Then the frozen
  representation does not carry the instance structure the mechanism needs, and the conclusion is
  that instance grouping requires training the representation for it, which is the full build
  (step 3) run blind. That is a different, larger commitment and gets its own preregistration.
- **Steps 1 and 2 pass, step 3 fails** — end-to-end validation `<= 0.351207` after joint
  training. Then the mechanism works in principle but not at this model size or budget, and the
  grouping lever is recorded as closed for this architecture.
- **Step 3 passes** — validation `> 0.351207`. The grouping build continues: more fit data,
  per-class τ, and then a long run.

## Results

### Step 1 — the clustering rule is not the limitation

With ground-truth segment centres as the "predictions", the rule scores **0.986148** pooled PQ
(thing 0.981534) on the 512 validation images at τ = 0.005, against 0.343407 for the adopted rule
and 0.582606 for oracle grouping. The sweep over τ on train PQ is flat and high at the small end
(0.984392 at 0.005, 0.981612 at 0.02, 0.979625 at 0.03) and degrades only when τ starts merging
distinct instances (0.964842 at 0.05, 0.771740 at 0.2).

Two notes on reading this. The number exceeds the oracle-grouping bound because it also uses
ground-truth semantics, so it is not comparable to 0.582606 — it measures the *rule*, not the
pipeline. And the gate written into the preregistration (0.351207) was far too weak for a test
with perfect centres; it is reported as passed, but the informative result is the size of the
number: the rule recovers essentially the whole ground truth when its centres are right, so any
loss under this mechanism is prediction error, not clustering error.

### Step 2 — the frozen representation does not carry instance centres

A 128×128 MLP on the frozen encoder's per-primitive hidden state plus the primitive's own centre,
trained for 30 epochs on 2,048 training drawings (453,659 rows, 150,977 of them supervised), with
a length-agnostic smooth-L1 loss on the offset to the ground-truth instance centre. The training
loss fell from 0.024016 to 0.002018, which is not informative on its own: many instances are
single-primitive, so their target offset is exactly zero and a position prior already scores well.

The quantity that decides the mechanism, measured on 199,138 same-class validation primitive pairs
(33.65% of them same-instance):

| pair separation by centre distance | AUC |
|---|---|
| raw centre distance (what the adopted rule uses) | 0.6893 |
| predicted centre distance (this head) | **0.6901** |
| ground-truth centre distance (the ceiling) | 1.0000 |

The head moves the AUC by **+0.0008**. Its median offset error is 0.03759 in normalised drawing
units — about 37 pixels on a 980-pixel drawing, larger than the symbols it is supposed to
localise. In other words it learned a position prior, not instance centres.

The preregistered criterion for this step was "predicted AUC `<=` proximity AUC". Taken literally
the head clears it by 0.0008, which is not a margin any build should rest on, so the step is
recorded as a **failure by any useful margin** and the ambiguity in the criterion is a
preregistration defect: an AUC criterion needs a margin, not a sign.

### Verdict, and what it does and does not say

Step 2's outcome is the preregistration's second branch: the frozen representation does not carry
the instance structure the mechanism needs. It does **not** say the mechanism is wrong — step 1
showed the opposite — it says that recovering it requires training the representation for
grouping rather than adding a head to a representation trained for semantics. That is step 3, the
full build, which the preregistration reserved for a separate commitment and which is not run
here.

Two things are worth recording for whoever runs step 3. The cheap probe that would decide it is
not another frozen-head variant: it is a short joint run (the offset loss added to the semantic
loss) re-measuring this same AUC, because the question is precisely whether the encoder can be
pushed to carry instance structure, and only joint training can answer it. And the criterion for
that run should require a margin — the measured proximity baseline is 0.6893, so a joint run that
lands below ~0.75 has not produced a usable signal.

### Step 3 — run, and it also fails

That cheap joint probe was run rather than the full build: the same encoder plus the offset head,
initialised from the 560-epoch weights, trained for 28 epochs on 2,048 training drawings with
`semantic CE + 20 × smooth-L1 offset` (`h19_step3_joint_probe.py`). Gates fixed before running:
AUC at least 0.75 for a usable signal, end-to-end val PQ above 0.351207 for a working mechanism.

| quantity, 512 validation images | value |
|---|---|
| pair separation AUC, raw centre distance | 0.6738 |
| pair separation AUC, jointly trained predicted centres | **0.6732** |
| pair separation AUC, ground-truth centres | 1.0000 |
| semantic accuracy after joint training | 0.5995 (560-epoch model: 0.5944) |
| end-to-end pooled PQ, τ fitted on train PQ | 0.329560 (adopted rule 0.343407) |

Both gates fail. Joint training did not damage the semantics — accuracy rose slightly — but it did
not move the pair separation at all, in either direction (−0.0006), and the pipeline is worse than
the rule it would replace. The τ sweep on train PQ (0.083652 at 0.005 rising to 0.417965 at 0.08)
is still climbing at the grid edge, which is the same defect the H18 sweep had: the fit did not
bracket its optimum, because centres this coarse can only be merged aggressively. The direction of
that sweep is the diagnosis — the learned centres are too noisy to separate anything, so the best
use of them is to merge.

**What this does not settle.** The probe is 28 epochs on 2,048 of 27,272 drawings with a 70K
parameter model, so it rules out "a small head on this representation, briefly trained" rather
than the mechanism in general. The residual possibility is that instance structure needs the
model scaled with the budget and the full split, which is the same commitment the capacity arm of
H17 tests for semantics. What it does establish is that the offset target is not learnable for
free: the loss curve is uninformative (0.00653 → 0.00200 in seven epochs, because most
single-primitive instances have an exactly zero offset), and the metric that matters does not
move.

So the grouping lever stands as it did: a +0.239198 prize, three mechanisms tried against it
(radii exhausted, pruning refuted, instance centres without signal), and no measured way to claim
it at this scale.

## Artifacts

- `outputs/panoptic/_runs_20260917/h19_step1_ceiling.py`, `outputs/panoptic/h19_step1_ceiling.json`
- `outputs/panoptic/_runs_20260917/h19_step2_learnability.py`, `outputs/panoptic/h19_step2_learnability.json`
