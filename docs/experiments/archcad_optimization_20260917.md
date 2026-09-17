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

Recorded after completion. The preregistered endpoints were evaluated first, in the
order written above; the exploratory work in section E was started only after both
preregistered endpoints were recorded and is labelled exploratory throughout.

### Control check: FAILED — the 2026-09-16 baseline is not bit-reproducible

The control rule required an epoch-20 rerun to reproduce the saved seed-1337 result
within 1e-6 scalar metric error. Both reruns failed it.

| Comparison | Val thing recall | Pooled PQ | Val loss |
|---|---:|---:|---:|
| Saved `primitive_256_s1337` (2026-09-16, nondeterministic) | 0.096102 | 0.119358 | 7.229416 |
| Preregistered budget run at epoch 20 (nondeterministic) | 0.107888 | 0.116262 | 7.281432 |
| `det_control_a` (deterministic) | 0.110607 | 0.111647 | 7.182919 |
| `det_control_b` (deterministic, independent rerun) | 0.110607 | 0.111647 | 7.182919 |

Data and code identity were verified before comparing: identical selected 256 train /
128 val image IDs and source hashes, identical split fingerprint
`b0f01e46…5118c`, identical data/model/evaluator implementation hashes, same device,
torch version and seed. Epoch-0 (no gradient updates) metrics were bit-identical
between the two nondeterministic runs — train loss `14.884698033332825`, `fp=503`,
`fn=2780`, semantic macro F1 `0.000942101071330067` — so selection, initialization and
evaluation are reproducible and the divergence is introduced by training.

Cause: the training path was not deterministic. `cudnn.benchmark=False` was set, but
`torch.use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG` and the
flash/mem-efficient attention backwards were uncontrolled. The `--deterministic` flag
now sets all of them (deviation D1).

With determinism enabled:

- **A1 — reproducible:** `det_control_a` and `det_control_b` agree on every reported
  metric with delta `0.000e+00` (bit-identical, not merely within 1e-6). Epochs 1–20 of
  the separate 100-epoch run also match both.
- **A2 — still fails against the historical run:** thing recall `+0.014506`, pooled PQ
  `−0.007711`, val loss `−0.046496`. A nondeterministic run cannot be retro-fitted to a
  reproducible trajectory, so the saved 2026-09-16 result is not bit-reproducible.
- **A3 — spread across three identical-protocol seed-1337 runs** (two nondeterministic,
  one deterministic, all measured at epoch 20): thing recall `0.014506`, pooled PQ
  `0.007711`, semantic macro F1 `0.018334`. The thing-recall range alone is 48% of the
  H2 gate's `+0.03` requirement, so a single nondeterministic pair cannot resolve it.

Chaos amplification, on the 8-image overfit set: the deterministic and nondeterministic
runs differ in epoch-1 loss at the 6th significant digit (`13.344882` vs `13.344877`)
and end 200 epochs apart at train thing recall `0.3529` vs `0.4706`.

### H1 — overfit: FAILED

| Endpoint (epoch 200, 8 train images, dim 64) | Train thing recall | Train pooled PQ | Gate |
|---|---:|---:|---|
| Preregistered run (nondeterministic, `overfit8_20260917`) | 0.4706 | 0.4404 | needs 0.95 / 0.90 |
| Same budget, deterministic (`cap_dim64_200`) | 0.3529 | 0.4145 | needs 0.95 / 0.90 |

Both fail. Per-epoch training loss fell `13.34 → 2.94` without divergence, so this is
not an instability. Two configurations of the same budget reach visibly different
endpoints, which is the nondeterminism above rather than a protocol difference.

The failure is budget-limited, not architectural: E1 extends the same configuration to
3000 epochs and the recall half of the gate is met from epoch 2000 onward (0.9608), with
semantic macro F1 0.9313 and instance AP50 0.9729 on the same 8 images. Only the
PQ ≥ 0.90 half is unmet at 3000 epochs.

### H2 — budget effect: FAILED

| Endpoint | Val thing recall | Pooled PQ |
|---|---:|---:|
| Preregistered (nondeterministic): epoch 20 → epoch 100 | 0.107888 → 0.087035 (**−0.020853**) | 0.116262 → 0.095550 (**−0.020712**) |
| Deterministic rerun: epoch 20 → epoch 100 | 0.110607 → 0.114234 (**+0.003626**) | 0.111647 → 0.109054 (**−0.002593**) |

The gate requires `+0.03` with no pooled-PQ regression; both runs fail both parts. The
deterministic run overfits rather than underfits: training loss `9.7506 → 3.3105` while
validation loss rises monotonically after epoch 20 (`7.182919 → 7.467253 → 7.788266 →
8.805120 → 10.030336`). Per-epoch validation variation is larger than the endpoint
deltas — pooled PQ across the evaluated epochs spans `0.102589–0.119018`.

Neither preregistered hypothesis is supported. Per the preregistration, the failure of
H1 only means the setup did not overfit within the fixed budget, and H2 on one seed was
already declared diagnostic rather than a replicated result.

### E. Exploratory work (not preregistered)

**E1 — capacity and budget sweep, 8 train images, deterministic, seed 1337.** Same
selection and decoder; only `dim`/`encoder_layers`/`decoder_layers`/`epochs` change.

| Run | Params | Epochs | Train recall | Train PQ | Train F1 | RQ | SQ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cap_dim64_200` | 149,343 | 200 | 0.3529 | 0.4145 | 0.5856 | 0.450 | 0.865 |
| `cap_dim64_600` | 149,343 | 600 | 0.7255 | 0.6902 | 0.7487 | 0.796 | 0.907 |
| `cap_dim64_1200` | 149,343 | 1200 | 0.8824 | 0.7424 | 0.8397 | 0.900 | 0.875 |
| `cap_dim64_3000` | 149,343 | 3000 | **0.9608** | 0.8706 | 0.9313 | 0.961 | 0.927 |
| `cap_dim128_200` | 1,107,231 | 200 | 0.6275 | 0.5885 | 0.7502 | 0.696 | 0.888 |
| `cap_dim128_600` | 1,107,231 | 600 | 0.8431 | 0.7581 | 0.8469 | 0.869 | 0.912 |
| `cap_dim256_200` | 4,377,119 | 200 | 0.5882 | 0.5231 | 0.6888 | 0.606 | 0.848 |

At the preregistered 200-epoch budget, capacity helps (0.3529 → 0.6275 recall from 149K
to 1.1M parameters) but 4.4M parameters at the same budget is worse than 1.1M. Budget
helps much more at fixed capacity: 149K parameters at 200/600/1200/3000 epochs reaches
recall 0.3529/0.7255/0.8824/**0.9608**. `dim128` at 600 epochs (0.8431) roughly matches
`dim64` at 1200 epochs (0.8824). Validation performance stayed at 0.0080–0.0400 thing
recall in every configuration while validation loss rose monotonically (to 31.50 at
3000 epochs), i.e. memorization without transfer, as expected for a diagnostic val set.

**Correction to an earlier reading of this table.** The 1200-epoch point alone suggested
that pooled PQ had saturated near 0.75 and that the gate's PQ ≥ 0.90 half was
"close to unreachable for this decoder irrespective of capacity". The 3000-epoch point
falsifies that: PQ continued to 0.8706 (final) with 0.8896 at its best evaluated epoch
(2250), and both RQ (0.961) and SQ (0.927) rose. PQ is budget-limited too, and the
saturation claim is withdrawn.

What the sweep does establish is that the preregistered H1 endpoint was
budget-limited rather than capacity-limited or architectural: at the same 149,343
parameters and the same 8 images, the recall half of the gate (0.95) is met from epoch
2000 onward (0.9608, 0.9804, 0.9608, 0.9608, 0.9608 at 2000–3000), and the model reaches
semantic macro F1 0.9313 and instance AP50 0.9729 on those 8 images. Within 3000 epochs
the PQ half is not met: PQ oscillates 0.8533–0.8896 over the last four evaluations and
its best value is 0.8896, about one point below the threshold.

**E2 — feature-ambiguity audit, extended from 8 to 256 train / 128 val images.**
Identical float32 model inputs that carry conflicting non-ignore targets are irreducible
for a geometry-only, permutation-equivariant model.

| Subset | Primitives | Duplicate-input groups | Mask conflicts | Label conflicts | Ambiguous primitives |
|---|---:|---:|---:|---:|---:|
| 8 train | 1,378 | 73 | 1 | 1 | 2 (0.15%) |
| 256 train | 55,784 | 4,358 | 177 | 134 | 368 (0.66%) |
| 128 val | 28,453 | 1,979 | 184 | 46 | 697 (2.45%) |

Ambiguity is real but small on the training subsets, so it cannot explain the H1
failure. The val subset contains a 3.7x higher ambiguous rate than the train subset,
which is worth remembering when comparing train and val numbers. The label-conflict
counter was added during this work: identical inputs with different semantic labels are
irreducible for the class head even when their target masks agree, which the original
mask-only check missed.

**E3 — where the 12.15% → 9.61% postprocessing loss happens.** Checkpoint
`primitive_256_s1337/last.pt`, 128 val images, 1,103 thing GT segments, decoder
gates fixed at IoU > 0.5, score ≥ 0.1, mask ≥ 0.5.

| Stage | Covered GT | Share |
|---|---:|---:|
| Raw query mask, any class | 322 | 29.19% |
| Plus correct predicted class | 134 | 12.15% |
| Plus score ≥ 0.1 | 134 | 12.15% |
| Plus decoder disjoint assignment and mask binarisation | **110** | **9.97%** |
| Plus class-conditional one-to-one matching (reported thing recall) | 106 | 9.61% |

24 of the 28 lost segments (86%) are lost inside the decoder's own
`argmax(score × mask)` assignment and `mask ≥ 0.5` binarisation, not in the evaluator's
one-to-one matching. The score gate costs nothing, confirming the earlier diagnosis.
The semantic map cannot move any of these rows: instances are built from query class
scores, never from the semantic map.

**Correction: that loss is a property of an under-trained model, not of the decoder.**
The same script on the converged 3000-epoch overfit checkpoint
(`cap_dim64_3000/last.pt`, train split, 51 thing GT segments) gives a flat ladder:

| Stage | Under-trained (20 epochs, val) | Converged (3000 epochs, train) |
|---|---:|---:|
| correct class | 134 / 1,103 (12.15%) | 49 / 51 (96.08%) |
| plus decoder disjoint assignment and binarisation | 110 / 1,103 (9.97%) | 49 / 51 (96.08%) |
| plus class-conditional one-to-one matching | 106 / 1,103 (9.61%) | 49 / 51 (96.08%) |

At convergence the decoder and the matcher together cost nothing; on the 20-epoch model
they cost 28 segments. Mask probabilities near 0.5 make the binarisation and the
`argmax` assignment noisy, so this cost tracks training progress rather than decoder
design. The earlier suggestion that the decoder assignment is "the first thing to
change" is withdrawn: the decoder is not the binding constraint, and the priority
returns to optimization budget. A 64-query cap was also ruled out as the cause — the
selected subsets peak at 52 (train) and 44 (val) target segments, with p95 at 28 and 27,
so no image comes near the cap.

**E4 — applying the decoder thresholds consistently to the semantic map.** The
semantic map is the only part of the decoder that votes with raw mask probabilities
instead of the gated, binarised masks. Making it consistent changes labels for 9,319 of
28,453 primitives (32.8%) and degrades every semantic variant:

| Semantic metric | Default (frozen) | Consistent thresholds |
|---|---:|---:|
| macro F1 | 0.095732 | 0.082691 |
| micro F1 | 0.229618 | 0.198300 |
| weighted macro F1 | 0.098713 | 0.083060 |
| weighted micro F1 | 0.250265 | 0.217878 |

So the ungated vote is not leaving metric points on the table; the alternative is
worse. The parameter is opt-in (deviation D4) and the default path was verified
bit-identical before/after the change by comparing per-image prediction digests over all
128 val images (combined digest `3010b830…abc7`), so the frozen decoder is untouched.

### Deviations from the preregistration

- **D1** — `--deterministic` was added to the pilot script after the preregistered runs
  and changes kernel selection, therefore numerics. The preregistered runs stay labelled
  nondeterministic; the deterministic runs are separate artifacts. The flag restarts the
  process once: cuBLAS reads `CUBLAS_WORKSPACE_CONFIG` before its first handle and the
  fused transformer encoder kernel refuses to run deterministically without it, so an
  in-process assignment is too late. The first attempt at the 3000-epoch probe failed
  with a `RuntimeError` from `_transformer_encoder_layer_fwd` for exactly this reason and
  is the reason the flag now re-executes instead of relying on the caller's shell.
- **D2** — `--heads`, `--encoder-layers`, `--decoder-layers` were added so the E1
  variants are expressible from the CLI rather than by editing the script.
- **D3** — E1–E4 were not preregistered. They were started after both preregistered
  endpoints were recorded and must not be quoted as preregistered outcomes.
- **D4** — `decode_primitive_masks` gained an opt-in
  `consistent_semantic_thresholds` parameter. Default thresholds, assignment and merge
  behaviour are unchanged; E4 is the only consumer.
- No held-out test predictions were produced and no test split was read.

### Artifacts

- Deterministic control and budget: `outputs/panoptic/det_control_a`, `det_control_b`,
  `det_budget100`.
- Capacity sweep: `outputs/panoptic/cap_dim{64,128,256}_{200,600,1200}` and
  `outputs/panoptic/cap_dim64_3000`.
- Preregistered (nondeterministic) runs: `outputs/panoptic/overfit8_20260917`,
  `outputs/panoptic/budget100_20260917_s1337`.
- Aggregates: `outputs/panoptic/optimization_summary_20260917.json`, rebuilt by
  `scripts/research/summarize_archcad_optimization.py`.
- Feature ambiguity: `outputs/panoptic/feature_ambiguity_{8train,256train,128val}_20260917.json`
  from `scripts/research/audit_archcad_feature_ambiguity.py`.
- Decode stages: `outputs/panoptic/decode_stages_s1337_20260917.json` from
  `scripts/research/diagnose_archcad_decode_stages.py`.
- Decoder inertness: `outputs/panoptic/_runs_20260917/decoder_{before,after,final}.json`,
  taken on the pre-edit code, after the opt-in parameter was added, and after the final
  dtype fix. All three agree.

### What would change these conclusions

1. Re-run the three-seed replication with `--deterministic` before treating
   `PQ = 0.11663 ± 0.01155` as the baseline; the ± is now known to include a
   kernel-level component that a fixed seed does not control.
2. Continue past 3000 epochs, or at `dim128`, to test whether the PQ ≥ 0.90 half of the
   gate is reachable: PQ was 0.8706 at epoch 3000 with a best of 0.8896 at epoch 2250,
   so the remaining gap is about one point and may close with budget rather than with a
   different decoder or representation.
3. The decoder is not the lever it first appeared to be: on a converged checkpoint its
   assignment and binarisation cost nothing (E3 correction). Spend the next GPU hours on
   optimization budget for the real subset, not on decoder redesign.

