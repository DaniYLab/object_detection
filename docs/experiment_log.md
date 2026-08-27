# Experiment Log — FloorPlanCAD Conditioned CenterNet

**Manifest:** `splits.json` fingerprint `f83fcffd294a7d536c5bad89fb132c79feaca87b250d185d6ea78d1fe5c064cc`
**Benchmark:** stuff excluded, min_size 8, image 512, stride 8, 30 classes with GT.
**Checkpoint selection:** `val AP50:95` (best_val_ap.pt), evaluated on val split.
**Data audit:** `docs/data_audit.md` — no blockers.

Parameter counts (measured):

| Preset | Params |
|---|---:|
| centernet_baseline (A0) | 3,674,031 |
| shared_no_condition (A1) | 4,886,959 |
| floorplan_base (B) | 5,268,741 |
| dual_pathway (V) | 6,391,557 |
| dual_pathway_v2 (V2) | 6,785,797 |
| dual_pathway_late (L) | 5,980,677 |
| dual_no_fusion (V-ctl) | 5,987,845 |
| shared_wide (A1') | 82,236,079 |
| per_class_no_text (E) | 85,707,781 |
| per_class_small (E') | 4,865,981 |

## Runs

Results are filled in from `outputs/evaluation_val_<preset>_seed<seed>.json` after each run.
Do NOT run held-out test until preset/checkpoint/threshold are frozen on val.

| ID | Preset | Seed | Epochs | Val AP50 | Val AP50:95 | Notes |
|---|---|---|---:|---:|---:|---|
| A0 | centernet_baseline | 1337 | 30 | 0.3819 | 0.1908 | best_val_ap.pt @ep25; val loss 10.56 @ep30 |
| A1 | shared_no_condition | 1337 | 30 | 0.3513 | 0.1685 | revised multi-class pathway control; BF16 |
| B | floorplan_base | 1337 | 30 | 0.4021 | 0.2072 | class embedding + FiLM; BF16; best AP @ep30 |
| A1 | shared_no_condition | 2024 | 30 | 0.3522 | 0.1691 | BF16; best AP @ep30; replication of seed 1337 |
| B | floorplan_base | 2024 | 30 | 0.4084 | 0.2143 | class embedding + FiLM; BF16; best AP @ep30 |
| A1 | shared_no_condition | 4242 | 30 | 0.3610 | 0.1713 | BF16; best AP @ep30 |
| B | floorplan_base | 4242 | 30 | 0.4151 | 0.2130 | class embedding + FiLM; BF16; best AP @ep30 |
| V | dual_pathway | 1337 | 30 | 0.5032 | 0.2737 | image + SVG strokes; BF16; V−B = **+10.11 AP50 / +6.65 AP50:95** |
| V | dual_pathway | 2024 | 30 | 0.4951 | 0.2760 | image + SVG strokes; BF16; V−B = **+8.67 AP50 / +6.17 AP50:95** |
| V2 | dual_pathway_v2 | 1337 | 30 | 0.5160 | 0.2742 | class-conditioned vector FiLM + length-weighted truncation; V2−V1 = +1.28 AP50 |
| V2 | dual_pathway_v2 | 2024 | 30 | 0.4970 | 0.2847 | V2−V1 = +0.19 AP50 / +0.87 AP50:95; replication |

## Planned comparisons

- A0 vs A1: pathway architecture effect (CNN vs GatedSpatialMixer+attention, both no conditioning).
- A1 vs B: conditioning effect isolated (same architecture).
- B vs C/D: class embedding vs byte text; FiLM vs cross-attention.
- E' vs A1': budget-matched routing.
- B vs V (dual_pathway): SVG vector information source effect (same conditioning, +21% params from the vector branch; V-ctl `dual_no_fusion` isolates fusion from capacity).

## Seed 1337 results

| Comparison | Δ AP50 | Δ AP50:95 | Interpretation |
|---|---:|---:|---|
| A1 − A0 | −0.0306 | −0.0223 | pathway attention alone underperforms CNN baseline |
| B − A1 | **+0.0508** | **+0.0386** | class embedding + FiLM strongly improves the same pathway |
| B − A0 | **+0.0202** | **+0.0164** | conditioned model also exceeds the project-native CNN baseline |

Conditioning improves AP50 for 24/30 classes, degrades 4, and ties 2. Largest AP50 gains: `washing_machine` +0.425, `plant` +0.200, `sofa` +0.149, `column` +0.133, `floor_plan_area` +0.101. Largest degradation is `door_revolving` −0.288, but val has only 6 GT instances.

Absent-class behavior at decoder threshold 0.05:

| Model | Absent query with ≥1 prediction | Mean absent max score | Mean present max score |
|---|---:|---:|---:|
| A0 | 60.54% | 0.0895 | 0.5886 |
| A1 | 64.25% | 0.1016 | 0.5683 |
| B | **50.24%** | **0.0819** | **0.6477** |

B both suppresses absent-class hallucinations and raises confidence on present classes. The absent detection rate remains high at a permissive 0.05 threshold, so threshold calibration and explicit absent-query metrics remain follow-up work.

These results satisfy the **single-seed preliminary** success condition (B exceeds A1 by >1 AP50 point), but the protocol requires ≥2/3 seeds before making a robust final claim.

## Three-seed replication summary (val, full 1,016 images)

| Preset | Seed 1337 | Seed 2024 | Seed 4242 | Mean AP50 | Mean AP50:95 | Std AP50 |
|---|---:|---:|---:|---:|---:|---:|
| A0 `centernet_baseline` | 0.3819 | — | — | 0.3819 | 0.1908 | — |
| A1 `shared_no_condition` | 0.3513 | 0.3522 | 0.3610 | 0.3548 | 0.1696 | 0.0043 |
| B `floorplan_base` | 0.4021 | 0.4084 | 0.4151 | 0.4085 | 0.2115 | 0.0053 |

Per-seed conditioning effect (B − A1):

| Seed | Δ AP50 | Δ AP50:95 |
|---|---:|---:|
| 1337 | +0.0508 | +0.0387 |
| 2024 | +0.0562 | +0.0452 |
| 4242 | +0.0541 | +0.0417 |
| **Mean** | **+0.0537** | **+0.0419** |

The conditioning gain (+5.37 AP50 / +4.19 AP50:95 mean) replicates on all
three seeds with seed-to-seed std ≈ 0.005 — the effect is ~10× larger than
its variance. The ≥2/3 seeds protocol requirement is satisfied; the core
research claim (class conditioning improves the pathway architecture) is now
robust. A0 was run on seed 1337 only.

## Dual-pathway (vector branch) summary — val, full 1,016 images

| Preset | Seed 1337 | Seed 2024 | Mean AP50 | Mean AP50:95 |
|---|---:|---:|---:|---:|
| B `floorplan_base` | 0.4021 | 0.4084 | 0.4053 | 0.2108 |
| V `dual_pathway` | 0.5032 | 0.4951 | 0.4992 | 0.2749 |
| V2 `dual_pathway_v2` | 0.5160 | 0.4970 | 0.5065 | 0.2795 |

Per-seed vector-source effect (V − B):

| Seed | Δ AP50 | Δ AP50:95 |
|---|---:|---:|
| 1337 | +0.1011 | +0.0665 |
| 2024 | +0.0867 | +0.0617 |
| **Mean** | **+0.0939** | **+0.0641** |

The SVG vector-information gain (**+9.39 AP50 / +6.41 AP50:95 mean**) replicates
on 2/2 seeds and is ~1.8× the conditioning gain. The claim — in the protocol's
terms — is that the dual-pathway architecture exploits information the raster
pipeline discards (exact primitive geometry), at +21% parameters and +11%
step time.

V2 refinement (class-conditioned vector FiLM + length-weighted truncation):
+0.74 AP50 / +0.46 AP50:95 mean over V, replicated 2/2 seeds (1337: +1.28/+0.05;
2024: +0.19/+0.87). The AP50:95 gain is consistent; the AP50 gain varies by
seed (0.19–1.28). V2 is the repo's best model. Test evaluation should wait
until the vector design is frozen.

## Held-out test — seed 1337

The user explicitly requested test evaluation after checkpoints/decoder settings had already been selected on validation. No test result was used to alter checkpoint, threshold, top-k, or model config.

| Model | Test AP50 | Test AP50:95 | Test−val AP50 | Test−val AP50:95 |
|---|---:|---:|---:|---:|
| A0 `centernet_baseline` | 0.3529 | 0.1922 | −0.0290 | +0.0014 |
| A1 `shared_no_condition` | 0.3334 | 0.1773 | −0.0179 | +0.0088 |
| **B `floorplan_base`** | **0.3896** | **0.2170** | −0.0125 | +0.0098 |

Test deltas:

- B − A1: **+5.63 AP50**, **+3.97 AP50:95**.
- B − A0: **+3.67 AP50**, **+2.48 AP50:95**.
- B improves 24/29 test classes with GT vs A1.
- Absent-query detection rate at threshold .05: A0 61.55%, A1 64.89%, B 50.91%.

The conditioning gain therefore generalizes to test for seed 1337. This remains a single-seed result; seed-2024 replication was paused at user request.

## Held-out test — vector models (seed 1337, historical exploratory result)

V/V2 were evaluated after their designs were frozen, but these reports use
manifest fingerprint `a916b9fa...` (26,588 test GT), while the historical B
report uses `f83fcffd...` (27,095 test GT). The protocol forbids treating
cross-manifest deltas as a controlled comparison. Therefore the absolute V/V2
results remain valid for their recorded manifest, but **V−B test deltas are
withdrawn until all models are evaluated under one frozen manifest**.

| Model | Manifest | Val AP50 | Test AP50 | Test AP50:95 | Test−val AP50 |
|---|---|---:|---:|---:|---:|
| B `floorplan_base` | `f83fcffd...` | 0.4021 | 0.3896 | 0.2170 | −0.0125 |
| V `dual_pathway` | `a916b9fa...` | 0.5032 | **0.4868** | **0.2837** | −0.0164 |
| V2 `dual_pathway_v2` | `a916b9fa...` | 0.5160 | 0.4851 | 0.2759 | −0.0309 |

Allowed conclusions:

- V and V2 absolute test results are historical/exploratory results tied to
  their recorded manifests.
- V2 is lower than V on the same `a916b9fa...` manifest. This observation is
  retained, but it must not drive future architecture selection because test
  has now been exposed.
- The vector-information claim remains supported by the controlled validation
  comparison on matching manifests and two seeds. A final held-out V−B test
  claim requires one future frozen manifest and a prespecified final protocol.

## Fresh v2 validation hierarchy — speed-quality experiment

The following run uses the frozen schema-v2 manifest
`data/FloorPlanCAD_original/splits_v2.json` (fingerprint
`203f63588e906637f4dbcf6d1a001e67d0e688016c9b0c799c3ea89b438e8e9a`). The roles are
image-disjoint: `train_new` was used for training, `val_select` for checkpoint
selection, and `val_calibration` only for decoder calibration. `test_set` was not
accessed.

### `dual_pathway_late`, seed 1337

Configuration: 5 epochs, batch size 24, BF16, balanced sampler (`balance_power=0.5`),
one absent-class negative query per positive, 1,024 SVG stroke tokens, output stride
8, and periodic AP checks limited to 256 `val_select` images. The final checkpoint
was selected by validation AP50:95 and is `checkpoints/dual_pathway_late_seed1337/best_val_ap.pt`.
The limited periodic result at epoch 5 was AP50 0.1879 / AP50:95 0.0808; these are
not the official full-role metrics.

| Role | Images | GT boxes | Predictions | AP50 | AP50:95 | Decoder |
|---|---:|---:|---:|---:|---:|---|
| `val_select` | 1,016 | 8,476 | 801,252 | 0.1598 | 0.0668 | threshold 0.05, top-k 100, peak 3 |
| `val_calibration` baseline | 1,016 | 8,292 | 779,067 | 0.1553 | 0.0646 | threshold 0.05, top-k 100, peak 3 |
| `val_calibration` calibrated | 1,016 | 8,292 | 779,067 | 0.1553 | 0.0646 | per-class thresholds |

Per-class calibration selected the default 0.05 threshold for all 35 classes. The
validation-only global sweep also selected 0.05 by AP50:95; higher thresholds
monotonically reduced AP. Thresholds below 0.05 are not inferable from these
predictions because decoding had already discarded lower-scoring detections.
Artifacts:
`outputs/evaluation_val_select_dual_pathway_late_seed1337.json`,
`outputs/evaluation_val_calibration_dual_pathway_late_seed1337.json`,
`outputs/predictions_val_calibration_dual_pathway_late_seed1337.json`,
`outputs/class_thresholds_val_calibration_dual_pathway_late_seed1337.json`, and
`outputs/threshold_sweep_val_calibration_dual_pathway_late_seed1337.json`.

### Speed and memory profile

Synchronized CUDA BF16 profiling used batch size 1, 512px images, 35 classes,
1,024 synthetic valid vector tokens, 10 warm-up steps, and 20 measured steps.
The comparison is between `dual_pathway_v2` and `dual_pathway_late`; both use the
same input resolution and vector-token budget, but late mode is not capacity
matched and is not a pure conditioning-stage ablation. It removes V2 early fusion,
disables class-conditioned vector FiLM, and has fewer parameters.

| Class chunk | Model | Params | Selected train median (s/step) | All-class median (s/image) | All-class p95 (s/image) | Throughput (img/s) | Infer peak alloc (GiB) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | V2 | 6,785,797 | 0.01980 | 0.09973 | 0.10109 | 10.03 | 0.2337 |
| 4 | Late | 5,980,677 | 0.01909 | 0.01010 | 0.01101 | 99.00 | 0.0957 |
| 35 | V2 | 6,785,797 | 0.01980 | 0.08834 | 0.08908 | 11.32 | 1.5607 |
| 35 | Late | 5,980,677 | 0.01954 | 0.00583 | 0.00626 | 171.67 | 0.1985 |

Relative to V2, late mode reduces parameters by 11.86%. All-class latency is
9.87x lower for chunk 4 and 15.17x lower for chunk 35; p95 latency speedups are
9.18x and 14.23x. Inference peak allocated memory falls 59.08% and 87.28%,
respectively. Selected-query training is effectively unchanged (1.04x and 1.01x
lower step time for late mode). These are synchronized synthetic profiles, not an
end-to-end dataloader throughput claim.

### Interpretation and comparison boundary

The late run establishes a strong speed/memory result, but its quality is currently
well below the historical V2 numbers. Those V2 numbers were generated under the
legacy manifest (`splits.json`, fingerprint `a916b9fa...`), so the quality gap is
not a controlled architecture comparison. A direct quality claim requires a V2
control trained and evaluated under the same schema-v2 manifest. The late result
should therefore be reported as a speed-oriented architecture experiment, not as a
pure causal test of moving conditioning later and not as a parameter-matched result.

## Decisions / observations

- 2026-07-25: Pilot (200 samples, 2 epochs) confirmed loss decreases 55→41, no NaN, val-AP + dual-checkpoint pipeline works. First real run: A0 baseline, 30 epochs, batch 32, lr 5e-4, neg=1.
- 2026-07-30: A0 completed: full-val AP50=0.3819, AP50:95=0.1908. A1 FP32 attempt was stopped during epoch 3 because 4096-token attention at batch 32 took ~4.6 s/step and used ~15.8 GiB VRAM. Added `--precision bf16` AMP path (BF16 does not require GradScaler); A1/B runs use BF16 and record precision in checkpoint runtime config.
- 2026-08-24: Seed 2024 replication. A1: 0.3522/0.1691 (vs 1337: 0.3513/0.1685 — ΔAP50 0.0009). B: 0.4084/0.2143. Conditioning effect B−A1 = **+5.62 AP50 / +4.52 AP50:95** (seed 1337: +5.08/+3.86). The conditioning gain replicates on a second seed, satisfying the ≥2/3 seeds requirement.
- 2026-08-25: Seed 4242 replication completed. A1: 0.3610/0.1713; B: 0.4151/0.2130 (B−A1 = +5.41/+4.17). Three-seed means: A1 0.3548/0.1696, B 0.4085/0.2115; mean conditioning effect **+5.37 AP50 / +4.19 AP50:95**, seed std ≈ 0.005. Core claim confirmed on 3/3 seeds. SVG stroke audit (`docs/stroke_audit.md`) and dual-pathway spec (`docs/dual_pathway_spec.md`) prepared for the next research phase.
- 2026-08-25: Dual-pathway implemented: `src/data/strokes.py` (12-dim token superset, normalize/sample/pad), `extract_strokes` in metadata.py (schema-v3 `strokes` field, whole-drawing, geometry-only — never semantic/instance ids), `VectorEncoder` (SelfAttention+FFN reusing image-pathway primitives, TypeEmbedding, padding mask), vector cross-attention fusion in `FloorPlanDetector.encode_vector` (image tokens = Query, strokes = K/V, once per image before conditioning), presets `dual_pathway` (6.39M params, +21% over B) and `dual_no_fusion` control. Metadata regenerated with `--include-strokes` (instances unchanged; strokes add ~700 tokens/drawing mean, n_max=1024 random-sampled per epoch at train, linspace-deterministic at eval). 122 tests pass.
- 2026-08-25: **V (dual_pathway) seed 1337: val AP50=0.5032, AP50:95=0.2737** — vs B seed 1337 (0.4021/0.2072) = **+10.11 AP50 / +6.65 AP50:95** (+25% relative). The vector-information gain is twice the conditioning gain. Training curve: ep5 0.2801, ep10 0.3976, ep15 0.4766, ep20 0.5085, ep30 0.5224 (limited-256-image val); full-val numbers from best_val_ap.pt. Overhead: +11% step time, +0.9 GB VRAM. Claim discipline: the gain comes from an additional information source (SVG vector geometry), not architecture alone — V−B must be replicated on seed 2024 before final claims.
- 2026-08-25: **V seed 2024 replicated: val AP50=0.4951, AP50:95=0.2760** (V−B = +8.67/+6.17). Two-seed mean vector effect: **+9.39 AP50 / +6.41 AP50:95** — the vector-information claim holds on 2/2 seeds with seed variance (~0.004) far below the effect. V is now the best model in the repo; A1/B/V ladder demonstrates three stacked, individually replicated gains (architecture control → class conditioning → vector information source).
- 2026-08-26: **V2 (dual_pathway_v2) seed 1337: val AP50=0.5160, AP50:95=0.2742** — vs V1 same seed (0.5032/0.2737) = **+1.28 AP50** at +6% params (6.79M vs 6.39M). V2 bundles two improvements: (a) class-conditioned vector fusion — the queried class embedding FiLM-modulates encoded strokes before cross-attention, so a "toilet" query attends preferentially to arc-heavy strokes; (b) length-weighted stroke truncation (subsampling ∝ endpoint diagonal, 2% floor) replacing uniform random for the 4.5% of drawings above n_max=1024. Note V2−V1 mixes both effects; a single-seed result, pending replication. Threshold sweep on V1 confirmed decoder 0.05 is already optimal (AP flat 0.01–0.05, monotone decline above) — V is well calibrated, no free AP from recalibration.
- 2026-08-26: **V2 seed 2024 replicated: val AP50=0.4970, AP50:95=0.2847** (V2−V1 = +0.19/+0.87). Two-seed V2−V1 mean: **+0.74 AP50 / +0.46 AP50:95** — the refinement replicates, with the AP50:95 gain consistent across seeds and the AP50 gain seed-dependent (0.19–1.28). V2 mean (0.5065/0.2795) is the repo's best. The full ladder A1→B→V→V2 now reads: 0.3548 → 0.4085 → 0.4992 → 0.5065 mean val AP50 (+15.2 total, +43% relative over the architecture control), with every step replicated on ≥2 seeds.
- 2026-08-26: **Historical held-out test run for vector models.** V: test AP50=0.4868, AP50:95=0.2837; V2: 0.4851/0.2759. Later protocol audit found that these vector reports use manifest `a916b9fa...` while the historical B report uses `f83fcffd...` with a different GT count, so the previously stated V−B test delta is withdrawn as a controlled claim. V/V2 remain comparable to each other on their shared manifest, but this exposed test observation must not guide future model selection. Also fixed checkpoint config forward compatibility through semantic ModelConfig comparison.
- The original A1 query-head control was invalid: without class/text signal, a shared 1-channel head produces identical output for every class query. It achieved only AP50=0.0084 at epoch 5. Replaced it with `floorplan_unconditioned`, which keeps the pathway architecture but uses a multi-class 5C head. The revised control trained normally and reached full-val AP50=0.3513.
- Seed 1337 conclusion: conditioning provides +5.08 AP50 and +3.86 AP50:95 over the same pathway control, and +2.02/+1.64 over A0. This is preliminary until replicated on two more seeds.
