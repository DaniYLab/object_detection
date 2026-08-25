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

## Decisions / observations

- 2026-07-25: Pilot (200 samples, 2 epochs) confirmed loss decreases 55→41, no NaN, val-AP + dual-checkpoint pipeline works. First real run: A0 baseline, 30 epochs, batch 32, lr 5e-4, neg=1.
- 2026-07-30: A0 completed: full-val AP50=0.3819, AP50:95=0.1908. A1 FP32 attempt was stopped during epoch 3 because 4096-token attention at batch 32 took ~4.6 s/step and used ~15.8 GiB VRAM. Added `--precision bf16` AMP path (BF16 does not require GradScaler); A1/B runs use BF16 and record precision in checkpoint runtime config.
- 2026-08-24: Seed 2024 replication. A1: 0.3522/0.1691 (vs 1337: 0.3513/0.1685 — ΔAP50 0.0009). B: 0.4084/0.2143. Conditioning effect B−A1 = **+5.62 AP50 / +4.52 AP50:95** (seed 1337: +5.08/+3.86). The conditioning gain replicates on a second seed, satisfying the ≥2/3 seeds requirement.
- 2026-08-25: Seed 4242 replication completed. A1: 0.3610/0.1713; B: 0.4151/0.2130 (B−A1 = +5.41/+4.17). Three-seed means: A1 0.3548/0.1696, B 0.4085/0.2115; mean conditioning effect **+5.37 AP50 / +4.19 AP50:95**, seed std ≈ 0.005. Core claim confirmed on 3/3 seeds. SVG stroke audit (`docs/stroke_audit.md`) and dual-pathway spec (`docs/dual_pathway_spec.md`) prepared for the next research phase.
- 2026-08-25: Dual-pathway implemented: `src/data/strokes.py` (12-dim token superset, normalize/sample/pad), `extract_strokes` in metadata.py (schema-v3 `strokes` field, whole-drawing, geometry-only — never semantic/instance ids), `VectorEncoder` (SelfAttention+FFN reusing image-pathway primitives, TypeEmbedding, padding mask), vector cross-attention fusion in `FloorPlanDetector.encode_vector` (image tokens = Query, strokes = K/V, once per image before conditioning), presets `dual_pathway` (6.39M params, +21% over B) and `dual_no_fusion` control. Metadata regenerated with `--include-strokes` (instances unchanged; strokes add ~700 tokens/drawing mean, n_max=1024 random-sampled per epoch at train, linspace-deterministic at eval). 122 tests pass.
- 2026-08-25: **V (dual_pathway) seed 1337: val AP50=0.5032, AP50:95=0.2737** — vs B seed 1337 (0.4021/0.2072) = **+10.11 AP50 / +6.65 AP50:95** (+25% relative). The vector-information gain is twice the conditioning gain. Training curve: ep5 0.2801, ep10 0.3976, ep15 0.4766, ep20 0.5085, ep30 0.5224 (limited-256-image val); full-val numbers from best_val_ap.pt. Overhead: +11% step time, +0.9 GB VRAM. Claim discipline: the gain comes from an additional information source (SVG vector geometry), not architecture alone — V−B must be replicated on seed 2024 before final claims.
- The original A1 query-head control was invalid: without class/text signal, a shared 1-channel head produces identical output for every class query. It achieved only AP50=0.0084 at epoch 5. Replaced it with `floorplan_unconditioned`, which keeps the pathway architecture but uses a multi-class 5C head. The revised control trained normally and reached full-val AP50=0.3513.
- Seed 1337 conclusion: conditioning provides +5.08 AP50 and +3.86 AP50:95 over the same pathway control, and +2.02/+1.64 over A0. This is preliminary until replicated on two more seeds.
