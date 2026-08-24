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
| B | floorplan_base | 2024 | 30 | _running_ | _running_ | class embedding + FiLM; BF16 |

## Planned comparisons

- A0 vs A1: pathway architecture effect (CNN vs GatedSpatialMixer+attention, both no conditioning).
- A1 vs B: conditioning effect isolated (same architecture).
- B vs C/D: class embedding vs byte text; FiLM vs cross-attention.
- E' vs A1': budget-matched routing.

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
- The original A1 query-head control was invalid: without class/text signal, a shared 1-channel head produces identical output for every class query. It achieved only AP50=0.0084 at epoch 5. Replaced it with `floorplan_unconditioned`, which keeps the pathway architecture but uses a multi-class 5C head. The revised control trained normally and reached full-val AP50=0.3513.
- Seed 1337 conclusion: conditioning provides +5.08 AP50 and +3.86 AP50:95 over the same pathway control, and +2.02/+1.64 over A0. This is preliminary until replicated on two more seeds.
