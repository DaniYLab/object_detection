# Experiment Log — FloorPlanCAD Conditioned CenterNet

## ArchCAD continuation — 2026-09-16

**Fixed-subset replication:** preregistration `30d739f`; 256 train / 128 val,
20 epochs, seeds 1337/2024/4242, identical selected image IDs and source/code
hashes. Local pooled PQ = **0.11663 ± 0.01155**, semantic macro F1 =
**0.09245 ± 0.00330**, instance AP50 = **0.007637 ± 0.001405** (sample SD).
Thing recall = **0.09640 ± 0.00590**. This is a repeatable weak development
control. It is not a converged baseline or paper reproduction.
Protocol and per-seed/recall diagnosis: `docs/experiments/archcad_primitive_replication_20260916.md`.

The first learned public-40K panoptic pilot completed: 149,343-parameter
primitive-only query/mask model, seed 1337, 32 train / 16 validation chunks,
20 epochs (640 steps), no target truncation. Local validation macro semantic
F1=0.09362, primitive-mask AP50=0.000787, mAP=0.000475, pooled PQ=0.08073
(TP=10, FP=29, FN=154). Evaluation loss fell 14.84317 -> 8.44854 on validation
and 14.85088 -> 6.43248 on train. These are pilot feasibility observations, not
converged benchmarks or paper-comparable scores.

Protocol, input audit and executed upstream metric differences:
`docs/research/archcad_benchmark_audit_20260916.md`. Full report, pre-run config,
selected sample IDs/source hashes and checkpoint:
`outputs/panoptic/primitive_pilot_20260916_s1337/`.

## Historical FloorPlanCAD experiments

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

### Same-manifest V2 control (controls for the late comparison)

To make the late quality gap a controlled claim, `dual_pathway_v2` was retrained
on the same schema-v2 manifest, seed 1337, 30 epochs, BF16, batch 24, balanced
sampler, one absent-class negative query per positive. Checkpoint dir
`checkpoints/dual_pathway_v2_seed1337_v2manifest/` is distinct from the legacy
run. Full `val_select` (1,016 images, 8,476 GT boxes):

| Model | Epochs | AP50 | AP50:95 |
|---|---:|---:|---:|
| V2 control (`best_val_ap.pt`) | 30 | **0.4883** | **0.2779** |
| V2 control (`epoch_005.pt`, budget-matched) | 5 | 0.2309 | 0.0924 |
| Late (`best_val_ap.pt`) | 5 | 0.1598 | 0.0668 |

Budget-matched (5 epochs, same manifest/seed/config): late − V2 = **−0.0711 AP50 /
−0.0256 AP50:95**. The gap is structural (late drops V2 early fusion +
class-conditioned vector FiLM and is 11.86% smaller), not undertraining.

Ceiling: V2@30 reaches 0.4883/0.2779 on `val_select`. This is below the legacy
`a916b9fa...` V2 val number (0.5160/0.2742 on a different manifest) — a manifest
effect, not a regression, and confirms results must be compared only within one
manifest. Both manifests contain 1,016 validation images but differ in assignment
and provenance.

### One-pass student distillation (response-based KD)

`dual_pathway_late` was distilled from the frozen V2 control teacher
(`checkpoints/dual_pathway_v2_seed1337_v2manifest/best_val_ap.pt`) with
response-only KD (center-logits soft targets + masked size/offset MSE). Two
configurations were tested at 5 epochs on `val_select`, seed 1337, same
manifest/loader:

| Config | Full `val_select` AP50 | Full AP50:95 | limited-256 AP50 |
|---|---:|---:|---:|
| Late non-KD (baseline) | 0.1598 | 0.0668 | 0.1879 |
| KD center full-map (w=2.0) | — (stopped at ep5, limited 0.1535) | — | 0.1535 |
| KD center-gated (w=1.0) | 0.1550 | 0.0664 | 0.2213 |

Conclusion: **response-only distillation does not close the structural gap.**
Full-map center KD hurt (limited-256 0.1535 < 0.1879); gating center KD to
positive cells reverse that on the 256-image subset (0.2213 > 0.1879) but the
full-role result is a null (0.1550 ≈ 0.1598). The limited-256 subset is not a
reliable selection signal — full-role is the only trustworthy metric.

The −7 AP50 gap is structural: the late student lacks V2's class-conditioned
vector FiLM and early token-level fusion, so a class-agnostic response head
cannot absorb V2's class-specific soft targets. Response-only KD is rejected as
the mechanism to recover it; a future attempt would need feature-level KD on the
post-FiLM hidden, the class-conditioned vector encoding, or additional student
capacity — none of which preserve the current pure one-pass property for free.

### Per-class diagnosis of the V2 control (val_select, splits_v2, seed 1337)

Full-role `val_select` diagnostic of `best_val_ap.pt` (AP50 0.4883, GT-weighted
AP50 0.6778). Macro AP is dragged down by a small set of thin/rare classes, not
by an across-the-board weakness:

- **Thin-object bottleneck (the largest lever):** `stair` (214 GT, not rare) is
  only 0.104, `ramp` 0.122, `escalator_stair` 0.036. These share a signature:
  median min-side ≈ 5.1 px (< stride 8) and extreme aspect ratio (stair 7.0,
  ramp 20.0); 60–100% of their boxes have a side below the 8 px stride-8 cell.
  A CenterNet stride-8 heatmap cannot place a peak on an object thinner than one
  cell, so the box is missed regardless of how well the vector branch sees the
  parallel-line geometry. This matches the source-table "Stair" category being
  the lowest. `escalator_stair` (172 px median side) is a different symptom —
  large but zig-zag, likely split/inconsistent annotation.
- **Small-but-not-thin classes can still do well:** `plant` (median side 15.1 px,
  AP50 0.891) and `parking` (20.8 px, 0.899) prove small size alone is not the
  issue — only sub-stride thin extent is.
- **Rare-class variance:** `door_revolving` (4 GT), `window_bay` (3), `ramp` (6)
  have too few GT for stable single-seed AP.

Conclusion: the highest-value quality lever is a **stride-4 (or hybrid
thin-object) head**, not more vector encoding — the image path already sees the
geometry at 8 px resolution, but the detection grid cannot resolve sub-cell
thin objects. The DPSS paper reaches the same conclusion via its point-based
primitive branch; here the primitive branch exists but the box head resolution
is the binding constraint.

Collision/target audit for stride 4 (val_select, 8,476 valid boxes):

- 6.8% of boxes have min-side < 8 px (< 1 stride-8 cell); 100% of those are
  ≥ 1 stride-4 cell, i.e. stride 4 would resolve all of them.
- **33.6% of boxes have gaussian radius == 0 at stride 8** (radius from
  CornerNet min_overlap=0.7 collapses for sub-cell boxes), so a third of targets
  are bare 1-pixel peaks with no gaussian support for focal learning.
- Collisions at stride 8 are negligible (22 total, mostly toilet), so stride 4
  does not add a collision problem.
- Cost: stride 4 is 128×128 = 16,384 tokens/image (4× the 4,096 at stride 8),
  which stresses the global self-attention trunk (O(n²), OOM risk).

Decision: resolution is the binding constraint. A cheap counter-hypothesis was
tested first — widening the gaussian via `--min-overlap` — and refuted by a
10-second check: for any box whose shorter output edge is 1 cell, the CornerNet
radius stays < 1 (0.47→0.95 across min_overlap ∈ [0.2, 0.7]) and `int(...)` still
truncates to 0. A height-1 output cell physically cannot host a Gaussian bump, so
`min_overlap` cannot recover the 33.6% radius==0 set (it only reshapes the wider
boxes that already have radius ≥ 1). The `--min-overlap` knob is retained and
wired/tuned for that valid, weaker effect.

The real fix is a **hybrid stride-4 head**: keep the stride-8 trunk (4,096 tokens,
global attention — unchanged, no OOM) and upsample the trunk feature map to a
128×128 CenterNet head, so sub-cell thin objects (stair/ramp) gain a ≥1-cell
heatmap and a non-degenerate Gaussian. This targets the 6.8% sub-cell + 33.6%
radius==0 sets without the 16,384-token global-attention blowup of a full stride-4
trunk.

### Hybrid stride-4 thin-object head (head_stride=4)

`dual_pathway_v2_thin` keeps the stride-8 trunk (4,096 tokens, global attention
unchanged) but bilinearly upsamples the trunk feature map to a 128x128 grid
before the CenterNet head, so sub-cell thin objects gain a >=1-cell heatmap and
a non-degenerate Gaussian. Same manifest/seed/config/budget (30 epochs) as the
V2 control. Full `val_select` (1,016 images, 8,476 GT boxes):

| Model | AP50 | AP50:95 |
|---|---:|---:|
| V2 control (stride-8 head, seed 1337) | 0.4883 | 0.2779 |
| Thin (stride-4 head, seed 1337) | 0.4950 | 0.3091 |
| Thin (stride-4 head, seed 2024) | 0.4882 | 0.3035 |
| Thin mean (2 seeds) | 0.4916 | 0.3063 |
| Δ thin − V2 (seed 1337) | +0.0067 | **+0.0312** |
| Δ thin − V2 (seed 2024, vs seed-1337 V2) | −0.0001 | **+0.0256** |

**Two-seed replication verdict:** the thin-head AP50:95 gain replicates on
2/2 seeds (+2.6 to +3.1 points vs the V2 control), with seed-to-seed AP50:95
std ≈ 0.003 — the effect is ~10× its variance. AP50 is neutral (mean +0.3
points; seed 2024 AP50 ≈ the seed-1337 V2 control). Caveat: the V2 control
exists only for seed 1337, so the seed-2024 delta is read against the seed-1337
control; the thin seeds are mutually consistent, and the AP50:95 claim is
robust under either reading. The hybrid stride-4 head is the repo's first
architecture-level, same-manifest quality improvement replicated on ≥2 seeds.

The thin head improves AP50:95 by +3.1 points (box localization) and AP50 by
+0.7. Per-class, the hypothesis target **stair** jumps from 0.104 to 0.304
(**+0.201**, 214 GT — not rare), and shower +0.097, sink +0.100,
floor_plan_area +0.134. Thin-but-not-rare `stair` is the largest and most
decisive single-class gain. Some larger classes regress (column −0.095,
bathtub −0.076, oven −0.073) and rare thin classes are noise (ramp −0.102 on 6
GT, elevator −0.075 on 15 GT). `escalator` (min-side ~23 px, not thin) is
unchanged (−0.012), consistent with it being a representation problem, not a
resolution one.

### min_overlap sweep (gaussian radius width)

`--min-overlap` controls the CornerNet gaussian radius. Lower = wider gaussian.
Budget-matched 5-epoch runs on `dual_pathway_v2`, seed 1337, same
manifest/loader; full `val_select`:

| Config | AP50 | AP50:95 |
|---|---:|---:|
| mo=0.7 (baseline, V2 control) | 0.2309 | 0.0924 |
| **mo=0.5** | **0.2897** | **0.1233** |
| **mo=0.3** | **0.2925** | **0.1268** |
| Δ mo0.3 − mo0.7 | **+0.0616** | **+0.0344** |
| Δ mo0.5 − mo0.7 | **+0.0588** | **+0.0309** |

**30-epoch verification (the decisive check):** V2 + mo=0.5 trained to the full
30-epoch budget reaches full-role `val_select` **0.4965/0.2820** vs the mo=0.7
control 0.4883/0.2779 — only **+0.8 AP50 / +0.4 AP50:95** survives at
convergence. The large 5-epoch gain (+5.9 AP50) was almost entirely a
faster-early-convergence effect, not a higher ceiling, mirroring the KD
lesson in the opposite direction. `min_overlap` is therefore a training-speed
knob, not a quality lever, at this budget; the thin head remains the only
verified quality lever (+3.1 AP50:95 at 30 epochs). Future short-budget
screening results must always be re-verified at full budget before adoption.

### Interpretation and comparison boundary

The late run establishes a strong speed/memory result. Budget-matched quality on
the same manifest is now quantified: late@5ep (0.1598/0.0668) vs V2@5ep
(0.2309/0.0924) — a −7.1 AP50 / −2.6 AP50:95 structural gap. V2@30
(0.4883/0.2779) is the same-manifest quality ceiling next to which the late track
should be judged. The quality gap is not a controlled architecture claim until a
same-manifest V2 control exists; that control has now been produced. The late
result should therefore be reported as a speed-oriented architecture experiment
with a known quality cost, not as a pure causal test of moving conditioning later
and not as a parameter-matched result.

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

## Protocol audit — 2026-09-13 (documentation falsification + resume analysis)

**Context.** A fresh read of the whole `docs/` tree, the full session transcript
(14,690 lines / 135 real user turns) and `Resource/` was done to re-derive the
true state before further training. Result: the docs written on 2026-09-05/06
contain fabricated numbers and are now archived to
`docs/archive/fabricated_20260905/`. `experiment_log.md` remains the only
authoritative results record.

### Falsified claims (each verified against code or checkpoints)

| Claim in archived docs | Verified reality |
|---|---|
| "thin-head is image-only, 15M params" | `dual_pathway_v2_thin` = **6,785,797 params, vector branch enabled** |
| "ResNet18 dual-pathway ~0.8 s/batch" | That model was never trained; **no speed measurement exists** |
| "sequential = 35 x 5.8ms = 203ms" | All-class chunk-35 latency is **5.83 ms** (line 217). 203 ms multiplies a *selected-query training step* by 35 — inflated **35x** |
| "FloorPlanLLM ViT 56.5M is the current architecture" | No such model in the tree; ViT line was never evaluated |
| cites `image_encoder.py`, `centernet_heads.py`, `audit_stride8_collisions.py` | **Files do not exist** |
| "hybrid 15-20x speedup" | Built on the fabricated 203 ms baseline; target void |
| "Cross-attention Selector (trainable)" | `ReasoningSelector` has **0 parameters**; pure if/else heuristic |

### Convergence audit (the actionable finding)

Every top run stopped at a hard **30-epoch cap**. Reading
`checkpoint["metrics"]["val_ap50_95"]` (which had never been inspected before)
shows validation AP was **still rising** at the cap:

| Model | ep20 | ep25 | ep30 | Δ(25→30) |
|---|---:|---:|---:|---:|
| thin s1337 | 0.2932 | 0.3320 | **0.3389** | +0.0069 |
| thin s2024 | 0.2895 | 0.3154 | **0.3296** | +0.0142 |
| V2 s1337 | 0.2670 | 0.2899 | **0.3057** | +0.0158 |

Caveat: these trajectory numbers come from the **256-image periodic subset**
(`limit_val_ap_images=256`), so they are a monitoring signal, not the official
full-role metric. Re-running the full 1,016-image `val_select` evaluation of
`epoch_030.pt` reproduces exactly **0.4950 / 0.3091**, identical to the stored
`best_val_ap.pt` report — confirming `best_val_ap.pt` **is** epoch 30 (there is
no earlier converged checkpoint) and that the 256-subset overestimates the full
role by ~+3 AP50:95 (0.3389 vs 0.3091).

Therefore the honest convergence statement is: the only available trajectory
signal (256-subset, measured every epoch) was still rising at the cap, but the
full-role metric at epoch 30 is 0.3091 and no full-role measurement exists past
that point. Whether the full role keeps improving beyond 30 epochs is an **open
question**, not an established fact. The model has **never been trained past 30
epochs**.

### Exact-resume constraint discovered

`validate_exact_resume_compatibility` compares `runtime_config` byte-for-byte, and
`_build_runtime_config` embeds `optimization.epochs` and
`scheduler.total_steps`. Therefore extending 30 → 100 epochs **cannot** use
exact resume: the stored `total_steps=38010` would differ from the rebuilt
`100*1267=126700` and the call would raise `CheckpointError`.

Two viable routes:
1. **`--weights-only`** from `epoch_030.pt` with `--epochs 100`: starts a fresh
   optimizer/scheduler at epoch 1 of a 100-epoch cosine. Clean, but discards the
   optimizer state and re-runs warmup.
2. **Cold restart at 100 epochs** from scratch: maximally clean single-schedule
   run, costs the 30 epochs already spent.

Selection between these is a training-budget decision, not a correctness one.

## Clean-split implementation audit — 2026-09-15

- Built schema-v3 manifest `405db7a8706367f8d79ebf03d744d43fddb5068e9d76b978911a43d8adc4b8c1`:
  10,161 training-pool images collapse to 9,236 SHA-256 split units (549 duplicate
  groups, 925 redundant slots). Expanded roles are train 7,536, val-select 1,289,
  val-calibration 1,336; no content group crosses those three roles. The frozen
  5,502-image test role is byte-identical to v2 and retains 610 images that share
  content with the training pool, explicitly reported rather than silently moved.
- Rebuilding in validate-only mode reproduced the same fingerprint exactly. The full
  test suite passes (152 tests).
- Diagnostic evaluation of the v2-trained thin checkpoint on v3 val-select produced
  AP50=0.5450 / AP50:95=0.3344 (1,289 images). **This is not a clean baseline:** 350
  v3 val-select images (27.15%) were members of v2 `train_new`, so the checkpoint may
  have trained on them. The number is retained only as evidence that cross-manifest
  evaluation is invalid. A cold control training run on v3 is required before Phase 6/7
  comparisons.
- Completed the Phase-7 multi-scale data path with auditable requested/applied scale.
  Zoom-out uses padding; zoom-in uses a box-preserving feasible crop. This fixes the
  incomplete implementation that ignored factors below 1.0 and could drop boxes without
  synchronising labels. CLI/runtime config remains `--scale-range LOW HIGH`, with
  `(1.0, 1.0)` as the exact control. The same affine map is now applied to normalized
  SVG endpoints, arc centres/radii, and arc direction vectors; previously all geometric
  augmentation (including the pre-existing flips/rotations) moved the raster and boxes
  but left the dual-pathway stroke coordinates in the original frame.
