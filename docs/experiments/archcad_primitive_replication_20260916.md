# ArchCAD primitive-only replication — preregistered 2026-09-16

## Question

Does the small primitive-only query/mask model learn useful semantic and instance
assignments consistently across random initialization and training order, on a
larger fixed development subset?

The first 32/16 pilot matched 10 validation segments: 7 axis/grid stuff and
3 concrete-column things. Thing recall was 3/125; most of the pooled PQ came
from stuff. Loss reduction alone is insufficient to judge instance learning.

## Frozen protocol (written before runs)

- Architecture: `ArchCADPrimitiveBaseline`, 149,343 parameters, dim=64,
  64 queries, 1 encoder layer, 2 decoder layers, no dropout or pretrained weights.
- Input: geometry-only four-point features per entity; no GT label/layer/RGB.
- Data: existing SHA-group public-40K train/val split, fingerprint
  `b0f01e46718cbdef041c5b43a61bebec372acf1852d67d1f640d5772a335118c`.
- Select 256 train / 128 validation images with selection seed 1337 (val 1338).
  Use the identical images across all model seeds. Record IDs and source hashes.
- Eligibility: 1..1024 primitives and 1..64 target segments; report exclusions.
  No silent primitive/target truncation.
- Model/training seeds: 1337, 2024, 4242.
- Budget: 20 epochs, 5,120 optimizer steps per seed; batch=1 image.
- Optimizer: AdamW, fixed LR=0.0003, weight decay=0.01, gradient clip=1, FP32.
- Loss: class CE + mask BCE + Dice, weights 2/7/5; no-object class weight 0.05.
- Evaluation: initialized model and final epoch only; no checkpoint selection
  or threshold tuning. Decoder score threshold=0.1, mask threshold=0.5.
- Metrics: local v1 semantic F1/wF1, primitive-mask AP, pooled total/thing/stuff
  PQ/SQ/RQ and per-class TP/FP/FN. Published-evaluator parity remains unverified.

## Interpretation specified before runs

Report individual runs and mean/std, including instance recall. If losses improve
but thing recall remains low, inspect query classification/mask grouping before
adding fusion or conditioned mechanisms. These small-budget runs are development
controls, not convergence claims or reproductions of full-413K paper tables.

Do not compare the 32/16 pilot and 256/128 results as a controlled architecture
delta; both dataset size and the number of optimizer updates change.

## Reproduction

```powershell
python scripts/research/train_archcad_primitive_pilot.py --output-dir outputs/panoptic/primitive_256_s1337 --train-images 256 --val-images 128 --epochs 20 --seed 1337 --selection-seed 1337
```

Repeat with seeds 2024 and 4242 and separate fresh output directories. The script
refuses to overwrite an existing run.

## Results

Completed under preregistration commit `30d739f`. Model/input implementation
hashes, image IDs, source hashes, optimization budget and decoder settings were
checked identical across runs before aggregation. Scores below are on a 0–1
scale; standard deviation uses n−1 (three seeds).

| Seed | Val loss | Semantic macro F1 | Instance AP50 | Instance mAP | Pooled PQ | Thing recall |
|---|---:|---:|---:|---:|---:|---:|
| 1337 | 7.22942 | 0.09573 | 0.006836 | 0.002637 | 0.11936 | 0.09610 |
| 2024 | 7.35851 | 0.08913 | 0.009260 | 0.004704 | 0.12657 | 0.10245 |
| 4242 | 7.24594 | 0.09250 | 0.006815 | 0.003270 | 0.10396 | 0.09066 |
| Mean | 7.27795 | 0.09245 | 0.007637 | 0.003537 | 0.11663 | 0.09640 |
| Sample std | 0.07025 | 0.00330 | 0.001405 | 0.001059 | 0.01155 | 0.00590 |

Mean thing PQ = 0.09257, mean stuff PQ = 0.18651. Every seed improves over its
initial model but thing recall remains around 9–10%. There are 1,103 thing GT
segments and 349 stuff GT segments in this fixed validation subset. This is a
repeatable weak baseline, not a successful DPSS reproduction or a convergence
result. Eligibility skipped 13 training and 5 validation candidates for the
primitive cap, identically across runs.

Artifacts:

- `outputs/panoptic/primitive_256_s1337/{config.json,report.json,last.pt}`
- `outputs/panoptic/primitive_256_s2024/{config.json,report.json,last.pt}`
- `outputs/panoptic/primitive_256_s4242/{config.json,report.json,last.pt}`
- `outputs/panoptic/primitive_256_replication_summary.json`

Recompute the aggregate using `scripts/research/summarize_archcad_replication.py`.
It rejects mixed data/model/code protocols and duplicate seeds.

## Post-run diagnosis (seed 1337, no tuning)

On the same 128 validation images, compare the maximum raw-query mask IoU for
each thing GT, before disjoint postprocessing:

| Diagnostic | Covered GT / 1,103 |
|---|---:|
| Any query mask, regardless of class | 322 (29.19%) |
| A query mask with the correct predicted class | 134 (12.15%) |
| Correct class and score >=0.1 | 134 (12.15%) |
| Final one-to-one matches from saved evaluation | 106 (9.61%) |

The first three rows are optimistic per-GT coverage, allowing a query to cover
more than one GT; they are not AP or one-to-one recall. CPU and CUDA agree on
these counts. The raw decoder emits 920 thing masks; 718 have positive evaluated
weight after the local ignore/zero-weight policy. The saved evaluator reports
106 TP + 612 FP = 718 scorable thing predictions.

Thus there is evidence of both poor mask coverage and class confusion. Score
threshold removal alone cannot recover the missing raw-query coverage in this
run. Postprocessing also reduces matches, but the much larger losses happen
before it. This does not identify a single architectural cause: data budget,
training duration, features and decoder capacity are still confounded.

Diagnostic report: `outputs/panoptic/query_recall_s1337_cuda.json`.

## Next falsifiable experiment

Keep architecture/data/decoder fixed and extend a prespecified training budget
with full training/validation curves. Determine whether thing recall continues
to improve before attributing failure to the primitive representation or adding
conditioning/fusion. A separate small-set overfit test can isolate optimization
capacity. Do not tune threshold on these validation observations.

## Post-hoc note (2026-09-17): these runs are not bit-reproducible

The three runs above were trained before the training path was made deterministic.
Rerunning the seed-1337 configuration at the same budget and seed does not reproduce
the saved epoch-20 result: thing recall `0.096102 → 0.110607`, pooled PQ
`0.119358 → 0.111647`, i.e. outside any numerical tolerance. The spread across three
identical-protocol seed-1337 runs is 0.014506 in thing recall, comparable to the
sample standard deviation reported below, so `± 0.00590` understates the true
variation and the mean should not be quoted as a reproducible baseline.

Seeded initialization, sample selection and evaluation are reproducible (epoch-0
metrics were bit-identical); the divergence is introduced by training kernels. Two
independent reruns with `--deterministic` are bit-identical to each other. Full
evidence and the protocol deviations are in `archcad_optimization_20260917.md`.
