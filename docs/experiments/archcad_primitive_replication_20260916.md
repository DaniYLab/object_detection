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

Pending at preregistration. Populate from saved reports after all three runs.
