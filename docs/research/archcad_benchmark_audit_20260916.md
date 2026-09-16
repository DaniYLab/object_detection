# ArchCAD benchmark continuation — 2026-09-16

## Scope and result

The current milestone is a reproducible panoptic baseline on the public ArchCAD
release. Conditioned/unconditioned reflex remains the hypothesis for subsequent
ablations. The eventual written-dimension task remains deferred.

This audit delivers a raw-entity dataset/collate and fixes demonstrated parser
and evaluator defects. Exact equivalence to paper scores is **not established**.
An identity oracle alone does not validate metric definitions.

## Representation: a corrected blocker

The previous report treated decoding `point.zip` as mandatory before any loader
could be built. That conclusion was not supported by the released model code.
DPSS accepts per-primitive features/labels/lengths and image sampling centers.
Its missing data loader prevents exact feature reproduction, but the published
model does not establish a dependency on the auxiliary public `point.zip`.

Sources inspected:

- [ArchCAD model at f72c4b20](https://github.com/ArchiAI-LAB/ArchCAD/blob/f72c4b20aea5dbaf06157f0324749ce68342b734/svgnet/model/svgnet.py)
- [ArchCAD paper, primitive encoding and fusion](https://arxiv.org/html/2503.22346v1#S5)
- [SymPoint primitive sampling parser](https://github.com/nicehuster/SymPoint/blob/master/parse_svg.py)

The local representation samples four geometry points per JSON entity and adds
a four-class primitive-type encoding. It keeps geometry-only features separate
from semantic/instance targets. It is explicitly named
`archcad_entity_four_points_v1`; it is not asserted to reproduce DPSS's missing
10-channel preprocessing.

On the existing seed-1337 sample:

| Check | Result |
|---|---:|
| JSON/SVG element counts | 100/100 chunks agree |
| Semantic IDs by ordinal | 24,369/24,369 agree |
| Instance IDs by ordinal | 24,369/24,369 agree |
| LINE/ARC four-point geometry | 23,744/23,744 agree within declared tolerance |
| LINE/ARC/CIRCLE lengths | 24,359/24,359 agree within declared tolerance |
| Ellipses | 10; excluded from SVG length comparison |
| Public point-record count differs from entities | 100/100 chunks |

For point records with type code 1, 20,747/20,941 lie within 1.5 pixels of a
LINE endpoint when coordinates are interpreted as row/column; only 196 match
under x/y interpretation. This is evidence for quantized control points, not
an entity-order/semantic mapping. No labels are assigned to those points.

Reproduce with `python scripts/research/audit_archcad_representation.py`.
Report: `outputs/panoptic/representation_audit_20260916.json`.

## Parser corrections

- ARC uses signed sweep from `direction`, including crossing 0/360 degrees.
- ELLIPSE length uses its parameter interval, not always the full circumference.
- The elliptic-integral formula handles very eccentric ellipses; a regression
  fixture comes from actual chunk `11ebfe47-7eee-4b3f-b174-c6793fda21c2`.
- Geometry retains direction and ellipse parameters needed to build features.
- Invalid entities and unknown semantic IDs fail explicitly; silently dropping
  an entity would invalidate all ordinal-aligned predictions after it.

Lengths are in native PNG/SVG coordinates. Upstream loader length units remain
unverified: applying `log1p` after a change of units changes weights.

Full-corpus re-audit completed with integration warnings treated as errors:
41,097 chunks / 14,392,530 entities / 500,048 segments; 80 zero-length and
22,389 zero-rounded-weight primitives. The older 22,323 weight count is
superseded. Raw source fingerprint is unchanged. Report:
`outputs/panoptic/archcad_public40k_audit_20260916.json`.

## Evaluator corrections and parity limits

Local v1 excludes GT-ignore positions from semantic and mask weights, rejects
fractional/out-of-range class IDs and nonbinary masks, rejects duplicate image
IDs, and reports zero-weight target/prediction masks as unscorable. These are
explicit local policies. Void-overlap conventions for the paper remain open.

We executed actual evaluator classes from the pinned ArchCAD checkout on shared
non-identity fixtures. Source SHA-256:
`0b398c7787bd0ab51872a62b041daec4fa3546aadc42c703a253dd1ff60a50dc`.

| Fixture (two targets) | Released upstream | Local one-to-one |
|---|---|---|
| Identity | TP=2, PQ≈1 | TP=2, PQ=1 |
| Duplicate one prediction | TP=3, FP=0, PQ≈1 | TP=2, FP=1, PQ=0.8 |
| Wrong class on one target | TP=1, FP=1, FN=0, PQ≈0.667 | TP=1, FP=1, FN=1, PQ=0.5 |
| Missing one target | TP=1, FN=1, PQ≈0.667 | TP=1, FN=1, PQ≈0.667 |

The released matcher can count a target multiple times and omit its false
negative after a wrong-class overlap. Thus local corrected PQ and released PQ
must not be mixed in a comparison table. AP interpolation, F1 aggregation,
length units and full taxonomy correspondence to paper v1 also need independent
verification. The public config's 30 classes differ from paper v1's 27 groups.

Reproduce:

```powershell
python scripts/research/compare_archcad_upstream_evaluator.py --upstream-root PATH_TO_CHECKOUT
```

The report pins the executed commit and source hash. Upstream code is not copied
into the repository. Outputs are in `outputs/panoptic/`.

## Dataset/collate contract now implemented

```text
Frozen public-40K split → full PNG + raw JSON entities
→ entity_indices[N] + geometry features[N,12] + centers[N,2]
→ semantic[N] + instance[N] + class[K] + masks[K,N] + lengths[N]
→ packed features with cumulative offsets + per-image target masks
```

The loader verifies the existing split fingerprint and disjoint image IDs,
retains source hashes, and raises on an explicit primitive budget rather than
silently truncating labels. It preserves native image size and performs no
augmentation yet. It does not assert source-drawing-disjointness from image IDs.

Real-data smoke: 10 validation images, 3,212 primitives, finite features, native
980×980 raster batch. Tests change labels while holding geometry fixed and
assert identical model inputs, and verify masks stay local to each image.

## First learned pilot completed

`scripts/research/train_archcad_primitive_pilot.py` connects the new loader to
`ArchCADPrimitiveBaseline`: geometry MLP, one transformer encoder layer, two
query decoder layers, 64 queries, class logits and primitive-mask logits.
149,343 parameters, random initialization, geometry only. Loss uses Hungarian
class/BCE/Dice matching with weights 2/7/5 and no-object weight 0.05.

Frozen pilot conditions: seed 1337, 32 train images / 16 validation images,
20 epochs (640 optimizer steps), AdamW LR 0.0003, weight decay 0.01, FP32,
one image per step, gradient clipping 1.0. Selection caps are 1,024 primitives
and 64 target segments; two train candidates and one validation candidate were
excluded for primitive budget. Train/val PNG hashes were checked disjoint.

The run records selected IDs, raw JSON hashes, split fingerprint, decoder
settings implied by the versioned runner, config, initial/final evaluation,
history and final checkpoint. Run directory:
`outputs/panoptic/primitive_pilot_20260916_s1337/`.

| Metric | Train initial | Train final | Val initial | Val final |
|---|---:|---:|---:|---:|
| Loss (eval mode) | 14.85088 | 6.43248 | 14.84317 | 8.44854 |
| Semantic macro F1 | 0.00120 | 0.09857 | 0 | 0.09362 |
| Weighted semantic macro F1 | 0.00096 | 0.09813 | 0 | 0.09610 |
| Instance AP50 | 0 | 0.00638 | 0 | 0.000787 |
| Instance mAP | 0 | 0.00304 | 0 | 0.000475 |
| Local pooled PQ | 0 | 0.09897 | 0 | 0.08073 |

Validation matching: TP=10, FP=29, FN=154. The full optimization/evaluation
portion took 8.00 seconds on an RTX 5070 Ti, excluding sample-loading startup.
This is a small learning feasibility experiment; most segments remain missed.
It establishes neither convergence nor a gain over DPSS/other baselines.

Reproduce in a fresh output directory:

```powershell
python scripts/research/train_archcad_primitive_pilot.py --output-dir outputs/panoptic/NEW_RUN --seed 1337 --train-images 32 --val-images 16 --epochs 20
```

## Next experiment

Expand this primitive-only control on the same explicit local v1 protocol and
diagnose instance recall; then compare raster/fusion and conditioned decoding
under equal data and budgets. Missing full-413K data and drawing split restrict
paper-table reproduction, while controlled public-40K experiments can proceed.

## Verification

Full suite: 218 tests passed. Includes non-identity metric counterexamples,
curve wraparound/eccentricity, label-leak checks, ragged batching, Hungarian
target-order invariance and loss backpropagation. Oracle still scores 1.0
within floating-point tolerance on its 100 scorable samples. This test count
does not certify equivalence to the published benchmark.
