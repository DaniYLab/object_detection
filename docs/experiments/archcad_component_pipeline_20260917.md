# ArchCAD: the symbol is the right unit, and a learning-free pipeline beats every trained model — 2026-09-17

Three read-only probes, run after the semantic head on the full split reached pooled PQ
0.246188. They redirect the program.

## Probe 11 — the raw data has no appearance channel

Before building anything on appearance, this checked whether the styling fields that the
dataset deliberately drops ("no semantic, instance, layer or RGB priors") actually carry
information. Over 11,886 sampled validation entities:

| field | values |
|---|---|
| `rgb` | `[0, 0, 0]` for **100%** of entities |
| `linetype` | `Continuous` (10,253) or `CONTINUOUS` (1,617) — the same value, differing case |
| `line_width` | 0.25 (11,709) or 0.18 (177) |

Every drawing is black continuous lines at uniform width. **There is no appearance channel
in the vector data**, so a steel column and a concrete column are both black rectangles, and
no styling or raster feature can separate them. The plan's raster branch loses its
grouping motivation and most of its semantic one.

## Probe 12 — a primitive's own shape carries almost no class information

kNN(k=5) over each primitive's centroid and extent, fitted on 1,024 training images and
evaluated on 512 validation images:

| predictor | accuracy |
|---|---:|
| constant "wall" | 21.60% |
| **kNN on the primitive's own geometry** | **19.78%** |
| trained per-primitive semantic head (27,272 images) | 48.44% |

The non-parametric classifier is *below* the majority-class baseline. A single line does not
say what it is; the head's 48.44% therefore comes from context, not from the primitive.

Classes geometry cannot recover at all (kNN below 15%, all with ≥200 primitives):
`squat_toilet` 0.0%, `table` 1.3%, `bed` 1.3%, `rebar` 1.4%, `steel_column` 1.8%, `drain`
1.9%, `fire_hydrant` 2.6%, `elevator` 2.8%, `steel_beam` 2.9%, `foundation` 3.1%,
`other_fixtures` 3.6%, `bathtub` 3.9%, `other_door` 5.0%, `glass` 6.4%, `hole` 7.1%.

## Probe 13 — at component level the same data classifies far better

Sixteen aggregate features per component — primitive count, bounding extent, aspect, spread,
total and mean length, endpoint-sharing ratio, type composition, centroid — with components
defined two ways, classified by the same kNN:

| components defined by | components | accuracy |
|---|---:|---:|
| ground-truth instances (upper bound) | 5,475 | **70.78%** |
| the geometric grouper's own components (realistic) | 6,133 | **64.57%** |

Against 48.44% for the trained per-primitive head. Per-class, with ground-truth components:
`axis_and_grid` 93.6%, `pile` 92.1%, `single_door` 87.3%, `concrete_column` 85.3%,
`parking_space` 84.5%, `hole` 60.0%, `wall` 59.9%.

## Probe 14 — the whole pipeline, with no gradient training at all

Geometric grouping, then kNN classification of each component, through the frozen evaluator
on 512 validation images:

| configuration | pooled PQ | thing PQ | thing SQ | thing RQ |
|---|---:|---:|---:|---:|
| **grouper + kNN components (no training)** | **0.563245** | 0.538142 | **0.980859** | 0.548643 |
| semantic head on 27,272 images (previous best) | 0.246187 | 0.240317 | 0.841433 | 0.285604 |
| oracle semantics with the same grouper | 0.807876 | 0.748993 | 0.969945 | 0.794937 |

**2.3x the previous best, and 70% of the way to the oracle ceiling**, with no queries, no
mask head, no Hungarian matching and no gradient step. Component accuracy 58.26% over the
components whose reference is a single class.

Segmentation quality is essentially solved: SQ 0.9809 against the oracle's 0.9699. The
remaining loss is RQ 0.5486, and the gap from there to the oracle's 0.7949 is component
classification: perfect components would give 0.808.

Per-class component accuracy shows where that remains: `elevator` 5.2%, `other_door` 12.8%,
`parking_space` 19.8%, `staircase` 25.2%, `double_door` 42.7%, `glass` 44.3%, `hole` 52.5%,
`steel_column` 54.5%, `wall` 59.9%, against `axis_and_grid` 93.6%, `pile` 90.9%,
`single_door` 84.8%, `concrete_column` 81.6%.

## What this means

Every architecture this session started from — the DETR-style query/mask set predictor, and
then the per-primitive semantic head — classifies at the wrong granularity. A symbol in these
drawings is a group of primitives, its class is a property of the group's aggregate shape,
and a nearest-neighbour lookup over sixteen aggregate numbers beats a trained transformer
that has to infer the same thing one line at a time.

The two earlier measurements now fit together rather than competing: the geometric grouper
was shown to be good (oracle-semantics PQ 0.808), and classification was shown to carry the
gap (92.9% of it). The probes here say the fix is not a better classifier over primitives but
a classifier over components.

## Next

The learning-free pipeline is a floor, not a ceiling: it uses sixteen hand-written features
and an unweighted kNN vote. A trained component classifier over richer features — and with
cross-component context, since context is demonstrably what raises the primitive-level head
above its own kNN — is the obvious next experiment, and components are few enough (about
twelve per drawing) that it costs minutes rather than hours.

## Artifacts

- `outputs/panoptic/_runs_20260917/inspect_raw_entity.py` (probe 11)
- `outputs/panoptic/_runs_20260917/probe_geometry_separability.py` (probe 12)
- `outputs/panoptic/_runs_20260917/probe_component_classification.py` (probe 13)
- `outputs/panoptic/_runs_20260917/probe_learning_free_pipeline.py` (probe 14)
