# ArchCAD: no appearance channel, no per-primitive shape signal, and a circularity that blocks the component route — 2026-09-17

Four read-only probes, run after the semantic head on the full split reached pooled PQ
0.246188. Two of them close off directions, and the fourth corrects an earlier reading in
this same document.

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

## Probe 14 — the whole pipeline, and the correction that changes its reading

Geometric grouping, then kNN classification of each component, through the frozen evaluator
on 512 validation images:

| grouping | components | pooled PQ | thing RQ | component accuracy |
|---|---:|---:|---:|---:|
| **GT semantics for grouping (oracle, not deployable)** | 6,133 | **0.563245** | 0.548643 | 58.26% |
| the semantic head's predictions for grouping (bootstrap) | 12,263 | **0.211181** | 0.228637 | 28.49% |
| class-agnostic grouping, no semantics needed | 11,023 | **0.134469** | 0.185736 | 25.04% |

| reference | pooled PQ |
|---|---:|
| semantic head on 27,272 images (best deployable result) | **0.246187** |
| oracle semantics with the same grouper | 0.807876 |

### Correction: the 0.563245 row is oracle-leaked

`group_instances` groups *within each semantic class*, so every row above except the last
was fed ground-truth per-primitive labels. The first row therefore reads "given perfect
per-primitive semantics, geometry plus a kNN reaches 0.563"; it is not a deployable
pipeline and it must not be compared with a trained model.

An earlier version of this document reported that row as if it beat the trained model by
2.3x and called segmentation essentially solved. That reading was wrong. It was caught by
running the deployable variants, which fall *below* the semantic head: 0.211 with the head's
own predicted semantics for grouping and 0.134 with a class-agnostic grouper.

The same caveat applies to probe 13's 64.57%: those components were also formed with
ground-truth semantics. Only the class-agnostic row here is free of the leak.

### What actually survives, and why it matters

The conditional ceiling is real and high — 0.563 given perfect primitives classes, against
0.808 for perfect classes *and* perfect component classification — so the component-level
idea is not wrong. What prevents using it is a **circular dependency**:

```
per-primitive semantics  ->  class-conditional grouping  ->  component classification
        ^                                                             |
        +-------------------------------------------------------------+
```

Every arrow is real and none can be cut. Perfect semantics make components nearly ideal
(SQ 0.981 against the oracle's 0.970) and component classification accurate (58.26%); the
head's 48.44% semantics produce 12,263 components instead of 6,133, and the result drops
below the direct per-primitive route.

This also explains, after the fact, why the query/mask architecture was competitive at all:
it is a joint model of semantics and instances, which is the shape the problem actually has.
It was beaten by the per-primitive head not because joint modelling is wrong but because
that particular joint model, a DETR-style set predictor with a dot-product mask head, is a
poor fit for hundreds of small repetitive symbols.

## What this means

Two facts stand after the correction:

1. The raw data has **no appearance channel** (probe 11), so the raster branch cannot
   separate classes that geometry cannot separate, and the plan's step 5 loses its main
   motivation.
2. A primitive's own shape carries almost no class information (probe 12): kNN on it scores
   19.78%, below a constant "wall". The per-primitive head reaches 48.44%, so its accuracy
   comes from **context**, not from its input.

The remaining route that the measurements actually support is improving per-primitive
semantics with more context and capacity — not regrouping, and not appearance. The best
deployable result is still the semantic head at 0.246187, and the ceiling for perfect
semantics through the existing grouper is 0.807876.

## Next

The task is a **joint** one and the session has now measured why: semantics and grouping
depend on each other. The next build should be trained end to end on both, but at the
component granularity rather than the primitive one, and with context that the primitive
head demonstrably relies on — for example a model that predicts per-primitive embeddings and
a component assignment jointly, trained against the panoptic metric through the grouper.
Before building it, one cheap question is worth answering: how much of the per-primitive
head's 48.44% is context, i.e. how a per-primitive classifier scores with no context at all
beyond its own features. Probe 12 already gives the floor (19.78%) and the head gives the
ceiling (48.44%), so the value of context is bounded at 29 points and a no-context trained
classifier would localise it further.

## Artifacts

- `outputs/panoptic/_runs_20260917/inspect_raw_entity.py` (probe 11)
- `outputs/panoptic/_runs_20260917/probe_geometry_separability.py` (probe 12)
- `outputs/panoptic/_runs_20260917/probe_component_classification.py` (probe 13, GT-semantics caveat applies)
- `outputs/panoptic/_runs_20260917/probe_learning_free_pipeline.py` (probe 14, GT-semantics arm)
- `outputs/panoptic/_runs_20260917/probe_deployable_pipeline.py` (probe 14 corrected, all three arms)

