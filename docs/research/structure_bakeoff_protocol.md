# ArchCAD direct-ground-truth bake-off protocol v1

## Scope

The primary bake-off uses ground truth already present in ArchCAD:

- exact primitive geometry (`LINE`, `ARC`, `CIRCLE`, `ELLIPSE`);
- primitive `semantic` IDs;
- shared `instance` IDs and instance boxes derived from their primitives;
- endpoint junctions and `incident_to` edges deterministically compiled from
  annotated primitive endpoints;
- exact semantic counts executable from the graph.

It does **not** require the room annotation UI. Room polygons, containment,
adjacency, and `connected_by` are not explicit fields in the public JSON and are
therefore excluded from the primary scoreboard.

## Evaluation status

The bake-off can run immediately as an internal feasibility comparison. Missing
`source_drawing_id` still prevents publication/generalization claims because
neighboring chunks from one source drawing cannot be guaranteed split-disjoint.
It does not invalidate within-release direct-GT scoring.

## Common prediction contract

Every method receives the same full 980×980 raster and the declared vector
inputs for its arm. It emits one CDG JSON per drawing with any supported subset
of:

1. primitive nodes with geometry and predicted `semantic_id`;
2. instance nodes with class and `bbox` geometry;
3. junction nodes with point geometry;
4. `incident_to` edges from primitives to junctions.

No method receives GT crops, GT instance IDs, candidate room hints, or compiler
outputs at inference.

## Arms

| Arm | Role | Direct-GT outputs |
|---|---|---|
| `archcad_raw_oracle` | evaluator sanity ceiling only | all fields, expected 1.0 |
| raster detector baseline | learned control | instance bbox/class |
| raster+vector DPSS-style | main perception baseline | primitive semantic + instance |
| HEAT-style graph parser | structure baseline | junction + incidence/edge graph |

RoomFormer-style polygon decoding is deferred from this scoreboard because the
public ArchCAD JSON has no room-polygon GT. It may be evaluated later on a
separate dataset with explicit room polygons; derived wall cycles are not
promoted to GT.

## Pre-registered metrics

1. primitive geometry precision/recall/F1 at 2 px feature tolerance;
2. primitive semantic precision/recall/F1 and semantic accuracy conditioned on
   a geometry match;
3. per-semantic exact-count fraction;
4. class-aware instance precision/recall/F1 at bbox IoU 0.50;
5. junction precision/recall/F1 at 2 px;
6. `incident_to` edge precision/recall/F1 after node matching;
7. CDG validity rate, runtime and peak memory.

Instance nodes whose derived bbox has zero width or height are reported as
`unscorable_*_instance_count` and excluded from bbox F1. Their primitive
geometry/semantic remains fully scored; a zero-area box has no defined IoU.

Report macro metrics and per-drawing failures. The task-selection manifest
selects a representative subset but is explicitly not an evaluation split.

## Executable sanity run

```bash
python scripts/research/build_structure_baseline_predictions.py \
  --subset floor_plan_reading_candidate \
  --sample-count 100

python scripts/research/evaluate_structure_bakeoff.py \
  --prediction oracle=outputs/drawing_graph/predictions_archcad_oracle_v0
```

The oracle run verifies evaluator correctness. Learned prediction directories
are added through repeated `--prediction NAME=DIR` arguments.

## Decision

Use direct-GT metrics for model development now. Do not rank a model on
room/connectivity candidate metrics. For publication, first recover the
chunk-to-source-drawing mapping and freeze a drawing-disjoint split.
