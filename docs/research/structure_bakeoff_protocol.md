# Structural parser bake-off protocol v0

## Gate

This experiment remains blocked until the primary room pack contains at least
100 reviewed floor plans and the 40-record blinded secondary pack has measured
agreement. ArchCAD publication claims additionally require source-drawing
provenance.

## Common contract

Every method receives the same full 980×980 raster and the same allowed vector
primitives. Every prediction is serialized as one CDG JSON per drawing. Models
must not receive human regions, candidate hints, or ground-truth crops.

The initial arms are:

| Arm | Input | Output adapter |
|---|---|---|
| `endpoint_cycles_v0` | raw ArchCAD primitives | current half-edge face compiler |
| `raster_free_space_v0` | wall/door primitives | current barrier/component compiler |
| `heat_style_v0` | raster, optional primitives | corners + edges converted to CDG regions |
| `roomformer_style_v0` | raster, optional primitives | polygon queries converted to CDG regions |

The learned arms do not yet exist. Their implementation starts only after the
gold and provenance gates; otherwise model selection would optimize against
unverified pseudo-labels.

## Metrics

Pre-register:

1. region precision/recall/F1 at polygon IoU 0.50;
2. mean matched polygon IoU;
3. `connected_by` precision/recall/F1, including door-instance identity;
4. CDG validity rate;
5. runtime and peak memory, reported separately from quality.

Report macro metrics and per-drawing failures. Do not tune on the double-review
subset. Candidate relations are predictions, never gold.

## Executable workflow

After gold review:

```bash
python scripts/research/compile_room_gold_graphs.py

python scripts/research/build_structure_baseline_predictions.py \
  --method endpoint_cycles
python scripts/research/build_structure_baseline_predictions.py \
  --method raster_free_space

python scripts/research/evaluate_structure_bakeoff.py \
  --prediction endpoint=outputs/drawing_graph/predictions_endpoint_cycles_v0 \
  --prediction raster=outputs/drawing_graph/predictions_raster_free_space_v0
```

HEAT-style and RoomFormer-style adapters must emit the same `*.cdg.json`
contract and can then be added as repeated `--prediction NAME=DIR` arguments.

## Decision

Proceed to multitask training only if a learned arm improves region and
connectivity F1 over both deterministic baselines without reducing graph
validity. If no learned arm wins, repair target semantics/annotation before
increasing model scale.
