# Current research status

The active research direction is Drawing Understanding through a Canonical
Drawing Graph. Use these sources in order:

1. [`../research/README.md`](../research/README.md) — live milestone table and gates.
2. [`../research/drawing_understanding_direction_20260915.md`](../research/drawing_understanding_direction_20260915.md) — architecture and roadmap.
3. [`../experiments/experiment_log.md`](../experiments/experiment_log.md) — authoritative detector measurements.

The September 5–6 real-time training status files were moved to
`docs/archive/superseded_status_20260906/`. They describe historical processes
and must not be interpreted as active jobs or current architecture claims.

As of 2026-09-15, the immediate work is:

- run the bake-off directly on ArchCAD primitive/semantic/instance GT;
- use junction and `incident_to` metrics for HEAT-style structure output;
- recover source-drawing provenance before publication evaluation;
- keep room annotation and RoomFormer-style polygon evaluation as a separate optional extension.
