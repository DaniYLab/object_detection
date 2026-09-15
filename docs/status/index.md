# Current research status

The active research direction is Drawing Understanding through a Canonical
Drawing Graph. Use these sources in order:

1. [`../research/README.md`](../research/README.md) — live milestone table and gates.
2. [`../research/archcad_benchmark_protocol.md`](../research/archcad_benchmark_protocol.md) — active benchmark contract.
3. [`../experiments/experiment_log.md`](../experiments/experiment_log.md) — authoritative detector measurements.

The September 5–6 real-time training status files were moved to
`docs/archive/superseded_status_20260906/`. They describe historical processes
and must not be interpreted as active jobs or current architecture claims.

As of 2026-09-15, the immediate work is:

- audit all public-40K primitive semantic/instance targets;
- freeze the reconstructed semantic/instance/panoptic evaluator;
- implement the aligned panoptic dataset/collate;
- establish primitive-only, raster-only and dual-pathway baselines;
- defer CDG, room and OCR until the ArchCAD benchmark is reproducible.
