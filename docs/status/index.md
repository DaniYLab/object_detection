# Current research status

The active research milestone is ArchCAD primitive-level panoptic symbol
spotting, followed by conditioned/unconditioned ablations. Use these sources in order:

1. [`../research/README.md`](../research/README.md) — live milestone table and gates.
2. [`../research/archcad_benchmark_protocol.md`](../research/archcad_benchmark_protocol.md) — active benchmark contract.
3. [`../experiments/experiment_log.md`](../experiments/experiment_log.md) — authoritative detector measurements.

The September 5–6 real-time training status files were moved to
`docs/archive/superseded_status_20260906/`. They describe historical processes
and must not be interpreted as active jobs or current architecture claims.

As of 2026-09-16, the immediate work is:

- expand the completed primitive-only pilot on the fixed public-40K split;
- use the explicit local v1 metric profile; exact paper parity is unverified;
- diagnose instance recall and profile large-primitive chunks;
- establish raster-only and dual-pathway controls after the primitive baseline;
- defer CDG, room and OCR until the ArchCAD benchmark is reproducible.
