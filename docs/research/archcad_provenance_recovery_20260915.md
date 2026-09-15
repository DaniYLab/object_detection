# ArchCAD source-provenance recovery status

**Audit date:** 2026-09-15

## Decision

The public ArchCAD-40K batch cannot currently be converted into a
source-drawing-disjoint benchmark from fields present in the release.  Keep it
for compiler prototyping and annotation feasibility only.  Do not report
generalization until an authoritative chunk-to-drawing mapping is obtained.

## Local evidence

`scripts/research/audit_archcad_provenance.py` now checks all five archive ID
sets, recursively scans JSON keys, and scans SVG root/element attributes.
Re-running it on the local release found:

- 41,097 UUID-named samples in each modality and identical modality ID sets;
- JSON roots contain only `entities`; neither nested JSON keys nor SVG
  attributes expose a source drawing, project, parent, origin, or chunk group;
- exact-image SHA-256 groups remove duplicate bytes but do not group distinct
  neighboring slices from the same source drawing.

ZIP member order, modification timestamps, and UUID proximity are explicitly
rejected as provenance.  They may describe packaging order but provide no
authoritative drawing boundary.

Generated evidence:
`outputs/drawing_graph/archcad_provenance_audit.json` (schema
`archcad-provenance-audit-v1`).

## Upstream evidence

- Official project page: <https://archiai-lab.github.io/ArchCAD.github.io/>
- Official code: <https://github.com/ArchiAI-LAB/ArchCAD>
- Official public dataset: <https://huggingface.co/datasets/jackluoluo/ArchCAD>

The official dataset card describes each public sample as a 14 m × 14 m slice
and aligns modalities by filename, but does not publish the parent drawing ID.
The official repository publishes DPSS training code and still lists CADParser
release as a TODO.  No ArchCAD chunk mapping or original drawing-level split
manifest was found in the public repository or dataset tree at the audit date.

## Required recovery artifact

Any one of the following is sufficient if supplied by the dataset maintainers:

1. `chunk_uuid, source_drawing_id, project_id, slice_bounds` manifest;
2. original train/validation/test manifest with a documented drawing-disjoint
   guarantee and immutable sample lists;
3. CADParser output retaining source file identity and slice coordinates.

Requested minimum contract:

```text
chunk_id,source_drawing_id,project_id,slice_x0,slice_y0,slice_x1,slice_y1,split
```

## Maintainer request draft

> The public ArchCAD-40K release exposes five UUID-aligned modalities but no
> chunk-to-source-drawing mapping. The paper describes 5,538 source drawings and
> drawing-disjoint evaluation. Could you release either the original split
> manifest or a mapping from each public chunk UUID to its source drawing/project
> and slice bounds? We need this only to prevent neighboring slices from crossing
> train/validation/test boundaries.

Sending this request is an external communication and is not performed by the
repository tooling.

## Fallback

If the upstream mapping cannot be recovered, freeze ArchCAD-40K as
`prototyping_only`.  Build the publication benchmark from source drawings whose
identity is controlled locally, or from a room-layout dataset with explicit
drawing provenance.  Never promote a perceptual-similarity cluster to a true
`source_drawing_id`; such clusters are useful only for leakage diagnostics.
