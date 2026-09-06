# Dual-Pathway (Image + Vector) — Implementation Spec

**Status:** design approved after `docs/stroke_audit.md`; implementation starts
after the last replication run (B seed 4242) finishes — the no-src-edits-while-
training rule forbids touching `src/` while a `num_workers=6` job runs.

**Goal:** exploit the SVG vector geometry that `parse_path_bbox` currently
discards. New preset `dual_pathway`, params-matched against `floorplan_base`,
evaluated with the same protocol (2+ seeds, val AP50/AP50:95, test only after
freezing).

## Data flow

```
SVG <path d="M... L..."> ──parse──> stroke tokens [N_i, 12]
                                                  │ (dataset, collate → [B, N_max, 12] + mask)
                                                  ▼
PNG ── ConvImageEncoder ──> image_tokens ── cross_attention (Q=image, K/V=vector)
                                                  ▼
                                     pathway + FiLM conditioning (unchanged)
                                                  ▼
                                     heatmap / size / offset heads (unchanged)
```

## 1. Metadata schema v3 — `src/data/metadata.py`

Add `extract_strokes(path_data, viewbox, image_size) -> list[list[float]]` next
to `parse_path_bbox`; `parse_svg_metadata` gains kwarg `include_strokes=False`.
When enabled, each emitted instance dict gains:

```json
"strokes": [[x0,y0,x1,y1, cx,cy,r, cos0,sin0, cos1,sin1, flag, type_id], ...]
```

- 12-dim superset layout (both primitives share endpoints, see stroke audit):
  - dims 0–3: `(x0, y0, x1, y1)` endpoints, **pixels, normalized to [0,1]** by
    the PNG size (same frame as the raster branch after resize).
  - Line: dims 4–11 = 0.
  - Arc: dims 4–6 = center + radius; dims 7–10 = `(cosθ0, sinθ0, cosθ1, sinθ1)`
    (never raw angles — wrap-around continuity); dim 11 = large-arc flag.
- `type_id`: 0 = line, 1 = arc (consumed by TypeEmbedding, not by the linear
  projection).
- Arc params come from `svgpathtools` `Arc` objects (already a dependency);
  the dependency-free fallback stores endpoints only and zeros the arc slots.
- **Never store `semantic-id` / `instance-id` in strokes** (label leak).
- Schema version bump 2 → 3; `generate_metadata.py` regenerates `_meta.json`
  for all splits; manifest fingerprint changes → document in experiment_log.
  Backward compat: schema v3 readers must accept v2 files without strokes.

## 2. Dataset + collate — `src/data/dataset.py`, new `src/data/strokes.py`

`src/data/strokes.py` (new module, nothing else imports it during training):

- `StrokeTokenizer`: stacks instance strokes → `[N_i, 12]` float tensor.
- `sample_strokes(tokens, n_max, generator)`: if `N_i > n_max`, **uniformly
  random-sample `n_max` without replacement** (per-epoch stochasticity — free
  augmentation; deterministic at eval via fixed seed).
- `collate_strokes`: pads to `[B, N_max, 12]` with zeros + returns
  `key_padding_mask [B, N_max]` (True = pad).

`FloorPlanQueryDataset` gains `vector_mode` off/instance:
`__getitem__` returns `(image, targets, strokes, stroke_mask)` when on.
Instance-level strokes (not whole-drawing) keep the query-conditioned design:
each sample's vector context is the strokes of the queried instance's SVG.

**Open question to resolve at implementation:** whole-drawing strokes (all
N primitives of the SVG, the "what exists where" context) vs instance strokes
(the object's own shape). The brainstorm favors the fusion semantics "which
vector strokes exist at this image region" → **whole-drawing** is the right
first experiment; instance strokes would leak the bbox extent at train time
only. Decision: whole-drawing, N_max=1024 random-sample.

## 3. Vector encoder — new `src/models/vector_encoder.py`

- `Linear(12 → model_dim)` + `TypeEmbedding(2, model_dim)`.
- 2 × `SelfAttention` + FFN blocks reused from `object_learning_block.py`
  (same module class as the image pathway), each with `key_padding_mask`.
- Output: `[B, N_max, model_dim]`.

## 4. Fusion — `src/models/detector.py`

`FloorPlanDetector.__init__` gains `use_vector_branch: bool`. When on:

- `vector_encoder` runs before `EarlyFusion`;
- new fusion mode `"vector_cross_attention"`: image tokens (flattened
  stride-8 grid) = Query, vector tokens = Key/Value → reshape back to
  `[B, model_dim, H/8, W/8]`;
- existing FiLM class conditioning applies after vector fusion (both condition
  on class and on geometry);
- all-class chunked inference unchanged (vector branch computed once per image).

## 5. Presets + params matching — `src/models/config.py`

`dual_pathway`: `floorplan_base` config + vector branch. Width/depth of the
vector encoder tuned so total params ≤ `floorplan_base` + 10% (budget rule
from `test_presets.py`); record exact count in experiment_log. Control preset
`dual_no_fusion` (vector encoder present, fusion=none) isolates the fusion
contribution from the extra capacity.

## 6. Training + evaluation

- `train.py`: `--vector-branch` flag (off by default; checkpoints record it in
  runtime_config — exact-resume compatibility).
- `evaluate.py`: auto-detects from checkpoint manifest; decoder unchanged.
- Runs: `dual_pathway` seeds 1337 + 2024, compare to B same seeds.
  Success condition: ΔAP50 vs B > 0 on ≥2 seeds, ideally without widening the
  params gap. Test only after val-freezing.

## 7. Claim discipline

The PNG is a render of the SVG, so vector input is legitimate at inference,
but the paper must state that the gain comes from **an additional information
source**, not architecture alone — the `dual_no_fusion` control and the
params-matched B comparison carry that argument.
