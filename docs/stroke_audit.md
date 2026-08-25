# SVG Stroke Audit — Dual-Pathway Design Input

**Script:** `scripts/data/audit_strokes.py`
**Outputs:** `outputs/stroke_audit_train.json`, `outputs/stroke_audit_test.json`, `outputs/stroke_audit_pilot.json`
**Date:** 2026-08-24

Audit of raw SVG path geometry across all 15,663 FloorPlanCAD drawings, run to
inform the Dual-Pathway (Image + Vector) architecture brainstorm. Answered
three design questions:

## 1. Which path commands exist?

**Only `M`, `L`, `A` — verified on 100% of drawings, 0 parse failures.**

No Bezier (`C`/`Q`/`S`/`T`), no `H`/`V`, no unsupported commands. The
primitive tokenizer only needs two types: Line and Arc.

## 2. One primitive per path element

`M` count == `L` + `A` count on every split (train: 7,133,394 M vs
6,279,041 L + 854,353 A). Every SVG path element contains exactly one
primitive: `M → (L|A)`. Consequences:

- **No polyline grouping needed** — each `<path>` element is already one token.
- Arc share is 11–12% of primitives — a two-type TypeEmbedding is sufficient.

## 3. Primitives per drawing (drives N_max)

| Split | Elements/SVG median | mean | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| Train (10,161) | 363 | 702 | 1,451 | 2,326 | 5,310 | 27,275 |
| Test (5,502) | 333 | 622 | 1,293 | 2,044 | 4,933 | 19,190 |

N_max coverage (drawings fully covered, no truncation):

| N_max | Train | Test |
|---:|---:|---:|
| 512 | 36% | 39% |
| 768 | 52% | 56% |
| **1024** | **62%** | **65%** |
| 1536 | 75% | 78% |
| 2048 | 84% | 86% |

## Design conclusions

1. **Token layout (superset, 12 dims)** is valid:
   `[x0, y0, x1, y1, cx, cy, r, cosθ0, sinθ0, cosθ1, sinθ1, large_arc_flag]`
   with zeroed unused slots per type + a 2-entry TypeEmbedding. Both primitives
   share the two endpoints, so the first 4 dims are the common backbone.
2. **N_max = 1024** recommended: ~63% coverage, and the tail (p99 ≈ 10K,
   max 54K — likely hatch/dot patterns) is noise-prone anyway. For drawings
   exceeding N_max, **random-sample 1024 primitives per epoch** rather than a
   fixed truncation — the model sees different primitives of the same drawing
   each epoch, a free augmentation that avoids a deterministic information cut.
3. **Compute is cheap relative to the image branch**: mean ≈ 700 tokens →
   vector attention ≈ 0.49M pairs vs 4,096² ≈ 16.7M pairs for the stride-8
   image grid (34× smaller).
4. **Label-leak guard**: tokens must contain pure geometry only — never
   `semantic-id` or `instance-id` (both are ground-truth labels read by
   `parse_svg_metadata` and must not reach the encoder).
