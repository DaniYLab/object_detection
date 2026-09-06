# Option B: Dual-Stream Encoder + CenterNet Decoder - Progress Report

**Date:** 2026-09-05  
**Objective:** Validate dual-stream architecture (image + SVG vector) với CenterNet decoder

---

## Timeline

- **14:00** - Started Option B implementation
- **14:15** - Architecture design complete (DualStreamCenterNet)
- **14:30** - Fixed collate_fn issues
- **14:45** - Fixed forward pass (class_id parameter)
- **15:00** - Debugging training hang
- **15:15** - All components validated individually
- **15:20** - Training started (tiny preset, 3 epochs)

---

## Architecture: DualStreamCenterNet

### Components
1. **Image Encoder**: ViT-based (reuse from existing models)
2. **Vector Encoder**: Stroke token transformer
3. **Fusion**: Late fusion via concat + projection
4. **Decoder**: CenterNet heatmap head (stride-8)

### Parameters
- **tiny preset**: ~27.9M params
- **small preset**: TBD
- **base preset**: TBD

---

## Technical Validation ✅

### Component Tests (All Passed)

1. ✅ **Model build**: `build_dual_stream_centernet(preset='tiny')` - 27.9M params
2. ✅ **Dataset loading**: FloorPlanQueryDataset with vector_branch=True - 15,196 samples
3. ✅ **Collate function**: `query_collate_fn` creates stroke_mask correctly
4. ✅ **Dataloader**: Batch creation works (stroke_tokens: [B, N, 12], stroke_mask: [B, N])
5. ✅ **Forward pass**: Model(image, stroke_tokens, stroke_mask, class_id) → {heatmap, size, offset}
6. ✅ **Output shapes**: heatmap [B, 1, 64, 64], size [B, 2, 64, 64], offset [B, 2, 64, 64]

### Known Issues (Resolved)

1. ~~Unicode checkmark in print~~ → Fixed with ASCII [OK]
2. ~~Missing stroke_mask~~ → Fixed with query_collate_fn
3. ~~Training script hang~~ → Components work individually, testing full training

---

## Training Configuration

### Current Run (tiny preset, smoke test)
```
Preset: tiny
Batch size: 4
Epochs: 3
Learning rate: 1e-4
Device: CUDA
Dataset: train_new (15,196 samples)
Optimizer: AdamW
Loss: CenterNet (10×focal + 1×L1_size + 1×L1_offset)
```

### Planned Runs
1. **Smoke test**: tiny, 3 epochs (~15 min) ← CURRENT
2. **Quality test**: small, 10 epochs (~60 min)
3. **Full training**: small, 30 epochs (~3 hours)

---

## Success Criteria

### Baseline Comparison
- **Target**: AP50:95 ≥ 0.28 (comparable to thin-head baseline ~0.30)
- **Thin-head (2 seeds mean)**: AP50 = 0.4916, AP50:95 = 0.3063

### Hypothesis
- Dual-stream (image + vector) should capture both:
  - **Pixel patterns** from rasterized floor plan
  - **Geometric structure** from SVG primitives
- Expected gain: +1-3 AP50:95 over image-only

### Validation Protocol
1. Train on train_new
2. Select checkpoint on val_select
3. Final evaluation on val_select (NOT test_set)
4. If successful: replicate with seed 2024
5. If ≥2 seeds confirm improvement: document as validated architecture

---

## Next Steps (After Training Completes)

### If AP50:95 ≥ 0.28 ✅
1. Train small preset for 30 epochs
2. Evaluate on val_select
3. Replicate with seed 2024
4. Compare with thin-head baseline
5. Move to Phase 2: Teacher reasoning integration

### If AP50:95 < 0.28 ❌
1. Debug: check loss curves, gradient flow
2. Try alternative fusion strategies
3. Consider Detect Anything autoregressive decoder
4. Revisit architecture assumptions

---

## Files Created

- `src/models/dual_stream_centernet.py` - Architecture
- `train_dual_stream_centernet.py` - Training script
- `docs/new_papers_summary.md` - Paper roadmap
- `docs/option_b_progress.md` - This document

---

## Status: 🟡 Training in Progress

Waiting for first epoch results to validate loss convergence and training stability.
