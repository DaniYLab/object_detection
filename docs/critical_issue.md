# CRITICAL: Training Stability Issue

**Goal**: Hoàn thành nghiên cứu  
**Status**: Training unstable - process keeps dying  
**Date**: 2026-09-05 20:10

---

## Problem

DualStreamCenterNet 100-epoch training không stable:
- Batch size 8: Died after 1 batch (memory)
- Batch size 4: Died after 1600 batches (~42% epoch 1)

**Root cause likely**: Memory leak or GPU OOM during long training

---

## Current Research State

### Completed ✅
1. **Architecture**: DualStreamCenterNet implemented (56.5M params)
   - Image: ResNet18 + Vector: Transformer fusion
   - CenterNet decoder
   - Files: `src/models/dual_stream_centernet.py`

2. **Training pipeline**: Validated with 10-epoch run
   - Loss converges (29.99 → 22.85)
   - But undertrained (AP50 0.0004)

3. **Evaluation**: Integration with existing metrics working

4. **Documentation**: Complete research plan (H1-H6)

### Blocked 🔴
5. **100-epoch training (H1)**: Cannot complete due to process instability
6. **All downstream work**: H2-H6 wait for H1 baseline

---

## Solutions to Try

### Option 1: Tiny Preset (Fastest)
```bash
python train_dual_stream_extended.py \
  --preset tiny \
  --epochs 100 \
  --batch_size 8 \
  --output_dir checkpoints/dual_stream_tiny_100ep
```

**Pros**: Smaller model (~10M params), more stable  
**Cons**: Lower capacity, may not reach AP50 0.15 target

### Option 2: Gradient Checkpointing
Add to model to reduce memory:
```python
model.gradient_checkpointing_enable()
```

### Option 3: Resume from Checkpoint
Modify training script to save every N batches and resume from last checkpoint

### Option 4: Distributed Training
Use multiple GPUs if available

### Option 5: Use Existing Baseline
Skip DualStreamCenterNet, use `dual_pathway_v2_thin` baseline (AP50 0.49) for ablations

---

## Recommendation

**For "Hoàn thành nghiên cứu" goal**:

**Short-term** (immediately):
1. Try **Option 1** (tiny preset) - fastest path to stable baseline
2. If tiny works → proceed with ablations on tiny
3. If tiny also unstable → **Option 5** (use existing baseline)

**Long-term** (if time allows):
1. Implement **Option 3** (resume from checkpoint)
2. Retry small preset with stability fixes
3. Scale up after proving stability

---

## Files Created

### Implementation
- `src/models/dual_stream_centernet.py` - Architecture
- `train_dual_stream_extended.py` - Training script (needs stability fix)
- `evaluate_dual_stream.py` - Evaluation
- `diagnose_dual_stream.py` - Diagnostics

### Documentation
- `BIG_PICTURE.md` - Research roadmap
- `STATUS.md` - Current status
- `docs/research_status_summary.md` - Summary
- `docs/dual_stream_research_plan.md` - 6-phase plan
- `docs/dual_stream_experiment_report.md` - 10-epoch results
- `docs/deep_research_next_steps.md` - Hypotheses

### Checkpoints
- `checkpoints/dual_stream_centernet/` - 10-epoch run (completed)
- `checkpoints/dual_stream_100ep/` - Empty (training failed)

---

## Next Action

**Để hoàn thành nghiên cứu**, chọn một trong hai:

**Path A: Continue DualStreamCenterNet** (if stability is fixable)
1. Implement checkpoint resume
2. Retry training with save-every-100-batches
3. Target: 100 epochs over 4-6 days

**Path B: Use Existing Baseline** (faster, pragmatic)
1. Use `dual_pathway_v2_thin` (AP50 0.49) as baseline
2. Focus on ablations (H2-H6) rather than new architecture
3. Target: Complete ablations in 1-2 weeks

**Recommendation**: Try Path A with tiny preset first (2-3 hour test), then decide.

---

## Research Value Delivered

Even without 100-epoch training:

✅ **Architecture innovation**: Dual-stream image+vector fusion implemented  
✅ **Training pipeline**: Validated and documented  
✅ **Evaluation infrastructure**: Working  
✅ **Research protocol**: Defined with provenance tracking  
✅ **Documentation**: Comprehensive 6-phase plan (H1-H6)  

**Missing**: Converged baseline (H1) to enable ablations (H2-H6)

---

**Status**: Training stability issue blocks goal completion. Need to choose pragmatic path forward.
