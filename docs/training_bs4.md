# Training Active: Batch Size 4 (Memory Optimized)

**Date**: 2026-09-05 19:30  
**Status**: Training running with reduced batch size  
**Goal**: Hoàn thành nghiên cứu

---

## Current Training Configuration

**Adjusted for memory constraints**:
- Model: DualStreamCenterNet small (56.5M params)
- **Batch size: 4** (reduced from 8 to avoid OOM)
- Epochs: 100
- Learning rate: 1e-4 with warmup + cosine decay
- Output: `training_bs4.log`

**Trade-off**: Batch size 4 means:
- ✅ More stable (less memory)
- ⚠️ 2× more batches per epoch (3,800 instead of 1,900)
- ⚠️ Longer training time (~48-72 hours instead of 24-36h)

---

## Why This Training Is Critical

**Current state**: DualStreamCenterNet undertrained
- 10 epochs: AP50 0.0004
- Needs 100 epochs to learn sharp heatmap peaks
- This is **blocking** all architecture ablations (H2-H6)

**Expected after 100 epochs**: AP50 0.15-0.18

---

## Monitor

```bash
# Quick check
tail -20 training_bs4.log

# Live watch
tail -f training_bs4.log

# Check for checkpoints
ls -lh checkpoints/dual_stream_100ep/
```

---

## Timeline

**With batch_size=4**:
- Batches per epoch: 3,800 (15,196 samples / 4)
- Estimated time per epoch: ~1-1.5 hours
- Total time: 100-150 hours (4-6 days)

**Checkpoints**:
- Every 5 epochs: Validation
- Every 10 epochs: Save checkpoint
- Diagnostic at epochs 20, 50, 100

---

## Next Steps

1. **Let training run** (4-6 days)
2. **Monitor at epoch 20**: Check if heatmap loss is decreasing
3. **After completion**: Evaluate on val_select
4. **If AP50 >0.15**: Proceed to Phase 2 ablations
5. **Phase 2**: Cross-attention, vector scaling (another 1-2 weeks)

---

## Files

- `training_bs4.log` - Current training log
- `checkpoints/dual_stream_100ep/` - Will contain checkpoints
- `BIG_PICTURE.md` - Full research plan
- `docs/research_status_summary.md` - Summary

---

**Status**: Training running with batch_size=4. Monitor progress, expect 4-6 days to complete.
