# Ground Truth: Tình Huống Thực Tế (2026-09-06 00:45)

**Mục đích**: Ghi lại sự thật về tình trạng training hiện tại, loại bỏ confusion từ các file outdated

---

## ✅ SỰ THẬT THỰC TẾ (VERIFIED)

### Training Status: ĐANG CHẠY TỐT ✅

**Checked at**: 2026-09-06 00:45  
**Log file**: `training_bs8_workers4.log`  
**Process**: Running (PID 3755)

```
Current: Epoch 17/100, Batch 1300/1900
Loss: 16.26 (oscillating 12-19)
Heatmap loss: 1.49 (target <1.0 for good convergence)
Status: ✅ STABLE, no crashes
```

**Key Metrics**:
- ✅ Process hasn't died (unlike earlier attempts)
- ✅ Loss decreasing overall trend (49 → 17 → ~16)
- ⚠️ Heatmap loss still >1.0 (needs to reach <1.0)
- ⚠️ Loss oscillating (12-19 range) but stable

**ETA**: 
- Epochs remaining: 83
- Time per epoch: ~82 min (based on current progress)
- **Total ETA**: ~114 hours = **4.75 days**
- Expected completion: **Monday Sept 9, ~05:30**

---

## 🏗️ Architecture: FloorPlanLLM (ViT-based)

**Actual Architecture Running** (verified from git commits bb14bf8, 68d01af):

```python
Image Branch:
  - ViT-8L-512D (Vision Transformer, 8 layers, 512 dim, 8 heads)
  - Parameters: ~40M
  - Output: 512-dim features per patch
  - Stride: 8

Vector Branch:
  - Graph Transformer-4L-512D-8H
  - Parameters: ~15M
  - Input: SVG strokes + graph edges (1024 tokens)
  - Output: 512-dim features

Fusion:
  - Cross-attention between image patches and vector tokens

Decoder:
  - CenterNet heads (35 classes)
  - Heatmap + size + offset prediction
  - Parameters: ~15M

Total: 56.5M parameters
```

**NOT ResNet18** (despite what older docs say)

---

## 📊 Training Configuration (WORKING)

**What finally worked**:
```bash
Batch size: 8
Num workers: 4 (training), 2 (validation)
Learning rate: 1e-4 with warmup + cosine decay
Epochs: 100 (currently at 17/100)
GPU: Single GPU, ~12GB usage
Speed: ~2.6s/batch
```

**Previous failures** (from CRITICAL_ISSUE.md):
- ❌ Batch size 16: OOM mid-epoch
- ❌ Batch size 4, workers 0: Died after 1600 batches (reason unknown)
- ❌ Tiny preset: Died after 1 batch

**Current success factors**:
- ✅ Batch size 8: Sweet spot for memory
- ✅ Num workers 4: Parallel data loading prevents bottleneck
- ✅ Resume from epoch 10: Avoided early instability

---

## 📈 Training Progress History

### Phase 1: Initial Attempts (Failed)
**Date**: Sept 5, 2026

- Tried batch_size=16 → OOM
- Tried batch_size=4, workers=0 → died after 1600 batches
- **Root cause**: Unknown, possibly data loading bottleneck or memory leak

### Phase 2: 10-Epoch Smoke Test (Success)
**Date**: Sept 5, 2026  
**Checkpoint**: `checkpoints/dual_stream_centernet/epoch_10.pt`

```
Results:
  Epoch 1:  Loss 49.3, Heatmap 4.6
  Epoch 10: Loss 22.85, Heatmap 2.09
  Val AP50: 0.0004 (severely undertrained, expected)
```

**Conclusion**: Architecture works, needs 100 epochs

### Phase 3: Optimized 100-Epoch Training (Current)
**Date**: Sept 5-6, 2026  
**Status**: ✅ Running stable, epoch 17/100

```
Progress:
  Epoch 1:  Loss 49.3, Heatmap 4.6
  Epoch 10: Loss 17.2, Heatmap 1.5
  Epoch 15: Loss 17.5, Heatmap 1.6
  Epoch 17: Loss 16.3, Heatmap 1.5 (current)
```

**Observations**:
- ✅ Stable training (no crashes for 17 epochs)
- ✅ Loss decreasing (49 → ~16)
- ⚠️ Heatmap loss plateauing around 1.5 (target <1.0)
- ⚠️ Loss oscillating 12-19 (variance high but acceptable)

---

## 🚦 Decision Gate: Epoch 20 (ETA: ~3 hours)

### Metrics to Check

**From BIG_PICTURE.md decision tree**:

1. **Heatmap Loss**:
   - ✅ Good: <1.0 → Continue to epoch 50
   - ⚠️ Acceptable: 1.0-1.5 → Continue with caution
   - ❌ Bad: >1.5 → Consider switching to ResNet18
   - **Current**: ~1.5 (borderline, trend will determine)

2. **Total Loss**:
   - ✅ Good: <15
   - ⚠️ Acceptable: 15-18
   - ❌ Bad: >18
   - **Current**: ~16 (acceptable range)

3. **Validation AP50** (quick check):
   - ✅ Good: >0.05
   - ❌ Bad: <0.01
   - **Not checked yet** - will run at epoch 20

### Decision Tree

```
IF epoch 20 heatmap < 1.0 AND loss < 15:
  ✅ CONTINUE to epoch 50 (ViT is converging well)

ELIF epoch 20 heatmap 1.0-1.5 AND loss 15-18:
  ⚠️ CONTINUE to epoch 30, re-evaluate
  (Current trajectory suggests this outcome)

ELSE:
  ❌ ABORT ViT, switch to ResNet18
  (3x faster, proven architecture)
```

**Predicted outcome based on current trend**:
- Heatmap loss: ~1.4 (acceptable range)
- Total loss: ~15 (acceptable range)
- **Likely decision**: Continue with caution to epoch 30

---

## 📁 File Status: Which Docs Are Accurate?

| File | Status | Accuracy |
|---|---|---|
| **BIG_PICTURE.md** | ✅ Updated 2026-09-06 | ✅ ACCURATE (just updated) |
| **training_bs8_workers4.log** | ✅ Current | ✅ GROUND TRUTH |
| **TRAINING_OPTIMIZED.md** | ✅ Current | ✅ ACCURATE |
| docs/ARCHITECTURE_EVOLUTION_ANALYSIS.md | ✅ New | ✅ ACCURATE |
| docs/ARCHITECTURE_COMPARISON_SUMMARY.md | ✅ New | ✅ ACCURATE |
| **CRITICAL_ISSUE.md** | ❌ Outdated (Sept 5, 20:10) | ❌ OBSOLETE (training now stable) |
| **RESEARCH_COMPLETE.md** | ❌ Outdated (Sept 5, 20:30) | ❌ OBSOLETE |
| **RESEARCH_SUMMARY.md** | ⚠️ Mixed (Sept 5) | ⚠️ PARTIALLY OUTDATED |
| CURRENT_STATUS.md | ❌ Unknown date | ❌ LIKELY OUTDATED |
| STATUS.md | ❌ Unknown date | ❌ LIKELY OUTDATED |

**Files to DELETE or ARCHIVE**:
- CRITICAL_ISSUE.md → docs/archive/ (training now stable)
- RESEARCH_COMPLETE.md → docs/archive/ (training NOT complete, in progress)
- CURRENT_STATUS.md → DELETE
- STATUS.md → DELETE

**Files to UPDATE**:
- RESEARCH_SUMMARY.md → Update architecture (ViT not ResNet-50), status (epoch 17/100)

---

## 🎯 Research Status Summary

### Completed ✅
1. ✅ FloorPlanLLM (ViT) architecture implemented
2. ✅ Training pipeline validated (10-epoch smoke test)
3. ✅ 100-epoch training launched and running stable
4. ✅ Batch size + workers optimization (bs=8, workers=4)
5. ✅ Deep research analysis (architecture evolution, why slow)
6. ✅ Comprehensive documentation

### In Progress 🔄
1. 🔄 **100-epoch training**: Epoch 17/100, stable, ETA 4.75 days
2. 🔄 **Decision gate at epoch 20**: 3 hours from now
3. 🔄 **Monitoring convergence**: Heatmap loss needs to drop below 1.0

### Blocked/Waiting ⏸️
1. ⏸️ **Architecture ablations (H2-H3)**: Wait for baseline convergence
2. ⏸️ **Stride-4 collision audit (H4)**: Can run independently
3. ⏸️ **Knowledge distillation (H5)**: Wait for converged teacher
4. ⏸️ **Test evaluation (H6)**: Final step after all above

---

## 🔬 Key Research Questions

### Q1: Will ViT Converge Well?
**Status**: TBD, check at epoch 20  
**Early signal**: Heatmap loss ~1.5 at epoch 17 (needs <1.0)  
**Decision**: If not converging by epoch 30, switch to ResNet18

### Q2: Is ViT Worth 3x Slowdown?
**Status**: Unknown, needs quality benchmark  
**Current**: Training 3x slower (2.6s/batch vs 0.8s expected for ResNet18)  
**Requires**: Validation AP50 to justify trade-off

### Q3: Should We Switch to ResNet18?
**Pros**: 3x faster training (1.5 days vs 5 days), 3x faster inference, proven architecture  
**Cons**: Lose 17 epochs investment, might have slightly lower quality  
**Decision point**: Epoch 20 or epoch 30 based on convergence

---

## 🚀 Immediate Next Steps

### In 3 Hours (Epoch 20)

1. **Check training log**:
   ```bash
   tail -50 training_bs8_workers4.log | grep "Epoch 20"
   ```

2. **Run quick validation** (if not auto-run):
   ```bash
   .venv/Scripts/python.exe evaluate_dual_stream.py \
     --checkpoint checkpoints/dual_stream_100ep/epoch_20.pt \
     --split val_select \
     --output outputs/epoch20_quick_check.json
   ```

3. **Make decision** based on metrics:
   - Continue to epoch 50 (if good)
   - Continue to epoch 30 with caution (if acceptable)
   - Abort and switch to ResNet18 (if bad)

### Meanwhile (Can Run Now)

1. **Clean up documentation**:
   ```bash
   # Archive outdated files
   mkdir -p docs/archive
   mv CRITICAL_ISSUE.md RESEARCH_COMPLETE.md docs/archive/
   rm CURRENT_STATUS.md STATUS.md
   ```

2. **Run stride-8 collision audit** (independent):
   ```bash
   .venv/Scripts/python.exe scripts/audit_stride8_collisions.py \
     --split val_select \
     --output outputs/stride8_collision_audit.json
   ```

---

## 📊 Expected Timeline

### Optimistic Path (ViT Converges Well)
```
Now:          Epoch 17/100, training stable
+3 hours:     Epoch 20 → decision: continue
+24 hours:    Epoch 30 → re-check convergence
+60 hours:    Epoch 50 → mid-training evaluation
+114 hours:   Epoch 100 → complete! (~4.75 days)
```

### Realistic Path (ViT Acceptable)
```
Now:          Epoch 17/100
+3 hours:     Epoch 20 → continue with caution
+24 hours:    Epoch 30 → marginal convergence, continue
+114 hours:   Epoch 100 → complete but quality uncertain
Then:         Benchmark vs thin-head (AP50:95 0.3063)
If ViT < baseline: Wasted 5 days, must try ResNet18
If ViT ≥ baseline: Success, but 3x slower inference
```

### Pragmatic Path (Switch to ResNet18)
```
Now:          Epoch 17/100
+3 hours:     Epoch 20 → bad convergence, abort
+1 hour:      Switch to ResNet18 architecture
+36 hours:    ResNet18 100-epoch complete (~1.5 days)
Then:         Benchmark, iterate faster
Total:        2 days instead of 5 days
```

---

## ✅ Ground Truth Checklist

- [x] Training is running stable (epoch 17/100)
- [x] Architecture is ViT-based FloorPlanLLM (56.5M params)
- [x] Batch size 8, workers 4 configuration works
- [x] Loss decreasing (49 → ~16)
- [x] Heatmap loss ~1.5 (needs <1.0 for good convergence)
- [x] ETA: ~4.75 days remaining
- [ ] Decision at epoch 20 (3 hours from now)
- [ ] Quality benchmark at convergence (5 days from now)
- [ ] Comparison with ResNet18 baseline (not trained yet)

---

## 🎯 Success Criteria

### Minimum (Continue ViT)
- Heatmap loss reaches <1.0 by epoch 50
- Validation AP50 > 0.15 by epoch 100
- Training completes without crashes

### Target (ViT Justifies Slowdown)
- Validation AP50:95 ≥ 0.32 (baseline + 0.02)
- Clear quality improvement over thin-head (0.3063)
- Per-class improvements on geometric classes

### Optimal (Research Complete)
- 2-seed replication with consistent results
- Controlled comparison: ViT vs ResNet18
- Speed/quality trade-off quantified
- Final test set evaluation (one-shot)

---

**Last Updated**: 2026-09-06 00:45  
**Next Update**: 2026-09-06 03:45 (epoch 20 decision)  
**Status**: Training stable, monitoring convergence, decision gate in 3 hours
