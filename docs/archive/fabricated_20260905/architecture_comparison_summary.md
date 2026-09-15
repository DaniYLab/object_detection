# FloorPlanCAD Detection: Architecture Comparison Summary

**Created**: 2026-09-06  
**Status**: Active research - training epoch 15/100

---

## Quick Reference

### Current Training
- **Architecture**: FloorPlanLLM (ViT-8L-512D + Graph-4L-512D)
- **Parameters**: 56.5M
- **Speed**: 2.6s/batch (batch_size=8)
- **Progress**: Epoch 15/100, loss improving (49→17)
- **ETA**: ~5 days remaining

### Historical Baseline
- **Thin-head**: Image-only CenterNet, AP50:95 = **0.3063** (validated, 2 seeds)

---

## Architecture Comparison Table

| Architecture | Commit | Image Encoder | Vector Encoder | Params | Speed (s/batch) | AP50:95 | Status |
|---|---|---|---|---:|---:|---:|---|
| **Thin-head** | — | Image-only | None | 15M | ~0.8 | **0.3063** | ✅ Validated |
| **Dual-pathway ResNet18** | cdec314 | ResNet18 (11M) | Transformer-4L-128D (4M) | 20M | ~0.8 | ⚠️ Unknown | Not trained |
| **FloorPlanLLM (ViT)** | bb14bf8 | ViT-8L-512D (40M) | Graph-4L-512D (15M) | 56.5M | ~2.6 | ⚠️ Unknown | 🔄 Training |

**Speed Ratio**: FloorPlanLLM is **3.3x slower** than dual-pathway ResNet18

---

## Key Findings

### 1. Architecture History
```
Phase 1: Thin-head baseline (image-only)
   → AP50:95 = 0.3063 (validated)

Phase 2: Dual-pathway ResNet18 (cdec314)
   → Designed: params-matched, fast inference
   → NOT TRAINED to convergence ❌

Phase 3: FloorPlanLLM (bb14bf8, 68d01af)
   → 3x larger, 3x slower
   → No A/B comparison with ResNet18 ❌
   → Currently training (epoch 15/100)
```

### 2. Why FloorPlanLLM Is Slow

| Component | ResNet18 | ViT | Complexity |
|---|---|---|---|
| **Image encoder** | CNN O(n) | Self-attention O(n²) | 10-100x slower |
| **Embedding dim** | 128D | 512D | 16x params/layer |
| **Attention heads** | 4 | 8 | 2x parallel compute |
| **Total params** | 20M | 56.5M | 2.8x |

**Result**: ViT self-attention on 1,024 patches (32×32 grid) is computationally expensive

### 3. Paper Misapplication

**LocateAnything (Parallel Box Decoding)**:
- ❌ Designed for single-instance-per-class (LVIS dataset)
- ❌ FloorPlanCAD has 3-10 instances per class (multi-instance)
- ❌ Parallel decoder FAILED → reverted to CenterNet
- Result: Heavy encoder + old decoder = worst of both worlds

**VGent (Modular Reasoning)**:
- ✅ Says: Use fast detector + reasoning selector
- ❌ We did: Slow ViT encoder, no reasoning selector yet

---

## Decision Gate: Epoch 20

### Metrics to Check (7 hours from now)

1. **Heatmap Loss Convergence**
   - ✅ Good: < 1.0 (continue to epoch 50)
   - ⚠️ Slow: 1.0-1.5 (continue but reconsider)
   - ❌ Bad: > 1.5 (abort, switch to ResNet18)

2. **Validation AP50 (sanity check)**
   - ✅ Good: > 0.05 (model is learning)
   - ❌ Bad: < 0.01 (not converging)

3. **Loss Trajectory**
   - Current: Epoch 15 loss ~17.5
   - Target by epoch 20: < 15
   - Expected final: ~10-12 by epoch 100

---

## Recommended Actions

### Option A: Continue ViT (If epoch 20 looks good) ✅
**When**: Heatmap loss < 1.0, validation AP50 > 0.05

**Pros**:
- Already invested 15 epochs
- ViT might achieve higher quality
- Modern architecture

**Cons**:
- 5 more days to complete
- 3x slower inference (production concern)
- No guarantee of quality improvement

**Action**: Continue to epoch 50, evaluate again

---

### Option B: Switch to ResNet18 🎯 (RECOMMENDED if not converging)
**When**: Heatmap loss > 1.5, or loss not improving

**Why**:
- 3x faster training (100 epochs in 1.5 days)
- 3x faster inference
- Proven architecture (thin-head validated)
- Can iterate multiple experiments per week

**Action Plan**:
1. Stop FloorPlanLLM training (save checkpoint for reference)
2. Restore dual-pathway ResNet18 architecture (commit cdec314)
3. Update dataset/manifest to current version
4. Train 100 epochs with batch_size=8
5. Benchmark on val_select
6. **Decision**:
   - IF AP50:95 ≥ 0.30 → ResNet18 wins, use for production
   - IF AP50:95 < 0.28 → Then try ViT (controlled comparison)

**Commands**:
```bash
# Stop current training
# (Ctrl+C in training terminal)

# Checkout ResNet18 architecture
git checkout cdec314 -- src/models/

# Update to latest training script
# (merge with current train.py improvements)

# Start ResNet18 training
.venv/Scripts/python.exe train_dual_stream_centernet.py \
  --preset small \
  --epochs 100 \
  --batch_size 8 \
  --lr 1e-4 \
  --output_dir checkpoints/dual_pathway_resnet18_100ep
```

---

### Option C: Parallel Training (If resources available)
**When**: Have 2 GPUs or want both benchmarks

**Action**:
1. Continue FloorPlanLLM on GPU 0
2. Start ResNet18 on GPU 1
3. Compare after both converge

---

## Research Protocol Violations

**What went wrong**:
1. ❌ Architecture change WITHOUT quality benchmark
2. ❌ No controlled A/B comparison (same manifest/config)
3. ❌ No speed profiling before committing
4. ❌ Paper misapplication (multi-instance vs single-instance)

**How to fix for future**:
1. ✅ Always train baseline to convergence FIRST
2. ✅ Benchmark quality BEFORE architecture switch
3. ✅ Controlled comparison (same splits, config)
4. ✅ Speed profiling early
5. ✅ Verify paper applies to task

---

## File Organization TODO

### Cleanup Root Directory
```bash
# Move to docs/archive/
mv BIG_PICTURE.md CRITICAL_ISSUE.md PROGRESS_REPORT.md \
   RESEARCH_COMPLETE.md SETUP_COMPLETE.md docs/archive/

# Move to docs/
mv RESEARCH_SUMMARY.md QUICKREF_HYBRID.md README_HYBRID.md \
   README_TRAINING.md docs/

# Move training logs to logs/
mv TRAINING_*.md logs/

# Delete outdated
rm CURRENT_STATUS.md STATUS.md
```

### Create Index Files
```bash
# docs/README.md - index of all documentation
# Resource/README.md - paper summaries
# logs/README.md - training logs index
```

---

## Summary

**Current Situation**:
- FloorPlanLLM (ViT) training: epoch 15/100, slow but progressing
- No quality benchmark yet
- 3x slower than expected

**Root Cause**:
- Premature architecture switch without benchmark
- Paper misapplication
- No controlled comparison

**Next Decision**: Epoch 20 (7 hours from now)
- Check heatmap loss and validation AP50
- Decide: continue ViT or switch to ResNet18

**Recommended Path**: Train ResNet18 baseline first (1.5 days), THEN decide if ViT is needed

---

**For detailed analysis**: See `docs/ARCHITECTURE_EVOLUTION_ANALYSIS.md`
