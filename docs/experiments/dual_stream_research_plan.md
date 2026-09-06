# DualStreamCenterNet: Comprehensive Research Plan

**Status**: Initial implementation complete, extended training required  
**Date**: 2025-01-XX  
**Reference**: `docs/dual_stream_experiment_report.md`

---

## Phase 1: Establish Converged Baseline ⚠️ **CURRENT PRIORITY**

### Objective
Train DualStreamCenterNet to convergence and establish a reproducible baseline for architecture comparisons.

### Tasks

1. **Extended training run** (100 epochs minimum)
   - [x] Implement architecture and training pipeline
   - [x] Run initial 10-epoch smoke test
   - [ ] **Launch 100-epoch training with lr scheduling**
   - [ ] Monitor heatmap loss convergence (target: <0.5)
   - [ ] Save checkpoints every 10 epochs
   - [ ] Run diagnostic at epochs 20, 50, 100

2. **Convergence validation**
   - [ ] Verify heatmap predictions form sharp peaks (max prob >0.8)
   - [ ] Check val_select AP50 every 10 epochs
   - [ ] **Success gate**: AP50 >0.15 (matching dual_pathway_v2 range)

3. **Baseline evaluation**
   - [ ] Evaluate best checkpoint on `val_select`
   - [ ] Run per-class analysis
   - [ ] Document training curve, final metrics, and checkpoint provenance
   - [ ] **Do not access test_set**

**Expected timeline**: 8-12 hours training + 2 hours analysis  
**Blocker**: Current 10-epoch model is severely undertrained (AP50 ~0.0004)

---

## Phase 2: Architecture Ablations (after baseline convergence)

### 2.1 Fusion Strategy Comparison

**Hypothesis**: Cross-attention fusion captures richer image-vector interactions than element-wise blending.

**Experiments**:
- **A (baseline)**: Current learned alpha-blend
- **B**: Concatenation → 1x1 conv (parameter-matched)
- **C**: Cross-attention (vector queries image features)

**Protocol**:
- Same training config (100 epochs, lr schedule)
- Same checkpoint selection (val_select)
- 2 seeds minimum per variant
- Report AP50, AP50:95, params, latency

### 2.2 Vector Encoder Scaling

**Hypothesis**: Deeper/wider vector encoder improves SVG feature learning.

**Experiments**:
- **A (baseline)**: 4 layers, 128-dim, 4 heads
- **B**: 6 layers, 256-dim, 8 heads (capacity-matched with image branch)
- **C**: Hierarchical pooling (878 tokens → 128 → 16)

**Metrics**: AP50 + per-class analysis (geometry-heavy classes: wall, window, door)

### 2.3 Backbone Alternatives

**Hypothesis**: Stronger image encoder improves overall detection.

**Experiments**:
- **A (baseline)**: ResNet18
- **B**: ResNet34 (deeper)
- **C**: EfficientNet-B1 (speed/accuracy trade-off)

**Trade-off analysis**: AP vs latency vs memory

---

## Phase 3: Training Efficiency Improvements

### 3.1 Learning Rate Tuning

**Current**: Flat 1e-4 → warmup + cosine decay  
**Test if epoch 50 loss >1.0**:
- Lower base lr: 5e-5
- Two-stage: freeze backbone first 20 epochs, then finetune
- Differential lr: backbone 1e-5, decoder 1e-4

### 3.2 Class-Balanced Sampling

**Observation**: Rare classes get few positive examples per epoch  
**Experiment**: Weight rare classes higher in sampling (test on validation first)

### 3.3 Data Augmentation

**Baseline**: Standard image augmentation from dataset  
**Test**: Stronger augmentation, mixup, or cutout for regularization

---

## Phase 4: Decoder Improvements (accuracy track)

### 4.1 Hybrid Stride-4 Head

**Hypothesis**: Small objects benefit from higher-resolution heatmap.

**Approach**:
- Audit val_select for small-object collision rate at stride 8
- If collisions >10%, implement selective stride-4 refinement
- Compare AP50 on small-object subset vs full dataset

### 4.2 Multi-Scale Fusion

**Approach**: FPN-style decoder using multiple ResNet stages  
**Expected gain**: Better small/large object trade-off

---

## Phase 5: Speed Optimization Track

### 5.1 Knowledge Distillation

**Teacher**: Converged DualStreamCenterNet (Phase 1 result)  
**Student**: Image-only CenterNet (no vector branch)

**Target**: Preserve 90% AP50 with 50% latency reduction

### 5.2 Quantization

**Baseline**: FP32 inference latency  
**Test**: INT8 quantization via PyTorch quantization API  
**Trade-off**: Speed gain vs AP drop

### 5.3 Architecture Pruning

**Approach**: Structured pruning of image/vector encoders  
**Target**: 30% parameter reduction with <5% AP drop

---

## Phase 6: Final Validation (test_set access)

**Prerequisites** (all must be complete):
1. ✅ Architecture choices finalized from Phase 2 ablations
2. ✅ Decoder settings calibrated on `val_calibration`
3. ✅ Thresholds selected on `val_calibration` only
4. ✅ Checkpoint provenance documented
5. ✅ Minimum 2 seeds for final architecture

**Final evaluation protocol**:
1. Select ONE checkpoint from `val_select`
2. Calibrate decoder on `val_calibration`
3. Run test_set evaluation ONCE
4. Report AP50, AP50:95, per-class metrics with full provenance
5. Compare against historical baselines (same manifest fingerprint)

**No further test_set access after this point.**

---

## Success Criteria

### Phase 1 (baseline)
- [x] Architecture implemented and tested
- [ ] Training converges (heatmap loss <0.5)
- [ ] AP50 >0.15 on val_select
- [ ] Reproducible with documented checkpoint

### Phase 2-3 (architecture/training)
- [ ] At least 2 fusion strategies compared
- [ ] Best architecture identified via validation AP
- [ ] Training efficiency improvements documented

### Phase 4-5 (optimization)
- [ ] Speed/accuracy trade-off quantified
- [ ] At least one distillation experiment complete

### Phase 6 (final validation)
- [ ] One clean test_set evaluation with full provenance
- [ ] Comparison with repository baselines documented

---

## Risk Mitigation

### Risk: Extended training doesn't converge
- **Mitigation**: Check diagnostic at epoch 20; if heatmap loss >1.5, stop and debug (lr, loss weights, data quality)

### Risk: Architecture doesn't match baseline performance
- **Mitigation**: Reproduce simpler baseline first (image-only CenterNet) to isolate vector branch contribution

### Risk: Phase 1 takes too long (>2 days)
- **Mitigation**: Switch to smaller "tiny" preset for faster iteration, scale up after validation

---

## Files and Artifacts

### Code
- `src/models/dual_stream_centernet.py` - Main architecture
- `train_dual_stream_centernet.py` - Training script
- `evaluate_dual_stream.py` - Evaluation script
- `diagnose_dual_stream.py` - Diagnostic tool

### Experiment tracking
- `docs/dual_stream_experiment_report.md` - Phase 1 report
- `docs/dual_stream_ablations.md` - Phase 2-3 results (to be created)
- `outputs/dual_stream_*.json` - Evaluation results

### Checkpoints
- `checkpoints/dual_stream_100ep/` - Extended training (Phase 1)
- `checkpoints/dual_stream_ablations/` - Variants (Phase 2)

---

## Timeline Estimate

| Phase | Duration | Blocking? |
|---|---|---|
| Phase 1: Baseline | 12-24 hours | Yes (required for all others) |
| Phase 2: Ablations | 3-5 days | Yes (architecture selection) |
| Phase 3: Training | 1-2 days | No (can parallelize) |
| Phase 4: Decoder | 2-3 days | No (validation-only) |
| Phase 5: Speed | 2-3 days | No (post-architecture) |
| Phase 6: Test | 4 hours | Final (one-shot) |

**Critical path**: Phase 1 → Phase 2 → Phase 6  
**Total estimated time**: 1-2 weeks

---

## Next Immediate Action

**Launch extended training run:**

```bash
.venv/Scripts/python.exe train_dual_stream_centernet.py \
  --preset small \
  --epochs 100 \
  --batch_size 8 \
  --lr 1e-4 \
  --output_dir checkpoints/dual_stream_100ep
```

**Monitor progress** at epochs 20, 50, 100:
```bash
python diagnose_dual_stream.py
```

**Evaluation after convergence**:
```bash
python evaluate_dual_stream.py \
  --checkpoint checkpoints/dual_stream_100ep/best_val.pt \
  --split val_select \
  --output outputs/dual_stream_100ep_val_select.json
```
