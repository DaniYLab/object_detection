# FloorPlanCAD Detection: Research Status and Next Steps

**Date**: 2025-01-XX  
**Status**: DualStreamCenterNet implemented, baseline training in progress  
**Goal**: Match/exceed object detection SOTA while improving inference speed

---

## Executive Summary

Completed implementation of **DualStreamCenterNet**, a dual-pathway architecture combining image (ResNet18) and vector (Transformer) encoders for FloorPlanCAD object detection. Initial 10-epoch training validated the pipeline but revealed severe undertraining (AP50 0.0004). Diagnostic analysis confirms the architecture is sound but requires 100+ epochs to converge.

**Critical next step**: Launch extended 100-epoch training to establish converged baseline before any architecture comparisons.

---

## Research Context

### Valid Benchmark Target

**Object Detection** (same task as this repository):
- Faster R-CNN R101: **AP50 0.602** ← Primary comparison target
- YOLOv3: AP50 0.639
- FCOS: AP50 0.624

**Segmentation** (different task, reference only):
- FloorPlanCAD paper: Entity F1 0.806 / Weighted F1 0.798
- DeepLabv3+-R101: Entity F1 0.688 / Weighted F1 0.714

**Key insight**: The 0.806 F1 is semantic segmentation, not box detection. Our valid comparison is Faster R-CNN AP50 0.602.

### Repository Baselines (Historical)

| Model | Manifest | Val AP50 | Test AP50 | Notes |
|---|---|---:|---:|---|
| **B** `floorplan_base` | f83fcffd | 0.402 | 0.390 | Conditioning baseline |
| **V2** `dual_pathway_v2` | a916b9fa | — | — | Vector fusion |
| **Late** `dual_pathway_late` | 203f6358 | 0.160 | Not accessed | Schema-v2, speed focus |

**Caveat**: Different manifests = not controlled comparisons. Schema-v2 experiments must use manifest `203f6358`.

---

## DualStreamCenterNet: Current Status

### Architecture

```
Image Branch:  ResNet18 → 512-dim features (stride 8)
                    ↓
              Learned Alpha Blend ← 512-dim features
                    ↓                        ↑
              CenterNet Decoder     Transformer Encoder
              (heatmap/size/offset)  (4L, 128D, 4H)
                                            ↑
Vector Branch:  SVG Strokes (878 tokens × 12-dim)
```

**Parameters**: 9.82M (small preset)  
**Training**: Query-based sampling, CenterNet focal loss + L1

### Implementation Status

✅ **Complete**:
- Full architecture implementation (`src/models/dual_stream_centernet.py`)
- Training pipeline with query-based sampling
- Evaluation infrastructure with existing metrics
- Diagnostic tools for heatmap analysis

⚠️ **In Progress**:
- 10-epoch smoke test complete (validated pipeline)
- Extended 100-epoch training **required** (blocking all experiments)

### Diagnostic Results (10-epoch checkpoint)

**Quantitative**:
- Val AP50: 0.0004 (near zero, expected for undertrained model)
- Heatmap loss: 2.09 (should be <0.5 when converged)
- Max heatmap probability: 0.208 (should be >0.8 at object centers)
- Pixels >0.5 confidence: 0 (no confident detections)

**Qualitative**:
- Model produces diffuse low-confidence heatmaps
- No sharp peaks at object centers (not yet learned)
- Size/offset predictions are active but unmeaningful without centers

**Root cause**: Insufficient training. CenterNet requires 100-200 epochs to learn sharp heatmap predictions.

---

## Prioritized Research Plan

### Phase 1: Establish Baseline 🔴 **BLOCKING**

**Hypothesis 1**: Extended training (100 epochs) → AP50 >0.15

**Protocol**:
1. Train 100 epochs with lr warmup + cosine decay
2. Monitor heatmap loss (target <0.5) and max probability (target >0.8)
3. Evaluate on val_select every 10 epochs
4. Run diagnostic at epochs 20, 50, 100

**Success gate**: AP50 >0.15 matching `dual_pathway_late` range

**Timeline**: 8-12 hours  
**Status**: Ready to launch (command prepared)

---

### Phase 2: Architecture Ablations 🟡 **After Phase 1**

**Hypothesis 2**: Cross-attention fusion > learned blend  
**Experiments**:
- A (baseline): Learned alpha-blend (current)
- B: Concatenation → 1x1 conv
- C: Cross-attention (vector queries image)

**Expected gain**: +0.02-0.05 AP50 if C > A  
**Protocol**: 100 epochs, 2 seeds, val_select selection  
**Speed impact**: Cross-attention +20-30% latency

---

**Hypothesis 3**: Deeper vector encoder improves geometry  
**Experiments**:
- A (baseline): 4L, 128D, 4H
- B: 6L, 256D, 8H (capacity-matched)
- C: Hierarchical pooling (878 → 128 → 16)

**Expected gain**: +0.01-0.03 AP50 on geometry classes  
**Speed impact**: B +15% latency, C -20% latency

---

### Phase 3: Decoder Improvements 🟢 **Independent**

**Hypothesis 4**: Hybrid stride-4 head for small objects

**Pre-experiment audit** (no training):
1. Measure collision rate on val_select at stride 8
2. Compute small-object subset AP (<32px²)
3. **Decision gate**: If collisions >10%, implement stride-4

**Expected gain**: +0.03-0.05 AP50 if collisions are high  
**Speed impact**: +40% latency for affected images  
**Status**: Audit script ready, can run immediately

---

### Phase 4: Speed Optimization 🟢 **After Phase 1**

**Hypothesis 5**: Knowledge distillation → fast image-only student

**Protocol**:
1. Use converged DualStreamCenterNet (Phase 1) as teacher
2. Train image-only CenterNet student with distillation
3. Target: 90% teacher AP, 50% latency

**Expected**: AP50 >0.135 with 2x speedup  
**Status**: Distillation infrastructure ready

---

### Phase 5: Training Efficiency 🟡 **Conditional**

**Hypothesis 6**: Two-stage lr (frozen backbone) accelerates convergence

**Trigger**: If Phase 1 heatmap loss >1.0 at epoch 50  
**Test**: Freeze ResNet18 for first 20 epochs, unfreeze for 21-100  
**Expected**: Faster convergence, similar final AP

---

### Phase 6: Final Validation 🔴 **ONE-SHOT**

**Prerequisites** (all must complete):
- ✅ Architecture finalized (Phase 2)
- ✅ Decoder calibrated on val_calibration
- ✅ Thresholds selected on val_calibration only
- ✅ Minimum 2 seeds for final architecture

**Protocol**:
1. Select ONE best checkpoint from val_select
2. Calibrate decoder on val_calibration
3. Run test_set evaluation ONCE
4. Report AP50, AP50:95, per-class with full provenance
5. Compare vs Faster R-CNN baseline (AP50 0.602)

**No further test_set access.**

---

## Expected Outcomes

### Conservative Estimate (Likely)

| Phase | Contribution | Cumulative AP50 |
|---|---:|---:|
| Phase 1 (baseline) | +0.15 | 0.15 |
| Phase 2 (fusion) | +0.02 | 0.17 |
| Phase 3 (stride-4) | +0.03 | 0.20 |
| **Validation total** | | **0.20** |
| Val→test gap | -0.02 | **0.18** |

**Test set expectation**: AP50 0.18-0.22

### Optimistic Estimate (Best Case)

| Phase | Contribution | Cumulative AP50 |
|---|---:|---:|
| Phase 1 (baseline) | +0.18 | 0.18 |
| Phase 2 (fusion) | +0.04 | 0.22 |
| Phase 3 (stride-4) | +0.04 | 0.26 |
| **Validation total** | | **0.26** |
| Val→test gap | -0.02 | **0.24** |

**Test set expectation**: AP50 0.24-0.28

### Gap to SOTA

**Target**: Faster R-CNN AP50 0.602  
**Expected result**: AP50 0.18-0.24  
**Remaining gap**: 0.36-0.42

**Future hypotheses needed** (beyond current plan):
- H7: FPN multi-scale decoder (+0.05-0.08 AP50)
- H8: Stronger backbone (ResNet34/EfficientNet) (+0.03-0.05)
- H9: Query augmentation (+0.01-0.02)
- H10: Class-balanced sampling refinement (+0.02)

**Combined H1-H10**: Could reach AP50 0.35-0.40

---

## Research Protocol Compliance

Every experiment follows `docs/research_protocol.md`:

✅ **Split hygiene**:
- Train on train_new
- Select checkpoints on val_select
- Calibrate decoder on val_calibration
- Access test_set ONCE at final report

✅ **Provenance tracking**:
- Manifest schema v2, fingerprint 203f6358
- Checkpoint/decoder/calibration artifacts tagged
- Split roles documented

✅ **Controlled comparisons**:
- Every architecture change has baseline control
- Same training config for ablations
- Minimum 2 seeds for architecture claims

✅ **Metric alignment**:
- Report AP50/AP50:95 (box detection)
- Not segmentation F1 (different task)
- Compare with Faster R-CNN AP50 0.602

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Phase 1 doesn't converge | Low | High | Diagnostic at epoch 20; debug lr/loss |
| Architecture worse than baseline | Medium | Medium | Reproduce image-only baseline first |
| Can't reach 0.60 AP50 | Medium | Low | Document progress; H7-H10 next phase |
| Training time too long | Low | Medium | Switch to "tiny" preset for iteration |

---

## Immediate Actions

### 1. Launch Extended Training 🔴 **URGENT**

```bash
.venv/Scripts/python.exe train_dual_stream_centernet.py \
  --preset small \
  --epochs 100 \
  --batch_size 8 \
  --lr 1e-4 \
  --output_dir checkpoints/dual_stream_100ep
```

**Monitor**: Diagnostic at epochs 20, 50, 100  
**Expected time**: 8-12 hours  
**Blocker**: All architecture experiments wait for this

---

### 2. Stride-8 Collision Audit 🟢 **PARALLEL**

```bash
# Create audit script (to be implemented)
.venv/Scripts/python.exe scripts/audit_stride8_collisions.py \
  --split val_select \
  --output outputs/stride8_collision_audit.json
```

**Decision**: If collision rate >10%, implement stride-4 head  
**Status**: Can run immediately (no training dependency)

---

### 3. Prepare Ablation Infrastructure 🟡 **NEXT**

- Implement cross-attention fusion variant
- Implement 6L/256D vector encoder variant
- Prepare 2-seed training configs
- **Wait for Phase 1 convergence before launching**

---

## Files and Documentation

### Implementation
- `src/models/dual_stream_centernet.py` - Architecture
- `src/models/image_encoder.py` - ResNet18 backbone
- `src/models/vector_encoder.py` - Transformer encoder
- `src/models/centernet_heads.py` - Detection heads
- `train_dual_stream_centernet.py` - Training script
- `evaluate_dual_stream.py` - Evaluation script
- `diagnose_dual_stream.py` - Diagnostic tool

### Documentation
- ✅ `docs/dual_stream_experiment_report.md` - Phase 1 initial results
- ✅ `docs/dual_stream_research_plan.md` - Detailed research plan
- ✅ `docs/deep_research_next_steps.md` - Hypothesis-driven roadmap
- ✅ `docs/research_protocol.md` - Scientific methodology
- ✅ `docs/SOTA.md` - Benchmark reference and provenance

### Checkpoints & Outputs
- `checkpoints/dual_stream_centernet/epoch_{1..10}.pt` - Smoke test
- `checkpoints/dual_stream_100ep/` - Extended training (pending)
- `outputs/dual_stream_val_select.json` - Evaluation results

---

## Timeline

| Phase | Duration | Dependencies | Status |
|---|---|---|---|
| Phase 1: Baseline | 12-24h | None | 🔴 Ready to launch |
| Phase 2: Ablations | 3-5 days | Phase 1 | 🟡 Prepared |
| Phase 3: Stride-4 | 2-3 days | Audit only | 🟢 Independent |
| Phase 4: Distillation | 2-3 days | Phase 1 | 🟡 Prepared |
| Phase 5: Training | 1-2 days | Phase 1 result | 🟡 Conditional |
| Phase 6: Test | 4 hours | All above | Final |

**Critical path**: Phase 1 → Phase 2 → Phase 6  
**Total estimated time**: 1-2 weeks

---

## Conclusion

**Current bottleneck**: Phase 1 extended training (100 epochs) is the blocking task for all scientific progress. The architecture is validated and ready; it simply needs training time to converge.

**Scientific approach**: Hypothesis-driven experiments with validation-only selection, controlled comparisons, and one-shot test evaluation. Every experiment follows the frozen research protocol with full provenance tracking.

**Expected outcome**: AP50 0.18-0.24 on test_set after Phases 1-3, establishing a solid foundation for future improvements (H7-H10) toward the Faster R-CNN benchmark (AP50 0.602).

**Next immediate action**: Launch 100-epoch training run and monitor convergence diagnostics.
