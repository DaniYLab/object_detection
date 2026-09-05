# Deep Research Summary: Next Scientific Steps for FloorPlanCAD Detection

**Date**: 2025-01-XX  
**Research Goal**: Scientifically rigorous path to matching/exceeding SOTA while improving inference speed  
**Protocol**: `docs/research_protocol.md` - validate on val before test, isolate causal effects, match metrics/protocols

---

## 1. Current State Assessment

### Repository Baselines (Historical, Different Manifests)

| Model | Manifest | Val AP50 | Test AP50 | Notes |
|---|---|---:|---:|---|
| **B** `floorplan_base` | f83fcffd | 0.402 | 0.390 | Conditioning baseline |
| **V2** `dual_pathway_v2` | a916b9fa | — | — | Vector fusion, different split |
| **Late** `dual_pathway_late` | 203f6358 | 0.160 | Not accessed | Schema-v2, speed focus |

**Caveat**: Cross-manifest comparisons are not controlled. Schema-v2 results must use manifest `203f6358` for valid deltas.

### SOTA Reference (FloorPlanCAD Paper, Entity-Level Segmentation F1)

| Method | F1 | Weighted F1 | Task |
|---|---:|---:|---|
| DeepLabv3+-R101 | 0.688 | 0.714 | Semantic segmentation |
| **FloorPlanCAD (paper)** | 0.806 | 0.798 | Entity-level F1 |
| Faster R-CNN R101 | — | — | **Object detection: AP50 0.602** |
| YOLOv3 | — | — | **Object detection: AP50 0.639** |

**Key insight**: The 0.806 F1 table is segmentation (different task). The **object detection benchmark** is Faster R-CNN AP50 0.602, which is the valid comparison target for this repository.

### Recent Architecture: DualStreamCenterNet

**Implementation status**: ✅ Complete and tested  
**Training status**: ⚠️ Undertrained (10 epochs, AP50 0.0004)  
**Architecture**:
- Image branch: ResNet18 → 512-dim
- Vector branch: Transformer (4L, 128D, 4H) → 512-dim
- Fusion: Learned alpha-blend
- Decoder: CenterNet heatmap + size + offset
- Parameters: 9.82M

**Diagnostic findings**:
- Heatmap loss plateaus at 2.09 (should be <0.5)
- Max heatmap probability 0.208 (should be >0.8 at centers)
- Zero pixels above 0.3 confidence
- **Root cause**: Needs 100+ epochs to converge

---

## 2. Prioritized Research Hypotheses

### Hypothesis 1: Extended Training → Converged Baseline
**Status**: 🔴 **BLOCKING** - Must complete before all architecture experiments  
**Claim**: DualStreamCenterNet with 100-epoch training will reach AP50 >0.15 (matching `dual_pathway_late` range)  
**Test**:
1. Train 100 epochs with lr warmup + cosine decay
2. Monitor heatmap loss convergence (target <0.5)
3. Evaluate on val_select at epochs 20, 50, 100
4. Run diagnostic to verify sharp heatmap peaks (max prob >0.8)

**Expected gain**: +0.14-0.15 AP50 (from 0.0004 → 0.15)  
**Risk**: Low (architecture is sound, just needs training time)  
**Timeline**: 8-12 hours  
**Speed implication**: Establishes baseline latency for fusion comparisons

---

### Hypothesis 2: Cross-Attention Fusion > Learned Blend
**Status**: 🟡 Depends on H1 convergence  
**Claim**: Cross-attention between image and vector features captures richer spatial interactions than element-wise blending  
**Ablation**:
- **A (baseline)**: Learned alpha-blend (current)
- **B**: Concatenation → 1x1 conv (parameter-matched control)
- **C**: Cross-attention (vector tokens query image features)

**Protocol**:
- Train each variant 100 epochs on train_new
- Select checkpoint on val_select
- 2 seeds minimum per variant
- Report AP50, AP50:95, params, synchronized inference latency

**Expected gain**: +0.02-0.05 AP50 if C>A  
**Risk**: Medium (attention may overfit with limited data)  
**Speed implication**: Cross-attention adds ~20-30% latency vs blend

---

### Hypothesis 3: Deeper Vector Encoder Improves Geometry Understanding
**Status**: 🟡 Depends on H1 convergence  
**Claim**: Vector encoder (4L/128D) is undertrained relative to image encoder; scaling to 6L/256D improves SVG feature learning  
**Ablation**:
- **A (baseline)**: 4L, 128D, 4H (current)
- **B**: 6L, 256D, 8H (capacity-matched with image branch)
- **C**: Hierarchical pooling (878 tokens → 128 → 16 via pooling layers)

**Diagnostic**: Compare AP50 on geometry-heavy classes (wall, window, door) vs full dataset  
**Expected gain**: +0.01-0.03 AP50 on geometry classes  
**Risk**: Low  
**Speed implication**: B adds ~15% latency, C reduces by ~20%

---

### Hypothesis 4: Hybrid Stride-4 Head Resolves Small-Object Collisions
**Status**: 🟢 Independent validation experiment (no training needed yet)  
**Claim**: Small objects at stride-8 suffer from quantization collisions; selective stride-4 refinement improves small-object AP  
**Pre-experiment audit**:
1. Run collision audit on val_select at stride 8
2. Measure GT center collision rate (multiple objects map to same heatmap pixel)
3. Compute small-object subset AP (<32px²)

**Decision gate**:
- If collision rate >10%: Implement stride-4 head for small objects
- If collision rate <5%: Skip (not the bottleneck)

**Expected gain**: +0.03-0.05 AP50 if collisions are high  
**Risk**: Low (validation-only audit first)  
**Speed implication**: Stride-4 head adds ~40% latency for affected images

---

### Hypothesis 5: Knowledge Distillation → Fast Image-Only Student
**Status**: 🟢 Independent speed track (after H1 teacher converges)  
**Claim**: Image-only student distilled from DualStreamCenterNet teacher preserves 90% AP with 50% latency reduction  
**Protocol**:
1. Use converged DualStreamCenterNet (H1 result) as teacher
2. Train image-only CenterNet student with distillation loss
3. Compare student AP vs teacher AP and latency

**Target**: AP50 >0.135 (90% of 0.15 baseline) with 2x speedup  
**Risk**: Medium (distillation may not transfer well)  
**Speed implication**: Primary speed optimization path

---

### Hypothesis 6: Two-Stage LR (Frozen Backbone) Accelerates Convergence
**Status**: 🟡 Depends on H1 result  
**Claim**: If H1 heatmap loss >1.0 at epoch 50, frozen backbone for first 20 epochs improves decoder learning  
**Test**:
- Train variant: freeze ResNet18 for epochs 1-20, unfreeze for 21-100
- Compare convergence speed and final AP vs H1 baseline

**Expected gain**: Faster convergence, similar final AP  
**Risk**: Low (validation of training efficiency)

---

## 3. Experiment Dependency Graph

```
H1 (Extended Training) [BLOCKING]
   ├─> H2 (Fusion Ablation)
   ├─> H3 (Vector Encoder Scaling)
   ├─> H5 (Distillation)
   └─> H6 (LR Strategy)

H4 (Hybrid Stride-4) [INDEPENDENT - validation audit first]
```

**Critical path**: H1 → H2/H3 → Test evaluation  
**Parallel track**: H4 audit, H5 preparation

---

## 4. Research Protocol Compliance Checklist

For every experiment:

- [ ] **Split hygiene**: Train on train_new, select on val_select, calibrate on val_calibration, test ONLY at final report
- [ ] **Manifest tracking**: Record schema version (v2), fingerprint (203f6358), and role provenance
- [ ] **Baseline control**: Every architecture change has a controlled baseline comparison (same training config)
- [ ] **Seed replication**: Minimum 2 seeds for architecture claims
- [ ] **Metric alignment**: Report AP50/AP50:95 (box detection), not segmentation F1
- [ ] **Threshold selection**: Calibrate only on val_calibration, never on test
- [ ] **Provenance**: Checkpoint/decoder/calibration artifacts carry manifest fingerprint

**Test set access**: ONE final evaluation after all choices frozen (H1-H6 complete)

---

## 5. Success Gates

### Gate 1: Baseline Convergence (H1)
- ✅ Heatmap loss <0.5
- ✅ Max heatmap probability >0.8 at centers
- ✅ AP50 >0.15 on val_select
- ✅ Checkpoint provenance documented

**If fail**: Debug training (lr, loss weights, data quality) before architecture experiments

### Gate 2: Architecture Selection (H2, H3)
- ✅ At least 2 fusion strategies compared
- ✅ Best architecture identified via validation AP (2 seeds)
- ✅ Speed/accuracy trade-off quantified

### Gate 3: Final Validation (One-Shot Test)
- ✅ ONE checkpoint selected from val_select
- ✅ Decoder calibrated on val_calibration only
- ✅ Test evaluation with full provenance
- ✅ Comparison vs Faster R-CNN benchmark (AP50 0.602)

---

## 6. Expected Outcomes

### Conservative Estimate (Realistic)
- **H1 baseline**: AP50 0.15-0.18 (matching dual_pathway_late range)
- **H2 fusion**: +0.02 AP50 if cross-attention wins
- **H3 vector scaling**: +0.01 AP50
- **H4 stride-4**: +0.03 AP50 if collisions are high
- **Final**: AP50 0.21-0.24 (validation)

**Test set**: AP50 0.19-0.22 (accounting for val→test gap)

### Optimistic Estimate (Best Case)
- **H1 baseline**: AP50 0.20
- **Combined improvements**: +0.08 AP50
- **Final**: AP50 0.28-0.30 (validation)

**Test set**: AP50 0.26-0.28 (approaching Faster R-CNN 0.602 with additional work)

### Speed Trade-Off
- **Baseline**: DualStreamCenterNet (H1) sets reference latency
- **Fast variant**: Image-only student (H5) at 50% latency, 90% AP
- **Hybrid variant**: Stride-4 head (H4) at +40% latency for small objects only

---

## 7. Remaining SOTA Gap Analysis

**Target**: Faster R-CNN AP50 0.602 (object detection benchmark)  
**Current best (historical)**: B `floorplan_base` AP50 0.390 (different manifest)  
**Expected H1-H4 result**: AP50 0.21-0.24 (schema-v2)

**Gap to close**: 0.36-0.38 AP50

**Additional hypotheses needed** (beyond H1-H6):

### H7: FPN-Style Multi-Scale Decoder
- Use multiple ResNet stages (C3, C4, C5) for FPN
- Expected gain: +0.05-0.08 AP50 (better small/large object trade-off)

### H8: Stronger Image Backbone
- Replace ResNet18 with ResNet34 or EfficientNet-B1
- Expected gain: +0.03-0.05 AP50

### H9: Query Augmentation
- Random query dropout, query mixup
- Expected gain: +0.01-0.02 AP50 (regularization)

### H10: Class-Balanced Sampling with Rare-Class Boost
- Careful rare-class upsampling (validated on val first)
- Expected gain: +0.02 AP50 on rare classes

**Combined H1-H10**: Could reach AP50 0.35-0.40, approaching Faster R-CNN range

---

## 8. Immediate Next Actions

### Action 1: Launch Extended Training (H1) 🔴 **URGENT**

```bash
.venv/Scripts/python.exe train_dual_stream_centernet.py \
  --preset small \
  --epochs 100 \
  --batch_size 8 \
  --lr 1e-4 \
  --output_dir checkpoints/dual_stream_100ep
```

**Monitor**: Check diagnostic at epochs 20, 50, 100
**Blocker**: All architecture experiments wait for this

---

### Action 2: Stride-4 Collision Audit (H4) 🟢 **PARALLEL**

```bash
.venv/Scripts/python.exe scripts/audit_stride8_collisions.py \
  --split val_select \
  --output outputs/stride8_collision_audit.json
```

**Decision**: If collision rate >10%, implement stride-4 head

---

### Action 3: Prepare Ablation Scripts (H2, H3) 🟡 **NEXT**

- Implement cross-attention fusion variant
- Implement 6L/256D vector encoder variant
- Prepare training configs for 2-seed runs
- **Wait for H1 convergence before launching**

---

## 9. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| H1 doesn't converge | Low | High | Diagnostic at epoch 20; debug lr/loss |
| Architecture worse than baseline | Medium | Medium | Reproduce image-only baseline first |
| Speed/accuracy trade-off poor | Low | Medium | Distillation (H5) as fallback |
| Can't reach Faster R-CNN AP50 | Medium | Low | Document progress; H7-H10 for next phase |

---

## 10. Documentation and Provenance

### Experiment Logs
- `docs/dual_stream_experiment_report.md` - H1 initial results
- `docs/dual_stream_ablations.md` - H2-H6 results (to be created)
- `docs/experiment_log.md` - Append all new results

### Checkpoints
- `checkpoints/dual_stream_100ep/` - H1 extended training
- `checkpoints/dual_stream_ablations/` - H2-H6 variants

### Outputs
- `outputs/dual_stream_*.json` - Evaluation results
- `outputs/stride8_collision_audit.json` - H4 audit
- `outputs/ablation_comparison_*.json` - H2-H3 comparisons

---

## Conclusion

**Immediate priority**: Complete H1 (extended training) to establish converged baseline. This blocks all architecture comparisons and is the critical path to scientific progress.

**Expected timeline**: 1-2 weeks for H1-H6, with H1 taking 12-24 hours.

**Final test set access**: ONE evaluation after all architecture/decoder/calibration choices frozen.

**SOTA target**: Faster R-CNN AP50 0.602 is the valid object detection benchmark. Schema-v2 experiments aim for AP50 0.21-0.24 (H1-H4), with H7-H10 needed to approach 0.35-0.40 range.
