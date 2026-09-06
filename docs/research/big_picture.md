# FloorPlanCAD Object Detection Research - Big Picture (UPDATED)

**Last Updated**: 2026-09-06 00:30  
**Goal**: Hoàn thành nghiên cứu - Match/exceed object detection SOTA  
**Status**: ⚠️ FloorPlanLLM (ViT) training epoch 15/100 - Decision gate at epoch 20

---

## 🚨 CRITICAL STATUS UPDATE

### Current Reality Check

**What BIG_PICTURE.md SAID** (outdated):
- Architecture: DualStreamCenterNet with ResNet18
- Parameters: 9.82M
- Status: "Ready to launch 100-epoch training"

**What is ACTUALLY HAPPENING** (now):
- Architecture: FloorPlanLLM with ViT-8L-512D
- Parameters: **56.5M** (5.7x larger!)
- Status: **Training in progress** - epoch 15/100
- Speed: 2.6s/batch (**3x slower than expected**)
- ETA: ~5 days remaining (vs expected 1.5 days)

**Why the discrepancy?**: Architecture was changed from ResNet18 → ViT (commits bb14bf8, 68d01af) WITHOUT updating this file or running quality benchmark first.

---

## 🎯 Research Objectives (UNCHANGED)

### Primary Goals
1. **Scientific rigor**: Validation-first protocol, controlled experiments, test accessed once
2. **Performance target**: Match Faster R-CNN AP50 0.602 (valid object detection benchmark)
3. **Speed optimization**: 2-3× faster inference through architectural innovations
4. **Reproducibility**: Multi-seed replication, full provenance tracking

### Current Status
- ✅ FloorPlanLLM architecture implemented (56.5M params)
- 🔄 Training in progress: epoch 15/100, loss 17.5 (improving from 49)
- ⚠️ **Speed concern**: 3x slower than expected (2.6s/batch vs 0.8s)
- ⚠️ **Quality unknown**: No benchmark yet, needs convergence
- 🔴 **Decision gate**: Epoch 20 (7 hours) - Continue ViT or switch to ResNet18?

---

## 📊 Benchmark Context (UNCHANGED)

### Valid Object Detection Benchmarks (Same Task)
| Model | AP50 | mAP | Paper |
|---|---:|---:|---|
| **Faster R-CNN R101** | **0.602** | 0.452 | FloorPlanCAD (target) |
| YOLOv3 | 0.639 | 0.413 | FloorPlanCAD |
| FCOS | 0.624 | 0.453 | FloorPlanCAD |

**Note**: The 0.806 F1 from FloorPlanCAD paper is **semantic segmentation** (different task). Our valid comparison is box detection AP50.

### Repository Baselines (Historical)
| Model | Manifest | Params | Speed | Val AP50 | Test AP50 | Notes |
|---|---|---:|---|---:|---:|---|
| **Thin-head baseline** | — | 15M | ~0.8s | — | **0.3063** | Image-only, validated |
| `floorplan_base` | f83fcffd | 5M | — | 0.402 | 0.390 | Different schema |
| `dual_pathway_late` | 203f6358 | 20M | — | 0.160 | Not accessed | Schema-v2 |
| **FloorPlanLLM (ViT)** | — | 56.5M | ~2.6s | ⚠️ Unknown | — | **Training now** |

**Protocol**: All new experiments use manifest `203f6358` (schema-v2) for valid comparisons.

---

## 🏗️ Current Architecture: FloorPlanLLM (ViT-based)

### Design
```
Image Branch:     ViT-8L-512D (40M params, stride 8)
                      ↓
                Cross-Attention Fusion
                      ↓
                CenterNet Decoder (35 heads, 15M params)
                (heatmap/size/offset)
                      ↑
Vector Branch:  Graph Transformer (4L-512D-8H, 15M params)
                      ↑
                SVG Strokes + Graph Edges (1024 tokens)
```

**Total Parameters**: 56.5M  
**Speed**: 2.6s/batch (batch_size=8, num_workers=4)  
**Key change**: Vision Transformer replaced ResNet18 CNN

### Why ViT Was Chosen (from docs/week1_floorplan_llm.md)
- Inspiration: LocateAnything paper (parallel box decoding)
- Goal: "Paradigm shift to structured understanding"
- Vision: Scene graph output, spatial relationships
- **BUT**: Parallel decoder FAILED (multi-instance problem)
- **THEN**: Reverted to CenterNet decoder
- **RESULT**: Heavy encoder + proven decoder = unclear quality/speed trade-off

### Architecture Comparison

| Component | ResNet18 (old) | ViT (current) | Ratio |
|---|---|---|---|
| **Image encoder** | CNN (11M) | ViT-8L-512D (40M) | 3.6x |
| **Vector encoder** | 4L-128D-4H (4M) | 4L-512D-8H (15M) | 3.8x |
| **Decoder** | 35 heads (5M) | 35 heads (15M) | 3.0x |
| **Total params** | 20M | 56.5M | 2.8x |
| **Speed** | ~0.8s/batch | ~2.6s/batch | **3.3x** |
| **Quality** | ⚠️ Unknown (not trained) | ⚠️ Unknown (training) | — |

---

## 🔬 Current Training Status

### Progress (Epoch 15/100)

**Loss Trajectory**:
```
Epoch 1:  Loss 49.3 → Heatmap 4.6
Epoch 5:  Loss 25.1 → Heatmap 2.2
Epoch 10: Loss 17.2 → Heatmap 1.5
Epoch 15: Loss 17.5 → Heatmap ~1.6 (current)
```

**Observations**:
- ✅ Loss is decreasing (model is learning)
- ⚠️ Heatmap loss still high (target <1.0 for convergence)
- ⚠️ Convergence slower than expected (loss plateauing around 17-18)
- ⚠️ 3x slower per epoch than ResNet18 would be

**ETA**:
- Epochs remaining: 85
- Time per epoch: ~82 minutes
- **Total ETA**: 85 × 82min = ~117 hours = **~5 days**
- Expected completion: **Tuesday Sept 10, ~05:00**

---

## 🚦 Decision Gate: Epoch 20 (7 Hours from Now)

### Metrics to Check

1. **Heatmap Loss Convergence**
   - ✅ **Good**: < 1.0 → Continue to epoch 50
   - ⚠️ **Acceptable**: 1.0-1.5 → Continue but reconsider
   - ❌ **Bad**: > 1.5 → Abort, switch to ResNet18

2. **Total Loss**
   - ✅ **Good**: < 15
   - ⚠️ **Acceptable**: 15-18
   - ❌ **Bad**: > 18 (not improving)

3. **Validation AP50 (quick sanity check)**
   - ✅ **Good**: > 0.05 (model is learning detections)
   - ❌ **Bad**: < 0.01 (not converging)

### Decision Tree

```
Epoch 20 Check:
├─ Heatmap loss < 1.0 AND loss < 15
│  └─ ✅ CONTINUE ViT to epoch 50
│     └─ Monitor, re-evaluate at epoch 50
│
├─ Heatmap loss 1.0-1.5 AND loss 15-18
│  └─ ⚠️ CONTINUE with caution to epoch 30
│     └─ Final decision at epoch 30
│
└─ Heatmap loss > 1.5 OR loss > 18
   └─ ❌ ABORT ViT training
      └─ Switch to ResNet18 baseline (3x faster)
```

---

## 🎯 Alternative Path: Dual-Pathway ResNet18

### Why Consider Switching?

**Pros of ResNet18**:
- ✅ **3x faster training**: 100 epochs in 1.5 days vs 5 days
- ✅ **3x faster inference**: Important for production
- ✅ **Proven architecture**: Thin-head baseline validated (AP50:95 0.3063)
- ✅ **Can iterate faster**: Multiple experiments per week
- ✅ **Params-matched**: Fair comparison with baseline

**Cons of ResNet18**:
- ❓ Might have slightly lower quality than ViT (unproven)
- ❌ Lose 15 epochs of ViT investment

**When to Switch**:
- If epoch 20 metrics are bad (heatmap >1.5, not converging)
- If we need to iterate faster (run multiple ablations)
- If inference speed is critical

### ResNet18 Architecture (Backup Plan)
```
Image Branch:     ResNet18 → 512-dim (11M params, stride 8)
                      ↓
                Learned Alpha Blend ← Transformer (4L-128D-4H, 4M params)
                      ↓                        ↑
                CenterNet Decoder           SVG Strokes
                (35 heads, 5M params)
```

**Total Parameters**: 20M (2.8x smaller than ViT)  
**Expected Speed**: ~0.8s/batch (3.3x faster)  
**Expected Quality**: AP50:95 ≥ 0.28 (target)

### Commands to Switch

```bash
# 1. Stop current ViT training
# (Ctrl+C in training terminal)

# 2. Save current checkpoint for reference
cp checkpoints/dual_stream_100ep/epoch_15.pt \
   checkpoints/floorplan_llm_vit_epoch15_backup.pt

# 3. Checkout ResNet18 architecture
git checkout cdec314 -- src/models/

# 4. Merge with current improvements (batch_size, workers)
# (manual merge needed)

# 5. Start ResNet18 training
.venv/Scripts/python.exe train_dual_stream_extended.py \
  --preset small \
  --epochs 100 \
  --batch_size 8 \
  --num_workers 4 \
  --lr 1e-4 \
  --warmup_epochs 5 \
  --output_dir checkpoints/dual_pathway_resnet18_100ep

# Expected completion: 1.5 days vs 5 days for ViT
```

---

## 📈 Research Hypotheses (UPDATED)

### H1: Current Training (FloorPlanLLM ViT) → Quality Unknown
**Status**: 🔄 In progress (epoch 15/100)  
**Claim**: ViT will achieve AP50:95 > ResNet18 despite 3x slowdown  
**Test**: Continue to convergence, benchmark on val_select  
**Expected gain**: Unknown (could be +0.05 vs ResNet18, or worse)  
**Risk**: High - 5 days training, uncertain quality, 3x slower inference  
**Decision**: Epoch 20 gate determines continue/abort

---

### H1-alt: Dual-Pathway ResNet18 → Fast Baseline
**Status**: 🟡 Not trained (backup plan)  
**Claim**: ResNet18 achieves AP50:95 ≥ 0.28 with 3x faster speed  
**Test**: 100 epochs with same training config as ViT  
**Expected gain**: Baseline +0.13-0.15 AP50:95 (vs thin-head 0.3063)  
**Risk**: Low - proven architecture, faster iteration  
**Timeline**: 1.5 days vs 5 days for ViT

---

### H2: Cross-Attention Fusion (After baseline converges)
**Status**: 🟡 Prepared, waiting for H1  
**Claim**: Cross-attention > learned blend  
**Expected**: +0.02-0.05 AP50  
**Speed**: +20-30% latency

---

### H3: Vector Encoder Scaling
**Status**: 🟡 Prepared  
**Expected**: +0.01-0.03 AP50  
**Speed**: Variable (-20% to +15%)

---

### H4: Hybrid Stride-4 Head
**Status**: 🟢 Independent (audit ready)  
**Pre-audit**: Run collision audit on val_select  
**Expected**: +0.03-0.05 AP50 if collisions >10%  
**Speed**: +40% latency for small objects

```bash
.venv/Scripts/python.exe scripts/audit_stride8_collisions.py \
  --split val_select \
  --output outputs/stride8_collision_audit.json
```

---

### H5: Knowledge Distillation
**Status**: 🟡 After baseline converges  
**Target**: 90% teacher AP, 50% latency  
**Teacher**: Best converged model (ViT or ResNet18)  
**Student**: Image-only CenterNet

---

## 📊 Expected Outcomes (REVISED)

### Scenario A: ViT Succeeds (Optimistic)
| Phase | Contribution | Cumulative |
|---|---:|---:|
| H1 ViT Baseline | +0.20 | 0.20 |
| H2 Fusion | +0.04 | 0.24 |
| H4 Stride-4 | +0.04 | 0.28 |
| **Val total** | | **0.28** |
| **Test (with gap)** | | **0.26** |

**Timeline**: 5 days for H1 + 3 days for H2/H4 = 8 days  
**Speed**: 3x slower inference (production concern)

---

### Scenario B: ResNet18 Path (Conservative)
| Phase | Contribution | Cumulative |
|---|---:|---:|
| H1-alt ResNet18 | +0.15 | 0.15 |
| H2 Fusion | +0.03 | 0.18 |
| H4 Stride-4 | +0.03 | 0.21 |
| **Val total** | | **0.21** |
| **Test (with gap)** | | **0.19** |

**Timeline**: 1.5 days for H1 + 3 days for H2/H4 = 4.5 days  
**Speed**: 3x faster inference (production-ready)

---

### Gap to SOTA (Both Scenarios)
- Target: Faster R-CNN AP50 0.602
- Expected: AP50 0.19-0.26
- **Remaining gap**: 0.34-0.41

**Future hypotheses needed** (H7-H10):
- H7: FPN multi-scale (+0.05-0.08)
- H8: Stronger backbone (+0.03-0.05)
- H9: Query augmentation (+0.01-0.02)
- H10: Class-balanced sampling (+0.02)

---

## 📋 Research Protocol Compliance

### What Went Right ✅
- ✅ Split discipline maintained (train_new, val_select, val_calibration, test once)
- ✅ Manifest tracking (schema v2, fingerprint 203f6358)
- ✅ Training infrastructure solid (batch_size=8, workers=4, lr schedule)

### What Went Wrong ❌
- ❌ Architecture change (ResNet18 → ViT) WITHOUT benchmark comparison
- ❌ No controlled A/B test (same config, same manifest)
- ❌ No speed profiling before committing to ViT
- ❌ Paper misapplication (LocateAnything for multi-instance task)
- ❌ ResNet18 never trained to convergence (no baseline)

### How to Fix Going Forward ✅
1. ✅ **Decision gate at epoch 20**: Data-driven choice
2. ✅ **If switch to ResNet18**: Complete 100-epoch baseline first
3. ✅ **Controlled comparison**: Same config for ViT vs ResNet18
4. ✅ **Document trade-offs**: Speed vs quality explicitly measured
5. ✅ **Benchmark before architecture changes**: Never again commit to heavy architecture without proof

---

## 📁 Repository Structure

### Key Implementation Files
```
src/models/
  ├── floorplan_llm.py              # Current ViT architecture
  ├── dual_stream_centernet.py      # Old ResNet18 (commit cdec314)
  ├── image_encoder.py              # ViT + ResNet18 implementations
  ├── vector_encoder.py             # Graph Transformer
  └── centernet_heads.py            # Detection heads

Training & Evaluation:
  ├── train_dual_stream_extended.py # 100-epoch training
  ├── evaluate_dual_stream.py       # Evaluation script
  └── diagnose_dual_stream.py       # Diagnostic tool

Scripts:
  └── scripts/audit_stride8_collisions.py  # H4 pre-audit

Checkpoints:
  ├── dual_stream_100ep/            # FloorPlanLLM ViT (epoch 15/100)
  └── dual_pathway_resnet18_100ep/  # Backup plan (not started)
```

### Documentation
```
docs/
  ├── research_protocol.md                    # Scientific methodology
  ├── ARCHITECTURE_EVOLUTION_ANALYSIS.md      # Why ViT is 3x slower
  ├── ARCHITECTURE_COMPARISON_SUMMARY.md      # Quick reference
  ├── research_status_summary.md              # Executive summary (outdated)
  ├── dual_stream_experiment_report.md        # 10-epoch results
  ├── dual_stream_research_plan.md            # 6-phase plan
  ├── deep_research_next_steps.md             # Hypothesis roadmap
  └── SOTA.md                                 # Benchmark provenance
```

---

## 🚀 Immediate Action Plan

### Now → Epoch 20 (Next 7 Hours)

1. **✅ Monitor training progress**
   ```bash
   tail -f training_bs8_workers4.log
   nvidia-smi  # Check GPU utilization
   ```

2. **📊 Prepare epoch 20 evaluation**
   ```bash
   # Quick validation check (will run automatically if --val_every 5)
   python evaluate_dual_stream.py \
     --checkpoint checkpoints/dual_stream_100ep/epoch_20.pt \
     --split val_select \
     --output outputs/epoch20_quick_check.json
   ```

3. **✅ Read detailed analysis**
   - `docs/ARCHITECTURE_EVOLUTION_ANALYSIS.md` (why 3x slower)
   - `docs/ARCHITECTURE_COMPARISON_SUMMARY.md` (quick reference)

---

### At Epoch 20 (Decision Point)

**Check metrics**:
```bash
# 1. Check training log
tail -50 training_bs8_workers4.log

# 2. Check validation results
cat outputs/epoch20_quick_check.json | grep "AP50"

# 3. Decide based on decision tree (see section above)
```

**Decision**:
- ✅ **Continue ViT**: If heatmap <1.0, loss <15, AP50 >0.05
- ⚠️ **Continue with caution**: If metrics acceptable but not great
- ❌ **Switch to ResNet18**: If not converging or too slow

---

### After Decision

**If Continue ViT**:
- Monitor to epoch 50
- Re-evaluate progress
- Complete to epoch 100 if promising

**If Switch to ResNet18**:
- Stop ViT training
- Save checkpoint for reference
- Restore ResNet18 architecture
- Start fresh 100-epoch training
- ETA: 1.5 days instead of 5 days

---

## ✅ Success Criteria (UPDATED)

### Phase 1: Establish Baseline
- [x] Architecture implemented (ViT version)
- [ ] **Decision at epoch 20**: Continue ViT or switch to ResNet18
- [ ] Training converges (heatmap loss <0.5)
- [ ] AP50 >0.15 on val_select
- [ ] Reproducible checkpoint documented

### Phase 2: Architecture Selection
- [ ] Controlled comparison: ViT vs ResNet18 (if both trained)
- [ ] Best architecture selected via validation AP
- [ ] Speed/accuracy trade-off measured and documented

### Phase 3-6: (Unchanged from original plan)

---

## 📊 Risk Assessment (UPDATED)

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| ViT doesn't converge well | Medium | High | Epoch 20 decision gate |
| ViT too slow for production | High | Medium | Have ResNet18 backup ready |
| ResNet18 never benchmarked | High | High | Train ResNet18 if ViT fails |
| Can't reach 0.60 AP50 | Medium | Low | Document progress; H7-H10 next |
| Wasted 5 days on ViT | Low-Medium | Medium | Epoch 20 early abort if needed |

---

## 🎯 Timeline (REVISED)

### Current Path (ViT)
| Phase | Duration | Status | Blocking? |
|---|---|---|---|
| H1: ViT training | 5 days (85 epochs left) | 🔄 Epoch 15/100 | YES |
| Decision gate | 7 hours (epoch 20) | 🔴 Critical | — |
| H2-H3: Ablations | 3-5 days | 🟡 After H1 | No |
| H4: Stride-4 | 2-3 days | 🟢 Independent | No |
| H5: Distillation | 2-3 days | 🟡 After H1 | No |
| H6: Test | 4 hours | Final | All above |

**Total with ViT**: 10-15 days from now

---

### Alternative Path (ResNet18)
| Phase | Duration | Status | Blocking? |
|---|---|---|---|
| H1-alt: ResNet18 | 1.5 days | 🟡 Backup plan | YES |
| H2-H3: Ablations | 3-5 days | 🟡 After H1 | No |
| H4: Stride-4 | 2-3 days | 🟢 Independent | No |
| H5: Distillation | 2-3 days | 🟡 After H1 | No |
| H6: Test | 4 hours | Final | All above |

**Total with ResNet18**: 6-11 days from now (3-4 days faster)

---

## 📝 Key Takeaways

1. **Current status**: Training FloorPlanLLM (ViT) at epoch 15/100, 3x slower than expected
2. **Root cause**: ViT replaces CNN, self-attention O(n²) vs convolution O(n)
3. **Decision point**: Epoch 20 (7 hours) - continue ViT or switch to ResNet18
4. **Quality unknown**: No benchmark yet for either ViT or ResNet18
5. **Research protocol violation**: Architecture change without controlled comparison
6. **Fix**: Establish decision gate, prepare ResNet18 backup, complete one baseline first

**Next critical milestone**: Epoch 20 evaluation and decision (7 hours from now)

---

**Status**: Training in progress. Epoch 20 decision determines research direction for next 1-2 weeks.
