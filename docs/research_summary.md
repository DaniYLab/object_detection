# FloorPlanCAD Object Detection - Complete Research Summary

**Date:** 2026-09-05  
**Status:** Training complete, evaluation in progress  
**Goal:** Scientifically rigorous improvements targeting SOTA detection benchmarks

---

## 🎯 Research Objectives

### Primary Goals
1. **Match/exceed detection baseline:** AP50 ≥ 0.602 (Faster R-CNN), ideally approach 0.649 (DINO)
2. **Improve inference speed:** 2-5× faster through architectural innovations
3. **Maintain scientific rigor:** Isolate causal effects, validate on val before test, exact protocol matching

### Current Baseline
**Thin-head (dual_pathway_v2_thin):**
- AP50: 0.4916 (mean of 2 seeds: 1337, 2024)
- AP50:95: 0.3063
- Architecture: Dual-pathway with per-class thin heads
- Parameters: 6.8M

---

## 🏗️ Architectures Implemented

### 1. DualStreamCenterNet (COMPLETED ✓)
**Status:** Training complete (10 epochs), evaluation running  
**Motivation:** Unified image+vector dual-stream from teacher reasoning data  
**Innovation:** Simultaneous raster and vector primitive processing

**Architecture:**
- Image branch: ResNet-50 backbone → 256-dim features
- Vector branch: Transformer encoder (stroke tokens) → 256-dim features
- Fusion: Cross-attention between modalities
- Decoder: CenterNet heatmap prediction
- Parameters: 56.5M

**Training Results (10 epochs):**
```
Epoch 1: Loss 29.99 | Heatmap 2.71
Epoch 5: Loss 24.68 | Heatmap 2.25
Epoch 10: Loss 22.85 | Heatmap 2.09 ← Final
```

**Evaluation:** Running on val_select (background task brq8blsvy)

**Key Files:**
- `train_dual_stream_centernet.py` - Training script
- `evaluate_dual_stream.py` - Evaluation script
- `src/models/dual_stream_centernet.py` - Model definition
- `checkpoints/dual_stream_centernet/epoch_*.pt` - 10 checkpoints

---

### 2. Hybrid Inference Strategy (IN PROGRESS)
**Status:** Phase 1 training complete, auto-pipeline orchestrating Phase 2-4  
**Motivation:** LocateAnything paper - parallel decoding with confidence-based fallback  
**Innovation:** 3-5× speed via parallel inference, robust fallback for ambiguous cases

**Architecture:**
- **Phase 1:** Parallel decoder (all 35 classes simultaneously) - 2.2M params
- **Phase 2:** Hybrid wrapper with confidence monitoring
  - Normal: Decode all classes in single pass
  - Fallback: Sequential when confidence < 0.7 or spatial ambiguity
- **Phase 3-4:** Evaluation and analysis

**Training Progress:**
```
Phase 1: ✓ Complete (3 epochs, loss converged to 11.9)
Phase 2: 🔄 Auto-pipeline running (ETA ~1 hour)
Phase 3: ⏸️ Queued (evaluation)
Phase 4: ⏸️ Queued (analysis vs baseline)
```

**Success Criteria:**
- Conservative: AP50:95 ≥ 0.3063 (thin-head parity) + 2-3× speedup
- Aggressive: AP50:95 ≥ 0.31 (+1 point improvement)
- Interpretable: Clear per-class fallback patterns

**Key Files:**
- `train_hybrid_phase1.py` - Parallel decoder training
- `train_hybrid_phase2.py` - Hybrid wrapper training
- `evaluate_hybrid.py` - Evaluation with fallback stats
- `analyze_hybrid.py` - Compare with baseline
- `auto_pipeline.sh` - Automatic orchestration
- `docs/hybrid_strategy.md` - Detailed strategy document

**Background Tasks:**
- Auto-pipeline: Task bg0qnoyk6 (running)
- Expected completion: ~2 hours from start (~12:35 PM)

---

## 📊 Expected Results Timeline

### Today (2026-09-05)
| Time | Event | Output |
|---|---|---|
| 13:15 | DualStreamCenterNet eval complete | `outputs/dual_stream_val_select.json` |
| 14:00 | Hybrid Phase 2 complete | `checkpoints/hybrid_phase2/` |
| 14:25 | Hybrid evaluation complete | `outputs/hybrid_val_select.json` |
| 14:35 | Hybrid analysis complete | `outputs/hybrid_analysis.json` |

### Decision Points
**After DualStreamCenterNet results:**
- If AP50:95 > 0.31: Strong improvement, replicate with seed 2024
- If 0.25-0.31: Promising, analyze per-class patterns
- If < 0.25: Investigate failure modes, consider architecture changes

**After Hybrid results:**
- If AP50:95 ≥ 0.31: Speed+quality win, replicate and document
- If 0.30-0.31: Speed contribution validated, analyze fallback patterns
- If < 0.30: Iterate on thresholds or decoder capacity

---

## 🔬 Research Protocol Compliance

✓ **v2 Manifest:** Frozen fingerprint `203f63588e906637f4dbcf6d1a001e67d0e688016c9b0c799c3ea89b438e8e9a`  
✓ **Image-level splits:** Before query expansion  
✓ **train_new:** 15,196 samples for training  
✓ **val_select:** 1,016 images for checkpoint selection  
✓ **val_calibration:** For decoder threshold tuning  
✓ **test_set:** Untouched, reserved for final report  
✓ **Fingerprint tracking:** All outputs carry manifest provenance  
✓ **Multi-seed requirement:** Acknowledged for publication-ready claims  

---

## 📈 Benchmark Context

### Detection Benchmarks (Same Task)
| Model | AP50 | mAP | Notes |
|---|---:|---:|---|
| Faster R-CNN R101 | 0.602 | 0.452 | FloorPlanCAD paper baseline |
| FCOS | 0.624 | 0.453 | |
| YOLOv3 | 0.639 | 0.413 | |
| **DINO** | **0.649** | — | DPSS paper (ArchCAD-400K) |
| Thin-head (ours) | 0.492 | — | Current baseline (2 seeds) |
| **Target** | **≥0.60** | **≥0.45** | Match Faster R-CNN |

### Segmentation Benchmarks (Different Task - Reference Only)
| Model | F1 | wF1 | Task |
|---|---:|---:|---|
| DeepLabv3+ | 0.688 | 0.714 | Semantic segmentation |
| DPSS | 0.920 | 0.932 | Symbol spotting (image+vector+DETR) |

**Note:** Cannot directly compare box AP with segmentation F1. DPSS is the closest architectural inspiration (dual-pathway fusion).

---

## 🚀 Next Scientifically Rigorous Experiments

### High Priority (After Current Results)

#### 1. Decoder Architecture Upgrade
**Hypothesis:** CenterNet decoder limits quality; DETR/Mask2Former may improve  
**Motivation:** DPSS paper shows DETR decoder achieves AP50 0.649 vs Faster R-CNN 0.602  
**Approach:**
- Keep dual-stream encoder (proven by DPSS)
- Replace CenterNet head with DETR decoder
- Target: AP50 ≥ 0.60 (Faster R-CNN parity)

**Ablation Required:**
- Baseline: Current CenterNet decoder
- Test: DETR decoder on same encoder
- Control: Same training data, same evaluation protocol

#### 2. Vector Encoder Enhancement
**Hypothesis:** Primitive-aware encoder improves geometric understanding  
**Motivation:** DPSS uses PointTransformerV2 for vector branch  
**Approach:**
- Upgrade stroke encoder to PointTransformer
- Add geometric attention (line/arc/circle awareness)
- Target: +2-3 AP points on geometric classes (door, window)

**Ablation Required:**
- Baseline: Current transformer encoder
- Test: PointTransformer encoder
- Control: Same image branch, same decoder

#### 3. Class-Balanced Training
**Hypothesis:** Rare-class undersampling hurts quality  
**Motivation:** Historical diagnostics show escalator (121:1 pred:GT), column (39:1 pred:GT)  
**Approach:**
- Implement balanced sampler (per-class equal probability)
- Target: Reduce FP rate on rare classes
- Measure: Per-class precision improvement

**Protocol:** Only use val diagnostics for tuning, never test observations

#### 4. Multi-Scale Feature Pyramid
**Hypothesis:** Single-scale features miss small objects (escalator, bathtub)  
**Motivation:** Standard FPN pattern from detection literature  
**Approach:**
- Add FPN to image branch
- Multi-scale heatmap prediction
- Target: +5-10 AP points on small objects

#### 5. Data Augmentation Ablation
**Hypothesis:** Current augmentation may be insufficient/harmful  
**Motivation:** Teacher data shows diverse scale/rotation patterns  
**Approach:**
- Baseline: Current augmentation
- Test: {none, light, medium, heavy} augmentation
- Measure: val_select AP (prevent overfitting)

---

## 🔍 Deep Analysis Required (After Results Available)

### Per-Class Diagnostic Analysis
1. **Which classes benefit from dual-stream?**
   - Compare image-only vs dual-stream on geometric classes
   - Expected: door, window, wall show largest gains

2. **Which classes trigger hybrid fallback?**
   - Analyze confidence distributions per class
   - Expected: Rare/ambiguous classes fallback more

3. **Correlation with spatial density?**
   - High-density images → more fallback?
   - Measure: fallback rate vs objects-per-image

### Error Mode Analysis
1. **False positives:** Which classes over-predict?
2. **False negatives:** Which classes under-predict?
3. **Localization errors:** IoU distribution analysis
4. **Confidence calibration:** Score vs actual precision curve

### Speed-Quality Tradeoff
1. **Hybrid inference:** Speedup vs AP tradeoff curve
2. **Batch size impact:** Throughput vs quality
3. **Decoder complexity:** Parameters vs AP correlation

---

## 📁 Repository Structure

```
object_detection/
├── src/
│   ├── models/
│   │   ├── dual_stream_centernet.py          ✓ Dual-stream architecture
│   │   └── hybrid_inference.py               ✓ Hybrid wrapper
│   ├── data/
│   │   ├── dataset.py                        ✓ Query dataset
│   │   └── splits.py                         ✓ v2 manifest handling
│   ├── training/
│   │   └── losses.py                         ✓ Focal + L1 losses
│   ├── evaluation/
│   │   └── metrics.py                        ✓ AP computation
│   └── postprocessing/
│       └── decode.py                         ✓ CenterNet decoder
├── train_dual_stream_centernet.py            ✓ Training script
├── evaluate_dual_stream.py                   ✓ Evaluation script
├── train_hybrid_phase1.py                    ✓ Hybrid phase 1
├── train_hybrid_phase2.py                    ✓ Hybrid phase 2
├── evaluate_hybrid.py                        ✓ Hybrid evaluation
├── analyze_hybrid.py                         ✓ Hybrid analysis
├── auto_pipeline.sh                          ✓ Auto-orchestration
├── checkpoints/
│   ├── dual_stream_centernet/                ✓ 10 epochs
│   └── hybrid_phase1/                        ✓ 3 epochs
├── docs/
│   ├── research_protocol.md                  ✓ Protocol definition
│   ├── sota_gap_plan.md                      ✓ Benchmark context
│   ├── hybrid_strategy.md                    ✓ Hybrid approach
│   └── experiment_log_hybrid.md              ✓ Current experiment log
└── data/
    └── FloorPlanCAD_original/
        └── splits_v2.json                    ✓ Frozen manifest
```

---

## 🎓 Key Research Principles Applied

### 1. Causal Isolation
- Each ablation changes ONE variable
- Baseline always measured on same split
- Never conflate architecture + data + training changes

### 2. Validation-First
- All model selection on val_select
- All decoder tuning on val_calibration
- test_set only for final report (one-shot)

### 3. Exact Protocol Matching
- Same manifest fingerprint across all runs
- Same image preprocessing (resize, normalization)
- Same evaluation metrics (AP50, AP50:95, per-class)

### 4. Reproducibility
- All seeds tracked
- All checkpoints saved with full provenance
- Multi-seed replication for publication claims

### 5. Benchmark Alignment
- Compare only with same-task baselines (box detection)
- Acknowledge metric differences (AP vs F1)
- Verify split/preprocessing before claiming superiority

---

## ✅ Completed Today (2026-09-05)

1. ✓ DualStreamCenterNet architecture designed and implemented
2. ✓ Training completed (10 epochs, loss converged to 22.85)
3. ✓ Evaluation pipeline built (AP50, AP50:95, per-class metrics)
4. ✓ Hybrid inference strategy designed (parallel + fallback)
5. ✓ Hybrid Phase 1 training completed (parallel decoder)
6. ✓ Auto-pipeline orchestration (Phase 2-4 automatic)
7. ✓ CenterNet decoder with NMS
8. ✓ Comprehensive documentation (protocol, strategy, logs)

## ⏳ In Progress

- 🔄 DualStreamCenterNet evaluation (task brq8blsvy)
- 🔄 Hybrid auto-pipeline (Phase 2-4, task bg0qnoyk6)

## 📋 Next Actions (After Results)

1. **Review DualStreamCenterNet results** (~15 min)
   - If AP50:95 > 0.31: Replicate with seed 2024
   - If < 0.31: Analyze per-class patterns

2. **Review Hybrid results** (~2 hours)
   - Fallback rate analysis
   - Speed-quality tradeoff
   - Per-class confidence patterns

3. **Prioritize next experiment** based on findings
   - If dual-stream strong: Upgrade to DETR decoder
   - If hybrid strong: Scale to full thin-head capacity
   - If both weak: Investigate training dynamics

4. **Document findings** in experiment_log.md
   - Results summary
   - Insights gained
   - Next experiment rationale

---

## 🎯 Success Metrics

### Minimum Viable (Go/No-Go)
- [ ] DualStreamCenterNet converges (loss < 30) ← ✓ Complete (22.85)
- [ ] Hybrid Phase 1 converges ← ✓ Complete (11.9)
- [ ] Hybrid Phase 2 stable ← 🔄 In progress
- [ ] Evaluations complete without errors

### Quality Targets
- [ ] AP50:95 ≥ 0.31 (thin-head +1 point)
- [ ] Per-class AP improvements on key categories
- [ ] Interpretable error patterns

### Publication-Ready
- [ ] 2-seed replication with consistent results
- [ ] Controlled ablation study (isolate causal effects)
- [ ] Speed measurements (FLOPs, latency, throughput)
- [ ] Final test_set evaluation (one-shot, after all selection frozen)

---

**Status:** Experiments running, awaiting results for next decision point.  
**ETA:** DualStream results ~13:15, Hybrid results ~14:35  
**Contact:** Check task outputs for real-time progress
