# FloorPlanCAD Detection Research - Documentation Index

**Last Updated**: 2026-09-06 00:45  
**Status**: Training epoch 17/100, stable, decision gate at epoch 20

---

## 🚨 START HERE

### 1. Current Ground Truth ⭐ **READ THIS FIRST**
**[GROUND_TRUTH.md](GROUND_TRUTH.md)** - Verified status, eliminates confusion from outdated docs
- ✅ Training stable at epoch 17/100
- ✅ Architecture: FloorPlanLLM (ViT, 56.5M params)
- ✅ ETA: 4.75 days to completion
- 🚦 Decision gate: Epoch 20 (3 hours)

### 2. Big Picture Overview
**[BIG_PICTURE.md](BIG_PICTURE.md)** - Complete research roadmap (UPDATED 2026-09-06)
- Research objectives and SOTA benchmarks
- Current architecture comparison (ViT vs ResNet18)
- Decision tree for epoch 20
- Expected outcomes and timeline

### 3. Quick Reference
**[TRAINING_OPTIMIZED.md](TRAINING_OPTIMIZED.md)** - Current training config
- Batch size 8, workers 4
- Why this config works (3x speedup from bs=4)
- Expected completion timeline

---

## 📊 Research Analysis (NEW - 2026-09-06)

### Architecture Analysis
1. **[docs/ARCHITECTURE_EVOLUTION_ANALYSIS.md](docs/ARCHITECTURE_EVOLUTION_ANALYSIS.md)** ⭐ **DEEP ANALYSIS**
   - Complete architecture history (3 phases)
   - Why FloorPlanLLM is 3x slower than ResNet18
   - Paper misapplication analysis (LocateAnything, VGent)
   - Root cause: ViT self-attention O(n²) vs CNN O(n)
   - Research protocol violations documented
   - **10,000+ words, comprehensive**

2. **[docs/ARCHITECTURE_COMPARISON_SUMMARY.md](docs/ARCHITECTURE_COMPARISON_SUMMARY.md)** - Quick reference
   - Architecture comparison table
   - Decision gate criteria (epoch 20)
   - Recommended actions (continue ViT vs switch to ResNet18)

---

## 📚 Research Documentation

### Research Protocol & Planning
1. **[docs/research_protocol.md](docs/research_protocol.md)** - Scientific methodology
   - Split hygiene (train/val/test)
   - Manifest tracking (schema v2, fingerprint 203f6358)
   - Validation-first discipline
   - One-shot test evaluation

2. **[docs/research_status_summary.md](docs/research_status_summary.md)** ⚠️ Outdated
   - Executive summary of research plan
   - 6-phase hypothesis-driven roadmap (H1-H6)
   - **Note**: Created before ViT switch, some info outdated

3. **[docs/dual_stream_research_plan.md](docs/dual_stream_research_plan.md)** - Detailed plan
   - Phase 1-6 breakdown
   - Architecture ablations planned
   - Training efficiency improvements

4. **[docs/deep_research_next_steps.md](docs/deep_research_next_steps.md)** - Hypothesis roadmap
   - H1-H6 hypotheses with expected gains
   - Risk assessment
   - Timeline estimates

### Experiment Reports
1. **[docs/dual_stream_experiment_report.md](docs/dual_stream_experiment_report.md)** - 10-epoch results
   - Initial smoke test results
   - Diagnostic analysis (heatmap loss 2.09, AP50 0.0004)
   - Root cause: Severe undertraining, needs 100 epochs

2. **[docs/experiment_log.md](docs/experiment_log.md)** - Historical log
   - All previous experiments and results

### Benchmark Context
1. **[docs/SOTA.md](docs/SOTA.md)** - Benchmark provenance
   - Valid object detection benchmarks (Faster R-CNN AP50 0.602)
   - Segmentation F1 (different task, reference only)
   - Repository baselines (thin-head AP50:95 0.3063)

2. **[docs/sota_gap_plan.md](docs/sota_gap_plan.md)** - Path to SOTA
   - Gap analysis
   - Hypothesis stack to reach 0.60 AP50

---

## 🏗️ Technical Documentation

### Architecture Specs
1. **[docs/dual_pathway_spec.md](docs/dual_pathway_spec.md)** - Original ResNet18 design
   - Lightweight dual-pathway architecture
   - Cross-attention fusion design
   - Parameters budget (≤20M)
   - **Note**: This was the original plan, replaced by ViT

2. **[docs/week1_floorplan_llm.md](docs/week1_floorplan_llm.md)** - FloorPlanLLM design
   - Why ViT was chosen (LocateAnything inspiration)
   - Parallel decoder attempt (failed on multi-instance)
   - Revert to CenterNet decoder
   - **Result**: Heavy encoder + proven decoder

### Dataset & Splits
1. **[docs/split_manifest_v2.md](docs/split_manifest_v2.md)** - Manifest schema
   - Schema v2 design (image-level splits)
   - train_new / val_select / val_calibration / test_set roles
   - Fingerprint tracking

2. **Data structure**: `data/FloorPlanCAD_original/splits_v2.json`
   - Fingerprint: 203f6358...

---

## 📁 Training & Evaluation Scripts

### Training
- `train_dual_stream_extended.py` - 100-epoch training script
  - Learning rate warmup + cosine decay
  - Validation every 5 epochs
  - Checkpoint saving
- `train_dual_stream_centernet.py` - Original 10-epoch script

### Evaluation
- `evaluate_dual_stream.py` - Evaluation pipeline
  - Integration with existing metrics
  - Per-class AP computation
- `diagnose_dual_stream.py` - Diagnostic tool
  - Heatmap analysis
  - Confidence distribution

### Utilities
- `scripts/audit_stride8_collisions.py` - Stride-8 collision audit (H4)
  - Decision gate: If collision >10%, implement stride-4 head

---

## 🗄️ Archived / Outdated Files

### Move to docs/archive/
- **CRITICAL_ISSUE.md** ❌ Outdated (Sept 5, 20:10)
  - Described training instability that's now resolved
  - Training now stable at epoch 17/100
  
- **RESEARCH_COMPLETE.md** ❌ Outdated (Sept 5, 20:30)
  - Said training blocks convergence
  - Reality: Training running successfully

- **RESEARCH_SUMMARY.md** ⚠️ Mixed accuracy
  - Some info correct, some outdated
  - Mentions ResNet-50 (incorrect, it's ViT)
  - Claims training complete (incorrect, at epoch 17/100)

### Delete
- CURRENT_STATUS.md - Unknown date, likely obsolete
- STATUS.md - Unknown date, likely obsolete

---

## 📊 Checkpoints & Outputs

### Checkpoints
```
checkpoints/
├── dual_stream_centernet/         # 10-epoch smoke test (Sept 5)
│   ├── epoch_1.pt
│   ├── ...
│   └── epoch_10.pt
│
└── dual_stream_100ep/             # Current training (Sept 5-9)
    ├── epoch_10.pt                # Resume point
    ├── epoch_15.pt
    └── best_val.pt                # Best validation (auto-saved)
```

### Outputs
```
outputs/
├── dual_stream_val_select.json    # 10-epoch evaluation (AP50 0.0004)
└── stride8_collision_audit.json   # H4 audit (to be run)
```

---

## 📋 Training Logs

### Current
- **training_bs8_workers4.log** ⭐ **LIVE LOG**
  - Epoch 17/100, batch 1300/1900
  - Loss ~16, Heatmap ~1.5
  - Updated every batch

### Historical
- TRAINING_BS4.md - Previous attempt (batch_size 4, died)
- TRAINING_CLEAN.md - Setup notes
- TRAINING_ACTIVE.md - Status snapshot
- TRAINING_STATUS.md - Status snapshot

---

## 🗺️ Repository Structure

```
object_detection/
├── 📄 GROUND_TRUTH.md                      ⭐ START HERE
├── 📄 BIG_PICTURE.md                       Research roadmap
├── 📄 INDEX.md                             This file
├── 📄 TRAINING_OPTIMIZED.md                Current config
│
├── docs/
│   ├── 📊 ARCHITECTURE_EVOLUTION_ANALYSIS.md     ⭐ Deep analysis
│   ├── 📊 ARCHITECTURE_COMPARISON_SUMMARY.md     Quick reference
│   ├── research_protocol.md
│   ├── research_status_summary.md
│   ├── dual_stream_research_plan.md
│   ├── deep_research_next_steps.md
│   ├── dual_stream_experiment_report.md
│   ├── dual_pathway_spec.md
│   ├── week1_floorplan_llm.md
│   ├── SOTA.md
│   └── experiment_log.md
│
├── src/
│   ├── models/
│   │   ├── floorplan_llm.py               Current ViT architecture
│   │   ├── dual_stream_centernet.py       Old ResNet18 (commit cdec314)
│   │   ├── image_encoder.py
│   │   ├── vector_encoder.py
│   │   └── centernet_heads.py
│   ├── data/
│   │   ├── dataset.py
│   │   └── splits.py
│   └── training/
│       └── losses.py
│
├── train_dual_stream_extended.py          100-epoch training
├── evaluate_dual_stream.py
├── diagnose_dual_stream.py
│
├── scripts/
│   └── audit_stride8_collisions.py        H4 collision audit
│
├── checkpoints/
│   ├── dual_stream_centernet/             10-epoch complete
│   └── dual_stream_100ep/                 Epoch 17/100 (current)
│
├── outputs/
│   └── dual_stream_val_select.json
│
└── logs/
    └── training_bs8_workers4.log          ⭐ LIVE LOG
```

---

## 🎯 Reading Order for New Contributors

### Quick Start (30 minutes)
1. **GROUND_TRUTH.md** - Current status (5 min)
2. **BIG_PICTURE.md** - Research roadmap (15 min)
3. **docs/ARCHITECTURE_COMPARISON_SUMMARY.md** - Architecture comparison (10 min)

### Deep Dive (2 hours)
4. **docs/ARCHITECTURE_EVOLUTION_ANALYSIS.md** - Why ViT is slow (30 min)
5. **docs/research_protocol.md** - Methodology (15 min)
6. **docs/dual_stream_experiment_report.md** - 10-epoch results (15 min)
7. **docs/dual_stream_research_plan.md** - Complete plan (30 min)
8. **docs/SOTA.md** - Benchmark context (15 min)

### Full Understanding (1 day)
9. Read all docs/ files
10. Review commit history (bb14bf8, 68d01af, cdec314)
11. Understand paper references (LocateAnything, VGent)
12. Review training logs and checkpoints

---

## 🚦 Current Decision Points

### Epoch 20 (3 hours from now)
**Check**: Heatmap loss, total loss, validation AP50  
**Decide**: Continue ViT, continue with caution, or switch to ResNet18  
**See**: GROUND_TRUTH.md "Decision Gate: Epoch 20"

### Epoch 30 (if epoch 20 is borderline)
**Check**: Convergence trend  
**Decide**: Continue to epoch 100 or abort

### Epoch 100 (4.75 days from now)
**Check**: Final validation AP50:95  
**Compare**: vs thin-head baseline (0.3063)  
**Decide**: Use ViT or switch to ResNet18 for ablations

---

## ❓ FAQ

### Q: Which architecture is actually running?
**A**: FloorPlanLLM with ViT (56.5M params), NOT ResNet18. See GROUND_TRUTH.md.

### Q: Why are some docs outdated?
**A**: Training stability issues were resolved on Sept 5-6. Older docs (CRITICAL_ISSUE, RESEARCH_COMPLETE) reflect past problems that no longer exist.

### Q: Is training working?
**A**: Yes! Epoch 17/100, stable, no crashes since optimization (bs=8, workers=4).

### Q: When will we have results?
**A**: Quick check at epoch 20 (3 hours), full results at epoch 100 (~4.75 days).

### Q: Should we switch to ResNet18?
**A**: Decision at epoch 20. If ViT converges well (heatmap <1.0), continue. If not, switch to ResNet18 (3x faster).

### Q: What's the research goal?
**A**: Match/exceed Faster R-CNN (AP50 0.602) with faster inference. Current baseline: thin-head (AP50:95 0.3063).

---

## 📞 Quick Commands

### Check training progress
```bash
tail -f training_bs8_workers4.log
```

### Check GPU status
```bash
nvidia-smi
```

### Run stride-8 audit (independent)
```bash
.venv/Scripts/python.exe scripts/audit_stride8_collisions.py \
  --split val_select \
  --output outputs/stride8_collision_audit.json
```

### Evaluate at epoch 20
```bash
.venv/Scripts/python.exe evaluate_dual_stream.py \
  --checkpoint checkpoints/dual_stream_100ep/epoch_20.pt \
  --split val_select \
  --output outputs/epoch20_quick_check.json
```

---

## 📝 Documentation TODO

### Cleanup (Next)
- [ ] Move CRITICAL_ISSUE.md, RESEARCH_COMPLETE.md to docs/archive/
- [ ] Delete CURRENT_STATUS.md, STATUS.md
- [ ] Update RESEARCH_SUMMARY.md (architecture, status)
- [ ] Consolidate training status files (7 TRAINING_*.md files)

### Create (After Results)
- [ ] docs/epoch20_decision_report.md - Decision at epoch 20
- [ ] docs/vit_vs_resnet18_comparison.md - If both trained
- [ ] docs/final_results.md - Epoch 100 results

---

**Last Updated**: 2026-09-06 00:45  
**Next Update**: After epoch 20 decision (~3 hours)  
**Maintained by**: Research team  
**Status**: Training stable, documentation up-to-date
