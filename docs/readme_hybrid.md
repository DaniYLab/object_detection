# Hybrid Inference Experiment

Implements confidence-based fallback strategy from LocateAnything paper for FloorPlanCAD detection.

## Overview

**Goal:** Improve robustness on ambiguous cases while maintaining speed through hybrid parallel/sequential inference.

**Status:** Phase 1 training in progress (~42% complete)

## Architecture

### Phase 1: Parallel Decoder Baseline
- Decode all 35 classes simultaneously (single forward pass)
- 2.2M parameters (lightweight compared to thin-head 6.8M)
- Fast inference baseline before adding hybrid logic

### Phase 2: Hybrid Inference Logic
- Wrap Phase 1 with confidence monitoring
- **Fallback triggers:**
  1. Low confidence: `max(sigmoid(heatmap)) < 0.7`
  2. Spatial ambiguity: multiple peaks within 8px
- Collect fallback statistics during inference

### Phase 3-4: Evaluation & Calibration
- Measure fallback rate on val_select
- Tune thresholds on val_calibration
- Compare AP with thin-head baseline (AP50:95 = 0.3063)

## Quick Start

### Manual Pipeline
```bash
# Phase 1: Train parallel decoder
python train_hybrid_phase1.py --epochs 3 --batch_size 4

# Phase 2: Add hybrid logic
python train_hybrid_phase2.py \
  --phase1_checkpoint checkpoints/hybrid_phase1/epoch_3.pt \
  --epochs 5

# Phase 3: Evaluate
python evaluate_hybrid.py \
  --checkpoint checkpoints/hybrid_phase2/epoch_5.pt \
  --split val_select

# Phase 4: Analyze
python analyze_hybrid.py \
  --hybrid outputs/hybrid_val_select.json

# Summary
python summarize_hybrid.py
```

### Automatic Pipeline
```bash
# Start Phase 1 manually
python -u train_hybrid_phase1.py --epochs 3 --batch_size 4 > phase1_full.log 2>&1 &

# Auto-run Phase 2-4 when Phase 1 completes
bash auto_pipeline.sh
```

### Monitoring
```bash
# Watch Phase 1 progress
bash monitor_phase1.sh

# Or tail log directly
tail -f phase1_full.log
```

## Expected Results

### Speed
- **Parallel mode:** 3-5× faster than sequential
- **Fallback rate:** 10-20% (target)
- **Overall speedup:** ~2.5-3× net gain

### Quality
- **Conservative target:** Match thin-head (AP50:95 = 0.3063)
- **Aggressive target:** +1-2 points (AP50:95 = 0.31-0.32)
- **Mechanism:** Better handling of ambiguous layouts

### Robustness
- Reduced false positives on low-confidence predictions
- Better multi-object disambiguation
- Per-class fallback analysis reveals which classes benefit

## Baseline Comparison

**Thin-head (dual_pathway_v2_thin):**
- Seed 1337: AP50 = 0.4950, AP50:95 = 0.3091
- Seed 2024: AP50 = 0.4882, AP50:95 = 0.3035
- **Mean (2 seeds): AP50:95 = 0.3063**

**Hybrid inference (target):**
- Maintain or exceed thin-head quality
- 3× faster inference (parallel decoding)
- Lower parameter count (2.2M vs 6.8M)

## Files

### Training Scripts
- `train_hybrid_phase1.py` — Parallel decoder baseline
- `train_hybrid_phase2.py` — Hybrid inference wrapper
- `HybridInferenceDetector` class in phase2 script

### Evaluation & Analysis
- `evaluate_hybrid.py` — Evaluation with fallback stats
- `analyze_hybrid.py` — Compare with baseline
- `summarize_hybrid.py` — Generate summary report

### Automation
- `run_hybrid_pipeline.sh` — Manual full pipeline
- `auto_pipeline.sh` — Auto-run Phase 2-4 after Phase 1
- `monitor_phase1.sh` — Training progress monitor

### Documentation
- `docs/hybrid_strategy.md` — Detailed strategy & timeline
- `README_HYBRID.md` — This file

## Protocol Compliance

✓ Image-level splits (v2 manifest)  
✓ train_new for training  
✓ val_select for checkpoint selection  
✓ val_calibration for threshold tuning  
✓ test_set untouched (final report only)  
✓ Manifest fingerprint tracking  

## Timeline

| Phase | Duration | Status |
|---|---|---|
| Phase 1: Parallel baseline | 30 min | 🔄 In progress (42%) |
| Phase 2: Hybrid training | 1 hour | ⏸️ Waiting |
| Phase 3: Evaluation | 20 min | ⏸️ Waiting |
| Phase 4: Analysis | 10 min | ⏸️ Waiting |
| **Total** | **~2 hours** | |

## Success Criteria

### Must-have:
- [x] Phase 1 training converges
- [ ] Phase 2 training stable
- [ ] Fallback mechanism triggers correctly (10-30% rate)
- [ ] val_select evaluation completes

### Quality targets:
- [ ] AP50:95 ≥ 0.3063 (thin-head parity)
- [ ] Fallback rate 10-20%
- [ ] Interpretable per-class patterns

### Publication-ready:
- [ ] AP50:95 ≥ 0.31 (+1 point)
- [ ] 2-seed replication
- [ ] Ablation study

## Next Steps

1. **Wait for Phase 1** (~15 minutes remaining)
2. **Launch Phase 2** via auto_pipeline.sh
3. **Evaluate** on val_select and val_calibration
4. **Analyze** fallback patterns and AP comparison
5. **Decision:**
   - If AP ≥ 0.31: Replicate with seed 2024
   - If 0.30 ≤ AP < 0.31: Speed contribution only
   - If AP < 0.30: Analyze failure modes, iterate

## Contact & References

**LocateAnything paper:** Resource/LocateAnything.pdf  
**Teacher data:** data/Teacher/teacher_distill.jsonl  
**Baseline results:** docs/experiment_log.md  
**Protocol:** docs/research_protocol.md
