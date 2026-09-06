# Hybrid Inference Strategy: Implementation Report

**Date:** 2026-09-05  
**Status:** Phase 1 in progress  
**Expected completion:** ~30 minutes

## 1. Motivation

From LocateAnything paper analysis, hybrid inference provides:
- **Primary benefit:** Robustness on ambiguous cases without sacrificing speed
- **Secondary benefit:** Graceful degradation when parallel decoding uncertain
- **Mechanism:** Confidence-based fallback from parallel to sequential mode

## 2. Implementation Architecture

### Phase 1: Parallel Decoder Baseline (in progress)
- Simple parallel decoder: all 35 classes decoded simultaneously
- 2.2M parameters (lightweight)
- Training: 3 epochs, batch size 4
- ETA: ~30 minutes total
- Purpose: Establish fast baseline before adding hybrid logic

### Phase 2: Hybrid Inference Logic (next)
- Wrap Phase 1 model with confidence monitoring
- Two fallback triggers:
  1. **Low confidence:** max(sigmoid(heatmap)) < 0.7
  2. **Spatial ambiguity:** multiple peaks within 8px distance
- Collect statistics: parallel_count vs fallback_count
- Training: 5 epochs with hybrid-aware optimization

### Phase 3: Evaluation (validation-only)
- Measure fallback rate on val_select
- Per-class fallback analysis
- Compare AP with thin-head baseline
- Calibrate thresholds on val_calibration

## 3. Expected Outcomes

### Speed
- **Parallel mode:** 3-5× faster than sequential (all classes in one pass)
- **Hybrid overhead:** Minimal (just threshold checks during inference)
- **Fallback cost:** Only invoked on uncertain cases (~10-20% expected)

### Quality
- **Target:** +1-2 AP50:95 over thin-head baseline (0.3063)
- **Mechanism:** Better handling of ambiguous spatial layouts
- **Risk:** Fallback logic adds complexity; may need threshold tuning

### Robustness
- **Primary gain:** Reduced false positives on low-confidence predictions
- **Secondary gain:** Better multi-object disambiguation
- **Measurable:** Per-class fallback rates reveal which classes benefit most

## 4. Current Progress

```
Phase 1 Training (batch 800/3799, epoch 1/3):
  Initial loss: 3391.4
  Current loss: 28.7
  Convergence: ✓ (loss dropped 100×)
  Speed: 0.16s/batch
  Remaining: ~30 minutes
```

## 5. Next Steps After Phase 1 Completes

1. **Verify Phase 1 checkpoint:**
   - Check `checkpoints/hybrid_phase1/epoch_3.pt` exists
   - Smoke test: single forward pass
   
2. **Launch Phase 2:**
   - Load Phase 1 weights
   - Add HybridInferenceDetector wrapper
   - Train 5 epochs with hybrid logic
   
3. **Evaluate on val_select:**
   - Measure fallback rate
   - Compute AP metrics
   - Compare with thin-head (seed 1337: AP50:95 = 0.3091)
   
4. **Calibrate on val_calibration:**
   - Sweep confidence_threshold: [0.5, 0.6, 0.7, 0.8]
   - Sweep ambiguity_threshold: [4, 8, 12, 16] pixels
   - Select best on val_calibration metrics

## 6. Protocol Compliance

- ✓ Image-level splits (v2 manifest)
- ✓ train_new for training
- ✓ val_select for checkpoint selection
- ✓ val_calibration for threshold tuning
- ✓ test_set untouched (final report only)
- ✓ No test exposure during development
- ✓ Manifest fingerprint tracking

## 7. Integration with Existing Results

### Thin-head baseline (dual_pathway_v2_thin):
- Seed 1337: AP50 = 0.4950, AP50:95 = 0.3091
- Seed 2024: AP50 = 0.4882, AP50:95 = 0.3035
- **Mean (2 seeds): AP50:95 = 0.3063**

### Hybrid inference target:
- Conservative: Match thin-head quality with 3× speed
- Aggressive: +1-2 AP50:95 with maintained speed
- Success gate: ≥2 seeds for architecture claim

## 8. Risk Mitigation

**Risk 1: Fallback too frequent**
- Monitor: If fallback_rate > 50%, thresholds too conservative
- Mitigation: Calibrate on val_calibration before claiming benefit

**Risk 2: No quality gain**
- Monitor: If AP50:95 ≤ thin-head baseline, parallel decoder insufficient
- Mitigation: This becomes speed-only contribution; still valuable

**Risk 3: Implementation complexity**
- Monitor: Fallback logic must not break existing CenterNet decoder
- Mitigation: Wrapper pattern preserves base model unchanged

## 9. Success Metrics

### Must-have (go/no-go):
- [ ] Phase 1 training converges (loss < 50)
- [ ] Phase 2 training stable (no NaN)
- [ ] Fallback mechanism triggers correctly (10-30% rate)
- [ ] val_select evaluation completes

### Nice-to-have (quality):
- [ ] AP50:95 ≥ 0.3063 (thin-head parity)
- [ ] Fallback rate < 20% (mostly parallel)
- [ ] Per-class fallback reveals interpretable patterns

### Stretch (publication-ready):
- [ ] AP50:95 ≥ 0.31 (+1 point over thin-head)
- [ ] 2-seed replication
- [ ] Ablation: parallel-only vs hybrid vs sequential-only

## 10. Timeline

| Phase | Duration | Status |
|---|---|---|
| Phase 1: Parallel baseline | 30 min | 🔄 In progress (27% complete) |
| Phase 2: Hybrid training | 1 hour | ⏸️ Waiting |
| Phase 3: Evaluation | 20 min | ⏸️ Waiting |
| Phase 4: Calibration | 30 min | ⏸️ Waiting |
| **Total** | **~2.5 hours** | |

**Updated ETA:** ~2 hours from now (assuming no errors)

## 11. Files Created

- `train_hybrid_phase1.py` — Parallel decoder baseline training
- `train_hybrid_phase2.py` — Hybrid inference wrapper training
- `evaluate_hybrid.py` — Evaluation with fallback statistics
- `analyze_hybrid.py` — Comprehensive analysis vs baseline
- `run_hybrid_pipeline.sh` — Full orchestration script
- `docs/hybrid_strategy.md` — This document

## 12. Open Questions

1. **Threshold sensitivity:** How much does (0.6 vs 0.7 vs 0.8) affect fallback rate?
2. **Class imbalance:** Do rare classes (elevator, ramp) use fallback more?
3. **Spatial patterns:** Does fallback correlate with room density or overlap?
4. **Speed/quality tradeoff:** Is 3× speed worth 0.5 AP drop (if happens)?

These will be answered during Phase 3 evaluation and Phase 4 calibration.

---

**Next update:** After Phase 1 completes (~30 minutes)
