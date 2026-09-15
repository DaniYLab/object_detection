# Hybrid Inference Experiment - Quick Reference

## Current Status
**Date:** 2026-09-05  
**Phase 1:** 74% complete (batch 2800/3799, epoch 1/3)  
**ETA:** ~45 minutes until Phase 1 complete, then 90 minutes for Phase 2-4  

## What We're Testing
Parallel box decoding with confidence-based fallback from LocateAnything paper.

## Key Innovation
- **Normal case:** Decode all 35 classes in single forward pass (fast)
- **Uncertain case:** Fall back to sequential when confidence < 0.7 or peaks too close
- **Goal:** 3× speed + better robustness

## Baseline to Beat
Thin-head (dual_pathway_v2_thin):
- AP50:95 = 0.3063 (mean of 2 seeds)
- Speed: Standard CenterNet inference

## Success = Any of:
1. AP50:95 ≥ 0.31 (+1 point improvement)
2. AP50:95 ≥ 0.30 with 3× speedup
3. Interpretable fallback patterns for future work

## Files to Monitor
- `phase1_full.log` - Phase 1 training
- `auto_pipeline.log` - Auto-pipeline progress
- `outputs/hybrid_val_select.json` - Final results

## Commands
```bash
# Watch progress
tail -f phase1_full.log

# Dashboard
bash dashboard.sh

# Summary (after complete)
python summarize_hybrid.py
```

## Timeline
- Phase 1: 30 min (parallel decoder training)
- Phase 2: 50 min (hybrid wrapper training)  
- Phase 3: 20 min (evaluation)
- Phase 4: 10 min (analysis)
- **Total: ~2 hours**

## Auto-Execution
✓ Auto-pipeline running in background  
✓ Will automatically run Phase 2-4 when Phase 1 completes  
✓ No manual intervention needed  

## Next Decision Point
After results available (~2 hours):
- If good: Replicate with seed 2024
- If mixed: Analyze fallback patterns
- If poor: Try different thresholds or architecture
