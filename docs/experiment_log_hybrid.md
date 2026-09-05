

---

## Experiment 2026-09-05: Hybrid Inference Strategy

**Date:** 2026-09-05  
**Status:** Phase 1 training in progress  
**Branch:** Hybrid parallel/sequential inference from LocateAnything paper

### Motivation

From LocateAnything paper analysis and teacher reasoning data exploration, hybrid inference provides:
1. **Speed:** 3-5× faster via parallel decoding (all 35 classes at once)
2. **Robustness:** Confidence-based fallback to sequential on ambiguous cases
3. **Quality:** Better handling of spatial ambiguity and low-confidence predictions

### Hypothesis

A hybrid inference detector with confidence monitoring can:
- Match or exceed thin-head quality (AP50:95 = 0.3063)
- Achieve 2-3× net speedup through mostly-parallel inference
- Show interpretable per-class fallback patterns

### Architecture

**Phase 1: Parallel Decoder Baseline**
- Single forward pass for all 35 classes
- 2.2M parameters (vs thin-head 6.8M)
- Training: 3 epochs, batch 4, lr 1e-4
- Loss function: 10×focal + 1×L1(size) + 1×L1(offset)

**Phase 2: Hybrid Inference Wrapper**
- Wrap Phase 1 with `HybridInferenceDetector`
- Fallback triggers:
  - Low confidence: `max(sigmoid(heatmap)) < 0.7`
  - Spatial ambiguity: multiple peaks within 8px
- Training: 5 epochs, batch 4, lr 5e-5
- Collect statistics: `parallel_count` vs `fallback_count`

**Phase 3-4: Evaluation & Calibration**
- val_select: Measure fallback rate, compute AP
- val_calibration: Tune confidence/ambiguity thresholds
- Compare with thin-head baseline (2 seeds)

### Implementation

**Files created:**
- `train_hybrid_phase1.py` — Parallel decoder training
- `train_hybrid_phase2.py` — Hybrid wrapper with fallback logic
- `evaluate_hybrid.py` — Evaluation with fallback statistics
- `analyze_hybrid.py` — Comprehensive analysis vs baseline
- `summarize_hybrid.py` — Summary report generator
- `auto_pipeline.sh` — Automatic Phase 2-4 orchestration
- `monitor_phase1.sh` — Training progress monitor
- `docs/hybrid_strategy.md` — Strategy document
- `README_HYBRID.md` — Experiment README

**Model design:**
```python
class HybridInferenceDetector(nn.Module):
    def __init__(self, base_model, confidence_threshold=0.7, ambiguity_threshold=8.0):
        # Wrap base parallel decoder
        # Monitor confidence: max(sigmoid(heatmap))
        # Detect spatial ambiguity: min distance between peaks
        # Fallback if: low_conf OR spatial_ambiguous
```

### Progress

**Phase 1 (in progress):**
- Started: 2026-09-05 ~10:00
- Status: Batch 1600/3799 (42%) of epoch 1/3
- Loss: Stable at 20-25 (down from 3391 initial)
- Speed: 0.16s/batch
- ETA: ~15 minutes remaining

**Phase 2-4 (queued):**
- Auto-pipeline started in background
- Will trigger automatically when Phase 1 completes
- Expected total time: ~2 hours from Phase 1 start

### Baseline Comparison

**Thin-head (dual_pathway_v2_thin):**
| Seed | AP50 | AP50:95 | val_select |
|---|---:|---:|---|
| 1337 | 0.4950 | 0.3091 | ✓ |
| 2024 | 0.4882 | 0.3035 | ✓ |
| **Mean** | **0.4916** | **0.3063** | |

**Hybrid inference (target):**
- Conservative: AP50:95 ≥ 0.3063 (parity)
- Aggressive: AP50:95 ≥ 0.31 (+1 point)
- Speed: 2-3× faster net (accounting for 10-20% fallback)

### Success Criteria

**Go/no-go:**
- [ ] Phase 1 converges (loss < 50) ← In progress
- [ ] Phase 2 stable (no NaN)
- [ ] Fallback triggers correctly (10-30% rate)
- [ ] val_select evaluation completes

**Quality gates:**
- [ ] AP50:95 ≥ 0.3063 (thin-head parity)
- [ ] Fallback rate 10-20%
- [ ] Per-class patterns interpretable

**Publication-ready:**
- [ ] AP50:95 ≥ 0.31 (+1 point over baseline)
- [ ] 2-seed replication
- [ ] Ablation: parallel-only vs hybrid vs sequential

### Expected Outcomes

**Best case:**
- AP50:95 = 0.31-0.32 (+1-2 points)
- Fallback rate 15%
- 3× net speedup
- Clear per-class fallback correlation (rare/ambiguous classes fallback more)

**Conservative case:**
- AP50:95 = 0.30-0.31 (parity)
- Fallback rate 20%
- 2.5× net speedup
- Speed contribution confirmed

**Failure case:**
- AP50:95 < 0.30
- Fallback rate > 50% (defeats purpose)
- Root cause analysis needed

### Protocol Compliance

✓ v2 manifest (203f63588e906637...)  
✓ Image-level splits before query expansion  
✓ train_new (15,196 samples) for training  
✓ val_select (1,016 images) for checkpoint selection  
✓ val_calibration for threshold tuning  
✓ test_set untouched (final report only)  
✓ Manifest fingerprint tracking in all outputs  

### Timeline

| Phase | Duration | Status | ETA |
|---|---|---|---|
| Phase 1 | 30 min | 🔄 42% | ~15 min |
| Phase 2 | 1 hour | ⏸️ Queued | After Phase 1 |
| Phase 3 | 20 min | ⏸️ Queued | After Phase 2 |
| Phase 4 | 10 min | ⏸️ Queued | After Phase 3 |
| **Total** | **~2 hours** | | **~11:45** |

### Next Actions

1. **Monitor Phase 1** via `tail -f phase1_full.log`
2. **Wait for auto-pipeline** to complete Phase 2-4
3. **Review summary** via `python summarize_hybrid.py`
4. **Decision based on results:**
   - If AP50:95 ≥ 0.31: Replicate with seed 2024, document as improvement
   - If 0.30 ≤ AP50:95 < 0.31: Speed-only contribution, still valuable
   - If AP50:95 < 0.30: Analyze failure, iterate on decoder architecture

### References

- **LocateAnything paper:** Resource/LocateAnything.pdf (parallel box decoding)
- **Teacher data:** data/Teacher/teacher_distill.jsonl (15,663 reasoning samples)
- **Baseline:** checkpoints/dual_pathway_v2_thin_seed{1337,2024}
- **Protocol:** docs/research_protocol.md
- **Strategy:** docs/hybrid_strategy.md
- **This log:** docs/experiment_log.md

### Notes

- Parallel decoder uses SimpleBackbone (3 conv layers) instead of full thin-head
- This is intentional: test hybrid logic first, then scale to full architecture
- If successful, can upgrade to thin-head backbone for full quality+speed
- Auto-pipeline will run Phase 2-4 unattended (~2 hours total)

**Status update frequency:** Will log when Phase 1 completes and final results available.
