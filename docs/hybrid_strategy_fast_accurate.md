# Hybrid Strategy: Fast + Accurate Detection

**Date:** 2026-09-05  
**Goal:** Achieve 15-20× speedup + 2-5 AP50:95 improvement over sequential baseline

## 🎯 Core Insight

**90% predictions are confident → use fast parallel decoding**  
**10% predictions are ambiguous → use reasoning refinement**

This selective approach combines the speed of parallel decoding with the accuracy of reasoning-guided refinement, optimizing the speed/quality trade-off.

## 📊 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  Hybrid Parallel Detector                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: Image [B, 3, 512, 512]                              │
│     ↓                                                        │
│  Backbone Encoder (ONCE)                                     │
│     ↓                                                        │
│  Features [B, 256, 64, 64]                                   │
│     ↓                                                        │
│  ┌──────────────────────────────────────────────┐           │
│  │     Parallel Decoder (ALL 35 classes)        │           │
│  │   Output: [B, 35, 5, 64, 64]                 │           │
│  │   (center, w, h, offset_x, offset_y)         │           │
│  └──────────────────────────────────────────────┘           │
│     ↓                                                        │
│  Confidence Gating                                           │
│     ↓                                                        │
│  ┌──────────────────────────────────────────────┐           │
│  │  If confidence < 0.7: Reasoning Refinement   │           │
│  │  - Cross-attention with teacher embeddings   │           │
│  │  - Spatial relationship guidance             │           │
│  │  - Only for ambiguous predictions (~10%)     │           │
│  └──────────────────────────────────────────────┘           │
│     ↓                                                        │
│  Final Predictions                                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Two-Phase Training

### Phase 1: Parallel Baseline (Fast)

**Goal:** Train parallel decoder without reasoning  
**Duration:** ~3-4 hours (10 epochs)  
**Expected:** Baseline AP50:95, measure fast inference speed

**Command:**
```bash
python train_hybrid_phase1.py --epochs 10 --batch_size 8 --lr 1e-4
```

**Key features:**
- Single forward pass encodes image once
- Parallel head predicts all 35 classes simultaneously
- No per-class iteration overhead
- Expected latency: ~10-15ms per image (vs 203ms sequential)

### Phase 2: Selective Reasoning (Accurate)

**Goal:** Add reasoning refinement for ambiguous predictions  
**Duration:** ~3-4 hours (10 epochs)  
**Expected:** +1-2 AP50:95 improvement, minimal latency overhead

**Command:**
```bash
python train_hybrid_phase2.py \
    --epochs 10 \
    --batch_size 8 \
    --lr 1e-4 \
    --phase1_weights checkpoints/hybrid_parallel_phase1/best_val.pt
```

**Key features:**
- Loads Phase 1 weights as initialization
- Adds reasoning selector module
- Uses teacher embeddings (256-dim) for ambiguous cases
- Confidence threshold: 0.7 (tunable)
- Spatial ambiguity detection: 8px threshold

## 📈 Expected Results

| Metric | Sequential (old) | Phase 1 (parallel) | Phase 2 (+ reasoning) | Target |
|---|---:|---:|---:|---:|
| **Latency** | 203ms | ~12ms | ~15ms | **17× faster** |
| **AP50:95** | 0.306 | ~0.31-0.32 | ~0.33-0.35 | **+2.5-4.5** |
| **Throughput** | 4.9 img/s | ~83 img/s | ~67 img/s | **13-17×** |

### Phase Breakdown

**Phase 1 expectations:**
- Speed: 15-20× improvement (encoding once vs 35 times)
- Quality: Slight drop (~0.5-1.0 AP) due to parameter sharing
- Memory: Similar to sequential (same backbone)

**Phase 2 expectations:**
- Speed: Small overhead (~3-5ms) from selective reasoning
- Quality: Recover drop + gain ~1-2 AP from reasoning
- Net result: 17× speed + 2-5 AP improvement

## 🔬 Key Design Decisions

### 1. Why Parallel Decoding?

**Problem:** Sequential queries encode image 35 times  
**Solution:** Single encoding + parallel prediction head  
**Insight:** From LocateAnything paper - 2.5× speedup

### 2. Why Selective Reasoning?

**Problem:** Reasoning for all predictions is slow  
**Solution:** Only refine ambiguous predictions (~10%)  
**Insight:** Most predictions are confident, reasoning overhead wasted

### 3. Why Confidence Threshold = 0.7?

**From LocateAnything:**
- Predictions with confidence < 0.7 are unreliable
- Spatial ambiguity (nearby predictions) also triggers reasoning
- Hybrid mode: fast when confident, accurate when ambiguous

### 4. Why Teacher Embeddings?

**From teacher data analysis:**
- 15,663 samples with spatial reasoning patterns
- 256-dim embeddings encode scene understanding
- Frozen features (no retraining cost)
- Only used for ambiguous cases (selective overhead)

## 🧪 Validation Protocol

### Training
- **Split:** train_new from v2 manifest
- **Epochs:** 10 per phase (20 total)
- **Batch size:** 8
- **Optimizer:** AdamW, lr=1e-4
- **Loss:** 10×focal + 1×L1(size) + 1×L1(offset)

### Evaluation
- **Split:** val_select (1,016 images)
- **Threshold:** 0.05 (standard)
- **Top-k:** 100 per class
- **Metrics:** AP50, AP50:95, per-class AP
- **Latency:** Median over 100 runs (after 10 warmup)

### Comparison Gates
1. Phase 1 vs Sequential: Speed must be 10× faster minimum
2. Phase 2 vs Phase 1: Quality must improve by +0.5 AP50:95 minimum
3. Phase 2 vs Sequential: Combined 10× speed + quality gain

## 📂 File Structure

```
src/models/
├── hybrid_parallel_detector.py    # Main architecture
├── reasoning_selector.py          # Confidence gating logic
└── dual_stream_centernet.py       # Thin-head baseline (comparison)

train_hybrid_phase1.py             # Phase 1 training
train_hybrid_phase2.py             # Phase 2 training
evaluate_hybrid_detector.py        # Evaluation + latency profiling
run_hybrid_pipeline.py             # Full pipeline orchestrator

checkpoints/
├── hybrid_parallel_phase1/        # Phase 1 checkpoints
└── hybrid_parallel_phase2/        # Phase 2 checkpoints

outputs/
├── hybrid_phase1_evaluation.json  # Phase 1 results
├── hybrid_phase2_evaluation.json  # Phase 2 results
└── hybrid_pipeline_report.json    # Final comparison
```

## 🔄 Running the Pipeline

### Option 1: Automated (Recommended)

Run complete pipeline with one command:

```bash
python run_hybrid_pipeline.py \
    --epochs_phase1 10 \
    --epochs_phase2 10 \
    --batch_size 8 \
    --profile_latency
```

Total time: ~6-8 hours on GPU

### Option 2: Manual Steps

```bash
# Phase 1: Train parallel decoder
python train_hybrid_phase1.py --epochs 10

# Evaluate Phase 1
python evaluate_hybrid_detector.py \
    --checkpoint checkpoints/hybrid_parallel_phase1/best_val.pt \
    --phase phase1 \
    --profile_latency

# Phase 2: Add reasoning
python train_hybrid_phase2.py \
    --epochs 10 \
    --phase1_weights checkpoints/hybrid_parallel_phase1/best_val.pt

# Evaluate Phase 2
python evaluate_hybrid_detector.py \
    --checkpoint checkpoints/hybrid_parallel_phase2/best_val.pt \
    --phase phase2 \
    --profile_latency
```

## 🎯 Success Criteria

### Minimum Requirements (Go/No-Go)
- ✅ Phase 1 latency < 20ms (10× improvement)
- ✅ Phase 2 AP50:95 ≥ baseline + 0.5
- ✅ Combined: 10× speed + quality gain

### Stretch Goals
- 🌟 Phase 1 latency < 15ms (15× improvement)
- 🌟 Phase 2 AP50:95 ≥ baseline + 2.0
- 🌟 Combined: 15× speed + 2-3 AP gain

## 📊 Comparison with Alternatives

| Approach | Speed Gain | Quality Gain | Implementation Time |
|---|---:|---:|---:|
| Sequential baseline | 1× | baseline | — |
| Thin-head only | 1× | +2.8 AP50:95 | Complete ✅ |
| Parallel only | 15-20× | −0.5 to +0.5 | Phase 1 (3h) |
| **Hybrid (this)** | **15-20×** | **+2-5 AP** | **7 days** ⭐ |
| Full reasoning (all) | 0.5× | +3-5 AP | 2-3 weeks |
| Data engine | 1× | +3-5 AP | 1 month |

**Why Hybrid wins:**
- Best speed/quality trade-off
- Fastest to implement (7 days vs 1 month)
- Builds on proven components (parallel + teacher data)
- Incremental: Phase 1 delivers speed immediately

## 🔗 Related Work

### Papers Referenced
1. **LocateAnything** (arXiv:2501.xxxxx)
   - Parallel Box Decoding (PBD)
   - Hybrid inference mode (confidence fallback)
   - Insight: 2.5× speedup with no quality loss

2. **Teacher Data** (internal)
   - 15,663 reasoning samples
   - Spatial relationship patterns
   - 256-dim frozen embeddings

3. **Thin-head** (this repo)
   - Stride-4 hybrid head
   - +2.8 AP50:95 improvement (2 seeds)
   - Baseline for speed comparison

## 📝 Next Steps After Hybrid

If hybrid succeeds, further improvements:

1. **Multi-scale parallel decoding** (+0.5-1.0 AP)
   - Parallel at multiple resolutions
   - Better small object detection

2. **Curriculum learning** (+0.5-1.0 AP)
   - Order training by reasoning complexity
   - Faster convergence

3. **Test-time augmentation** (+0.5-1.0 AP)
   - Flip/rotate inference
   - Ensemble predictions

4. **Data engine** (+3-5 AP, long-term)
   - Synthetic generation
   - Cross-domain augmentation
   - Self-training

## ⚠️ Known Limitations

1. **Teacher data coverage:** Only 15,663 / 79,542 train images have reasoning
   - Mitigation: Use reasoning only when available
   - Fallback: Fast parallel for others

2. **Parallel capacity:** Shared decoder may underfit rare classes
   - Mitigation: Class-balanced sampling (existing)
   - Alternative: Hierarchical decoder (future work)

3. **Reasoning overhead:** 3-5ms per ambiguous prediction
   - Mitigation: Only 10% predictions are ambiguous
   - Average overhead: ~0.3-0.5ms per image

4. **Frozen teacher:** Cannot adapt to new patterns
   - Mitigation: Phase 2 learns when to trust/ignore teacher
   - Alternative: Fine-tune teacher (expensive)

## 📚 References

- `docs/sota_gap_plan.md` - Original gap analysis
- `docs/experiment_log.md` - Thin-head replication
- `data/Teacher/` - Teacher reasoning data
- `Resource/LocateAnything.pdf` - Parallel decoding paper
- `src/models/dual_stream_centernet.py` - Sequential baseline

---

**Status:** Implementation ready  
**Priority:** High (fastest path to speed + quality)  
**Risk:** Low (incremental, validated components)  
**Next:** Run `python run_hybrid_pipeline.py` to start training
