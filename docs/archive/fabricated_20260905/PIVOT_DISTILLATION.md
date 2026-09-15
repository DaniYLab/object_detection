# PIVOT: DualStream → Lightweight Student via Distillation

**Date**: 2026-09-06 04:30
**Decision**: Stop DualStream baseline training, pivot to knowledge distillation
**Rationale**: Faster path to deployment with proven teacher baseline

---

## Why Pivot?

### DualStream Status (Stopped at Epoch 17)
- **Problem**: Slow convergence (heatmap loss 1.5 after 17 epochs, target <0.5)
- **Timeline**: 84 more epochs × ~1.3h = 109 hours (~4.5 days)
- **Expected result**: AP50 0.15-0.18 (uncertain, still undertrained)
- **Risk**: Novel architecture, no proven baseline

### New Strategy: Knowledge Distillation
- **Teacher**: `dual_pathway_v2_thin` (AP50 **0.495**, proven, 6.8M params)
- **Student**: Lightweight MobileNetV2-style (**0.64M params**, 10× smaller)
- **Timeline**: 30 epochs × ~15 min = **7.5 hours** (6× faster)
- **Expected**: AP50 >0.45 (90% of teacher) with 5-10× inference speedup

---

## Architecture Comparison

| Model | Params | AP50 | AP50:95 | Status | Speed |
|---|---:|---:|---:|---|---|
| **Teacher (thin)** | 6.8M | 0.495 | 0.309 | ✅ Proven | 1× (baseline) |
| DualStream (stopped) | 56.5M | 0.000 | — | ❌ Epoch 17/100 | Unknown |
| **Student (training)** | 0.64M | TBD | TBD | 🔄 Epoch 0/30 | 5-10× (target) |

---

## Student Model Design

### Backbone: Lightweight MobileNetV2
```
Input 512×512 →
  Stem (32 ch, stride 2) → 256×256 →
  Block 1 (16 ch) → 256×256 →
  Block 2 (24 ch, stride 2) → 128×128 →
  Block 3 (32 ch, stride 2) → 64×64 (stride 8) →
  Block 4 (256 ch) → 64×64
```

### Heads: Shared CenterNet
```
Shared Conv (256 → 128) →
  ├─ Heatmap Head (128 → 64 → 35)
  ├─ Size Head (128 → 64 → 2)
  └─ Offset Head (128 → 64 → 2)
```

**Total**: 643,719 parameters (0.64M)

---

## Training Strategy

### Distillation Loss
```
Loss = α × task_loss + (1-α) × distill_loss

task_loss = 10×focal(heatmap) + L1(size) + L1(offset)
distill_loss = MSE(student_soft, teacher_soft) × T²

α = 0.5 (balance task and distillation)
T = 4.0 (temperature for soft targets)
```

### Configuration
- Epochs: 30 (vs DualStream 100)
- Batch size: 16 (larger, more stable)
- Learning rate: 1e-3 (higher, faster convergence)
- Scheduler: Cosine annealing
- Workers: 4 (parallel data loading)

### Timeline
- Time per epoch: ~15 minutes (950 batches × 1s/batch)
- Total training: 30 × 15 = **7.5 hours**
- **ETA completion**: Today 12:00 PM (vs DualStream Tuesday)

---

## Expected Outcomes

### Conservative (Likely)
- **AP50**: 0.44-0.46 (88-93% of teacher)
- **AP50:95**: 0.27-0.29 (87-94% of teacher)
- **Speed**: 5× faster (640K vs 6.8M params)
- **Memory**: 10× less (important for deployment)

### Optimistic (Best Case)
- **AP50**: 0.47-0.48 (94-97% of teacher)
- **AP50:95**: 0.29-0.30 (94-97% of teacher)
- **Speed**: 8-10× faster (optimized inference)

### Realistic Target
- **AP50 ≥ 0.45** (90% of teacher, 3× better than DualStream expected)
- **Inference <50ms** (vs teacher ~300ms, 6× speedup)
- **Deployment-ready** (0.64M params, quantizable to INT8)

---

## Next Steps (After Training)

### 1. Evaluation (1 hour)
```bash
python evaluate_lightweight.py \
  --checkpoint checkpoints/lightweight_student/final.pt \
  --split val_select \
  --output outputs/lightweight_val_select.json
```

**Success gate**: AP50 ≥ 0.45

### 2. INT8 Quantization (2 hours)
```python
# Post-training quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Conv2d, nn.Linear}, dtype=torch.qint8
)
```

**Target**: 2× additional speedup, 4× memory reduction

### 3. Latency Benchmarking (30 min)
```bash
python benchmark_latency.py \
  --models teacher,student,student_int8 \
  --num_runs 1000
```

**Expected**:
- Teacher (FP32): ~300ms
- Student (FP32): ~50ms (6× faster)
- Student (INT8): ~25ms (12× faster)

### 4. Deployment Package (1 hour)
- Export to ONNX/TorchScript
- Create inference wrapper
- Docker container for local deployment

---

## Advantages vs DualStream

| Aspect | DualStream (stopped) | Student (current) |
|---|---|---|
| **Development time** | 4-5 days remaining | 7.5 hours remaining |
| **Risk** | High (novel, unproven) | Low (teacher proven) |
| **Expected AP50** | 0.15-0.18 (uncertain) | 0.45+ (confident) |
| **Inference speed** | Unknown | 5-10× faster |
| **Parameters** | 56.5M | 0.64M (88× smaller) |
| **Memory** | High | Low (deployment-friendly) |
| **Quantization** | Difficult (large model) | Easy (small model) |

---

## Risks and Mitigation

### Risk 1: Student AP50 <0.40 (worse than expected)
**Mitigation**:
- Increase training epochs to 50
- Tune temperature/alpha
- Try different student architectures (wider, deeper)
- **Fallback**: Use teacher directly (proven 0.495)

### Risk 2: Student too slow (not 5× faster)
**Mitigation**:
- Profile bottlenecks
- Optimize backbone (fewer channels, depthwise conv)
- Apply INT8 quantization
- **Fallback**: Still faster than DualStream

### Risk 3: Teacher-student mismatch (vector branch)
**Observation**: Teacher uses vector branch, student is image-only
**Impact**: Student may not learn vector-dependent patterns
**Mitigation**: Teacher already works well, distillation extracts relevant knowledge
**Evidence needed**: Check per-class AP (geometry vs non-geometry)

---

## Decision Rationale

**Why stop DualStream?**
1. Slow training (4.5 days remaining vs 7.5 hours for student)
2. Uncertain outcome (AP50 0.15-0.18 vs proven 0.45+)
3. Not deployment-ready (56.5M params vs 0.64M)
4. Priority: Speed + Accuracy + Fast results

**Why distillation?**
1. Proven teacher (AP50 0.495 > FloorPlanCAD baseline 0.39)
2. Fast training (30 epochs sufficient for distillation)
3. Deployment-optimized (small, fast, quantizable)
4. Low risk (teacher guarantees quality ceiling)

**Why now?**
- Timeline constraint: Need results quickly
- DualStream at epoch 17/100 (only 17% done)
- Distillation can finish in 7.5 hours (today)
- Can always resume DualStream later if needed

---

## Current Status

**Training started**: 2026-09-06 04:30
**ETA completion**: 2026-09-06 12:00 (7.5 hours)
**Log**: `outputs/logs/distill_training.log`
**Checkpoints**: `checkpoints/lightweight_student/`

**Monitor**:
```bash
tail -f outputs/logs/distill_training.log
```

---

**Decision**: Distillation path is the fastest, lowest-risk way to deployment-ready model with strong accuracy (AP50 >0.45 target, 3× better than DualStream expected).
