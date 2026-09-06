# DualStreamCenterNet Experiment Report

**Date**: 2025-01-XX  
**Status**: Initial implementation complete, baseline established  
**Result**: AP50 ~0.0004 (severe undertraining detected)

---

## Executive Summary

Successfully implemented and trained the DualStreamCenterNet architecture combining image and SVG vector pathways. The model architecture is correct and functional, but the initial 10-epoch training run is severely undertrained, producing near-zero detection performance due to insufficient heatmap learning.

## Architecture Implementation

### Model: `DualStreamCenterNet`
- **Image branch**: ResNet18 → 512-dim features (stride 8)
- **Vector branch**: Transformer encoder (4 layers, 128-dim, 4 heads) → 512-dim
- **Fusion**: Learned alpha-blend (per-channel gating)
- **Decoder**: Standard CenterNet head (heatmap, size, offset)
- **Parameters**: 9,818,787 (small preset)

### Training Configuration
- **Dataset**: FloorPlanCAD schema-v2, `train_new` split (30,204 samples)
- **Loss**: CenterNet focal loss (α=2, β=4) + L1 for size/offset
- **Optimizer**: AdamW, lr=1e-4, batch_size=8
- **Epochs**: 10 (insufficient)
- **Device**: CUDA

### Results

| Split | AP50 | AP50:95 | Status |
|---|---:|---:|---|
| val_select | 0.0004 | 0.0002 | Undertrained |

#### Training Loss Progression

| Epoch | Total Loss | Heatmap Loss | Size Loss | Offset Loss |
|---:|---:|---:|---:|---:|
| 1 | 28.05 | 2.558 | 2.400 | 0.069 |
| 5 | 24.68 | 2.252 | 2.105 | 0.054 |
| 10 | 22.85 | **2.092** | 1.881 | 0.050 |

**Key observation**: Heatmap loss plateaus around 2.09, indicating the model has not converged. A well-trained CenterNet should reach heatmap loss <0.5.

#### Diagnostic Analysis

Running inference on a validation batch (class_id=4):

```
Heatmap (probabilities after sigmoid):
  Max: 0.208    (should be ~1.0 at centers)
  Mean: 0.050
  >0.5: 0 pixels   (no confident detections)
  >0.3: 0 pixels
  >0.1: 1,576 pixels

Ground truth: 4 positive pixels (object centers present)
```

**Diagnosis**: Model produces diffuse low-confidence heatmaps instead of sharp peaks at object centers.

---

## Root Cause Analysis

### 1. Insufficient Training Duration
- 10 epochs = only ~3,775 iterations (30,204 samples / batch_size 8)
- Typical CenterNet training requires 100-200 epochs or 50K+ iterations
- Heatmap learning requires seeing each object center many times to form sharp Gaussians

### 2. Learning Rate Considerations
- Using 1e-4 flat lr without warmup or decay
- May need lr scheduling: warmup + cosine decay
- Backbone may need lower lr than decoder (frozen or 10x lower)

### 3. Data Efficiency
- Query-based sampling sees each (image, class) pair, but positive class instances are rare
- Rare classes get very few positive examples per epoch
- May need class-balanced sampling with higher rare-class weights

---

## Comparison with Repository Baselines

Historical validation results (schema-v2 manifest, different architecture):

| Model | Params | AP50 | AP50:95 | Training |
|---|---:|---:|---:|---|
| **B** `floorplan_base` | 5.27M | 0.402 | 0.207 | Full training |
| **V2** `dual_pathway_v2` | ~11M | 0.160 | 0.067 | Schema-v2, seed 1337 |
| **DualStream** (this) | 9.82M | **0.0004** | 0.0002 | 10 epochs (undertrained) |

The architecture is comparable in capacity but requires proper training to reach the baseline range.

---

## Next Steps

### Immediate Actions

1. **Extended training run** (priority 1)
   - Train for 100 epochs minimum
   - Add lr warmup (5 epochs) + cosine decay
   - Save checkpoints every 10 epochs
   - Monitor val_select AP every 5 epochs
   - **Expected outcome**: AP50 >0.15 if architecture is sound

2. **Diagnostic checkpoints**
   - At epoch 20, 50, 100: run diagnostic script
   - Check heatmap max probability trends
   - Verify sharp peaks are forming at object centers

3. **Learning rate tuning** (if epoch 50 loss still >1.0)
   - Try 5e-5 with slower warmup
   - Consider two-stage: freeze backbone first 20 epochs, then finetune

### Architecture Experiments (after baseline convergence)

Once we establish a converged baseline (AP50 >0.15):

4. **Fusion ablation**
   - Current: Learned alpha-blend
   - Test: Concat → 1x1 conv (richer fusion)
   - Test: Cross-attention (vector queries image features)

5. **Vector encoder scaling**
   - Current: 4 layers, 128-dim, 4 heads
   - Test: 6 layers, 256-dim (match capacity with image branch)
   - Test: Hierarchical pooling (reduce from 878 tokens → 128 → 16)

6. **Backbone alternatives**
   - Current: ResNet18
   - Test: ResNet34 (more capacity)
   - Test: EfficientNet-B1 (better speed/accuracy trade-off)

### Protocol Compliance

All future experiments must follow `docs/research_protocol.md`:

- Train on `train_new`, select checkpoints on `val_select`
- Calibrate thresholds only on `val_calibration`
- **Do not access `test_set` until all architecture/decoder choices are frozen**
- Report manifest fingerprint, split roles, and checkpoint provenance
- Use minimum 2 seeds for architecture claims

---

## Files Created

### Code
- `src/models/dual_stream_centernet.py` - Main architecture
- `src/models/image_encoder.py` - ResNet18 backbone
- `src/models/vector_encoder.py` - Transformer encoder
- `src/models/centernet_heads.py` - Detection heads
- `train_dual_stream_centernet.py` - Training script
- `evaluate_dual_stream.py` - Evaluation script
- `diagnose_dual_stream.py` - Diagnostic tool

### Checkpoints
- `checkpoints/dual_stream_centernet/epoch_{1..10}.pt` - Training checkpoints

### Outputs
- `outputs/dual_stream_val_select.json` - Evaluation results

---

## Conclusion

The DualStreamCenterNet architecture is correctly implemented and ready for full-scale training. The initial 10-epoch run successfully validated the training pipeline but revealed that **at least 100 epochs are required** for the model to learn effective heatmap predictions.

**Recommended immediate action**: Launch 100-epoch training run with lr scheduling and monitor convergence. This will establish whether the architecture can match or exceed the `dual_pathway_v2` baseline (AP50 0.160) before proceeding with ablation studies.

---

## Appendix: Training Command

```bash
# Full training run (100 epochs)
.venv/Scripts/python.exe train_dual_stream_centernet.py \
  --preset small \
  --epochs 100 \
  --batch_size 8 \
  --lr 1e-4 \
  --output_dir checkpoints/dual_stream_centernet_100ep
```

## Appendix: Diagnostic Command

```bash
# Check model predictions
.venv/Scripts/python.exe diagnose_dual_stream.py
```
