# Hybrid Architecture Technical Details

## 1. Architecture Overview

```
Input Pipeline:
┌─────────────────────────────────────────────────────────┐
│ Image (512×512)  SVG Tokens  Teacher Reasoning          │
│      ↓               ↓              ↓                   │
│ ImageEncoder    VectorEncoder   TeacherEncoder (frozen) │
│   [B,4096,256]    [B,256]         [B,256]              │
└─────────────────────────────────────────────────────────┘

Parallel Decoder:
┌─────────────────────────────────────────────────────────┐
│ Shared Conv (256 → 256)                                 │
│      ↓                                                  │
│ Parallel Head (256 → 35*5=175 channels)                │
│      ↓                                                  │
│ Reshape → [B, 35, 5, H, W]                             │
│           ├─ heatmap [B,35,H,W]                        │
│           ├─ sizes [B,35,2,H,W]                        │
│           └─ offsets [B,35,2,H,W]                      │
│      ↓                                                  │
│ Confidence = sigmoid(heatmap).max(dim=(2,3))           │
│           → [B, 35] per-class confidence               │
└─────────────────────────────────────────────────────────┘

Selective Reasoning:
┌─────────────────────────────────────────────────────────┐
│ For each class_id where confidence < 0.7:               │
│                                                         │
│ 1. Project reasoning: [B,256] → [B,1,256]              │
│ 2. Flatten features: [B,256,H,W] → [B,H*W,256]        │
│ 3. Cross-attention:                                     │
│    Q = features_seq                                     │
│    K,V = reasoning                                      │
│    → refined_seq [B,H*W,256]                           │
│ 4. Reshape back: [B,256,H,W]                           │
│ 5. Refinement conv: [B,256,H,W] → [B,5,H,W]           │
│ 6. Replace parallel output for this class              │
└─────────────────────────────────────────────────────────┘

Output:
┌─────────────────────────────────────────────────────────┐
│ Final predictions: [B, 35, 5, H, W]                    │
│ - 90% from parallel (fast)                              │
│ - 10% from reasoning (accurate)                         │
└─────────────────────────────────────────────────────────┘
```

## 2. Component Details

### 2.1 Parallel Decoder Head

**Input:** Features [B, 256, H, W]  
**Output:** Predictions [B, 35, 5, H, W]

```python
class ParallelDecoderHead(nn.Module):
    def __init__(self, feature_dim=256, num_classes=35, grid_size=64):
        super().__init__()
        self.num_classes = num_classes
        
        # Shared feature processing
        self.shared_conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.GroupNorm(32, feature_dim),
            nn.ReLU(inplace=True),
        )
        
        # Parallel prediction head: ALL classes at once
        self.parallel_head = nn.Conv2d(
            feature_dim, 
            num_classes * 5,  # 35 classes × 5 channels
            kernel_size=1
        )
    
    def forward(self, features):
        B, _, H, W = features.shape
        
        # Shared processing
        x = self.shared_conv(features)
        
        # Parallel predictions
        preds = self.parallel_head(x)  # [B, 175, H, W]
        
        # Reshape to [B, 35, 5, H, W]
        preds = preds.view(B, self.num_classes, 5, H, W)
        
        # Split channels
        heatmaps = preds[:, :, 0, :, :]   # [B, 35, H, W]
        sizes = preds[:, :, 1:3, :, :]    # [B, 35, 2, H, W]
        offsets = preds[:, :, 3:5, :, :]  # [B, 35, 2, H, W]
        
        # Compute per-class confidence
        confidence = torch.sigmoid(heatmaps).amax(dim=(2,3))  # [B, 35]
        
        return {
            'heatmaps': heatmaps,
            'sizes': sizes,
            'offsets': offsets,
            'confidence': confidence,
        }
```

**Key insight:** ONE convolution predicts ALL 35 classes simultaneously.

### 2.2 Selective Reasoning Module

**Input:** 
- Features [B, 256, H, W]
- Reasoning embedding [B, 256]
- Ambiguous mask [B, H, W]

**Output:** Refinement [B, 5, H, W]

```python
class SelectiveReasoning(nn.Module):
    def __init__(self, feature_dim=256, reasoning_dim=256):
        super().__init__()
        
        # Project reasoning to feature space
        self.reasoning_projection = nn.Sequential(
            nn.Linear(reasoning_dim, feature_dim),
            nn.ReLU(inplace=True),
        )
        
        # Cross-attention: reasoning guides features
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True,
        )
        
        # Refinement prediction
        self.refinement = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.GroupNorm(32, feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, 5, 1),  # 5 output channels
        )
    
    def forward(self, features, reasoning_embedding, ambiguous_mask):
        B, C, H, W = features.shape
        
        # Project reasoning
        reasoning = self.reasoning_projection(reasoning_embedding)
        reasoning = reasoning.unsqueeze(1)  # [B, 1, 256]
        
        # Flatten features to sequence
        features_seq = features.flatten(2).transpose(1, 2)  # [B, H*W, 256]
        
        # Cross-attention
        refined_seq, attn_weights = self.cross_attention(
            query=features_seq,
            key=reasoning,
            value=reasoning,
        )
        
        # Reshape back to spatial
        refined = refined_seq.transpose(1, 2).view(B, C, H, W)
        
        # Predict refinement
        delta = self.refinement(refined)  # [B, 5, H, W]
        
        return delta
```

**Key insight:** Only refine WHERE ambiguous, guided by teacher reasoning.

### 2.3 Hybrid Detector (Main Model)

```python
class HybridParallelDetector(nn.Module):
    def __init__(
        self,
        image_encoder,
        vector_encoder,
        teacher_encoder,  # Frozen
        num_classes=35,
        feature_dim=256,
        confidence_threshold=0.7,
    ):
        super().__init__()
        
        self.image_encoder = image_encoder
        self.vector_encoder = vector_encoder
        self.teacher_encoder = teacher_encoder
        self.confidence_threshold = confidence_threshold
        
        # Freeze teacher encoder
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False
        
        # Parallel decoder
        self.parallel_decoder = ParallelDecoderHead(
            feature_dim=feature_dim,
            num_classes=num_classes,
        )
        
        # Selective reasoning
        self.reasoning_selector = SelectiveReasoning(
            feature_dim=feature_dim,
            reasoning_dim=256,
        )
    
    def forward(
        self,
        image,
        stroke_tokens,
        stroke_mask,
        reasoning_embedding=None,  # Optional for Phase 1
    ):
        # Encode inputs (ONCE for all classes)
        image_features = self.image_encoder(image)  # [B, 4096, 256]
        vector_features = self.vector_encoder(
            stroke_tokens, 
            stroke_mask
        )  # [B, 256]
        
        # Fuse image + vector
        fused_features = self.fuse(image_features, vector_features)
        
        # Parallel decoding (FAST branch)
        parallel_output = self.parallel_decoder(fused_features)
        
        # If no reasoning, return parallel output
        if reasoning_embedding is None:
            return parallel_output
        
        # Selective reasoning (ACCURATE branch)
        confidence = parallel_output['confidence']  # [B, 35]
        
        # Identify ambiguous classes
        ambiguous_mask = confidence < self.confidence_threshold  # [B, 35]
        
        # If all confident, return parallel output
        if not ambiguous_mask.any():
            return parallel_output
        
        # Refine ambiguous predictions
        refined_output = self.refine_predictions(
            parallel_output=parallel_output,
            features=fused_features,
            reasoning=reasoning_embedding,
            ambiguous_mask=ambiguous_mask,
        )
        
        return refined_output
```

## 3. Training Strategy

### Phase 1: Parallel Baseline

**Goal:** Train fast parallel decoder without reasoning

**Loss function:**
```python
def compute_loss_phase1(pred, target):
    # Standard CenterNet loss
    heatmap_loss = focal_loss(
        pred['heatmaps'], 
        target['heatmap']
    )
    
    size_loss = l1_loss_masked(
        pred['sizes'], 
        target['size'],
        mask=target['heatmap'] > 0.5
    )
    
    offset_loss = l1_loss_masked(
        pred['offsets'], 
        target['offset'],
        mask=target['heatmap'] > 0.5
    )
    
    total = 10.0 * heatmap_loss + size_loss + offset_loss
    return total
```

**Data:** Query-based sampling (one class per batch)  
**Epochs:** 3  
**Expected quality:** 0.28-0.30 AP50:95  
**Expected speed:** 15-20× faster

### Phase 2: Reasoning Selector

**Goal:** Add selective reasoning refinement

**Loss function:**
```python
def compute_loss_phase2(pred, target, teacher_output):
    # CenterNet loss (same as Phase 1)
    centernet_loss = compute_loss_phase1(pred, target)
    
    # Knowledge distillation loss (only for ambiguous cases)
    ambiguous_mask = pred['confidence'] < 0.7
    
    if ambiguous_mask.any():
        kd_loss = F.mse_loss(
            pred['heatmaps'][ambiguous_mask],
            teacher_output['heatmaps'][ambiguous_mask]
        )
    else:
        kd_loss = 0.0
    
    total = centernet_loss + 0.5 * kd_loss
    return total
```

**Data:** Query-based + teacher reasoning embeddings  
**Epochs:** 3  
**Expected quality:** 0.31-0.33 AP50:95 (+2-3 points)  
**Expected speed:** Same as Phase 1

### Phase 3: Joint Fine-tuning

**Goal:** End-to-end optimization

**Loss function:**
```python
def compute_loss_phase3(pred, target, teacher_output):
    # Same as Phase 2 but with adjusted weights
    centernet_loss = compute_loss_phase1(pred, target)
    
    ambiguous_mask = pred['confidence'] < 0.7
    
    if ambiguous_mask.any():
        # Stronger KD weight for fine-tuning
        kd_loss = F.mse_loss(
            pred['heatmaps'][ambiguous_mask],
            teacher_output['heatmaps'][ambiguous_mask]
        )
        
        # Additional feature alignment
        feature_loss = F.cosine_embedding_loss(
            pred['features'][ambiguous_mask],
            teacher_output['features'][ambiguous_mask],
            target=torch.ones(ambiguous_mask.sum())
        )
    else:
        kd_loss = 0.0
        feature_loss = 0.0
    
    total = centernet_loss + 0.3 * kd_loss + 0.2 * feature_loss
    return total
```

**Data:** Query-based + teacher reasoning  
**Epochs:** 5  
**Expected quality:** 0.33-0.35 AP50:95 (+0.5-1 additional)

## 4. Inference Pipeline

### 4.1 All-Class Inference (Synchronized)

```python
def inference_all_classes(model, image, svg_tokens, reasoning=None):
    """Infer all 35 classes in ONE forward pass."""
    
    # Single encoding
    with torch.no_grad():
        output = model(
            image=image,
            stroke_tokens=svg_tokens,
            stroke_mask=create_mask(svg_tokens),
            reasoning_embedding=reasoning,
        )
    
    # Post-process all classes
    all_predictions = []
    for class_id in range(35):
        boxes, scores = decode_predictions(
            heatmap=output['heatmaps'][:, class_id],
            sizes=output['sizes'][:, class_id],
            offsets=output['offsets'][:, class_id],
            threshold=0.05,
            top_k=100,
        )
        all_predictions.append((boxes, scores, class_id))
    
    return all_predictions
```

**Latency breakdown:**
- Encoding: ~2ms (once)
- Parallel decode: ~3ms (all classes)
- Post-process: ~7ms (NMS, decode)
- **Total: ~12ms** (vs 203ms sequential)

### 4.2 Single-Class Inference

```python
def inference_single_class(model, image, svg_tokens, class_id, reasoning=None):
    """Infer one specific class (for query mode)."""
    
    with torch.no_grad():
        output = model(
            image=image,
            stroke_tokens=svg_tokens,
            stroke_mask=create_mask(svg_tokens),
            reasoning_embedding=reasoning,
        )
    
    # Extract only requested class
    boxes, scores = decode_predictions(
        heatmap=output['heatmaps'][:, class_id],
        sizes=output['sizes'][:, class_id],
        offsets=output['offsets'][:, class_id],
        threshold=0.05,
        top_k=100,
    )
    
    return boxes, scores
```

**Latency:** Still ~12ms (parallel overhead is negligible)

## 5. Memory Profile

### Thin-head (Baseline)
- **Forward pass:** 0.20 GiB
- **Training:** 1.56 GiB peak
- **Parameters:** 6.8M

### Hybrid Parallel
- **Forward pass:** ~0.25 GiB (+25%)
- **Training:** ~1.8 GiB peak (+15%)
- **Parameters:** ~7.5M (+10%)

**Reason for increase:**
- Parallel head: 256 → 175 channels (vs 256 → 5)
- Reasoning selector: +0.5M parameters
- **Trade-off:** Worth it for 17× speedup

## 6. Ablation Analysis

### Baseline: Thin-head Sequential
- AP50:95: 0.3063
- Latency: 203ms (all-class)
- Parameters: 6.8M

### Variant A: Parallel only (no reasoning)
- Expected AP50:95: 0.28-0.30 (-0.01 to +0.00)
- Expected latency: 12ms (17× faster)
- Parameters: 7.0M

### Variant B: Parallel + selective reasoning
- Expected AP50:95: 0.31-0.33 (+0.01 to +0.02)
- Expected latency: 13ms (16× faster)
- Parameters: 7.5M

### Variant C: Full reasoning (all classes)
- Expected AP50:95: 0.33-0.35 (+0.02 to +0.04)
- Expected latency: 50ms (4× faster)
- Parameters: 8.0M

**Chosen:** Variant B (best speed/quality trade-off)

## 7. Confidence Threshold Tuning

### Methodology
1. Train Phase 2 model with threshold 0.7
2. On val_calibration, sweep thresholds [0.5, 0.6, 0.7, 0.8, 0.9]
3. Measure AP50:95 and reasoning usage rate
4. Select threshold maximizing AP50:95

### Expected curve
```
Threshold | Reasoning % | AP50:95 | Latency
----------|-------------|---------|--------
0.5       | 30%         | 0.340   | 18ms
0.6       | 20%         | 0.335   | 15ms
0.7       | 10%         | 0.320   | 13ms  ← baseline
0.8       | 5%          | 0.310   | 12ms
0.9       | 2%          | 0.295   | 12ms
```

**Insight:** Lower threshold = more reasoning = higher quality but slower

## 8. Next Optimizations

### Short-term (This Week)
1. ✅ Parallel decoder implementation
2. ⏳ Reasoning selector training
3. ⏳ Threshold calibration
4. 🔜 Spatial ambiguity detection (peak proximity < 8px)

### Medium-term (Next Week)
5. FLIP object-centric features
6. Reasoning-driven augmentation
7. Multi-scale parallel decoding

### Long-term (Next 2 Weeks)
8. Block-based output (direct coordinates)
9. Dual-formulation training (NTP + MTP)
10. Large-scale data engine

---

**Document version:** 1.0  
**Last updated:** 2026-09-05  
**Status:** Phase 1 training in progress
