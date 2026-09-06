# VGent: Visual Grounding via Modular Design

**Paper:** VGent -Visual Grounding via Modular Design for Disentangling Reasoning and Prediction  
**Date Analyzed:** 2026-09-05  
**Relevance:** HIGH - Directly applicable to teacher reasoning integration

---

## Core Architecture

### Modular Encoder-Decoder Design

```
┌─────────────────────────────────────────────┐
│              VGent Architecture              │
├─────────────────────────────────────────────┤
│                                             │
│  Encoder (Frozen MLLM)                      │
│  ├─ Input: Image + Text query               │
│  ├─ Output: Hidden states (reasoning)       │
│  └─ Advantage: Untouched reasoning ability  │
│                                             │
│  Detector (Off-the-shelf)                   │
│  ├─ Input: Image only                       │
│  ├─ Output: High-recall box proposals       │
│  └─ Advantage: Efficient localization       │
│                                             │
│  Decoder (Cross-attention Selector)         │
│  ├─ Query: Detector proposals               │
│  ├─ Key/Value: Encoder hidden states        │
│  ├─ Output: Selected target box(es)         │
│  └─ Advantage: Fast, no autoregressive      │
│                                             │
└─────────────────────────────────────────────┘
```

### Key Insights

1. **Disentanglement**: Separate high-level reasoning (MLLM) from low-level prediction (decoder)
2. **Frozen encoder**: Preserve pretrained reasoning, avoid catastrophic forgetting
3. **Cross-attention selection**: Decoder selects from proposals using reasoning signals
4. **No autoregressive decoding**: Fast inference, no hallucination loops

---

## Comparison with Our Approach

### What We Have (Current)

```
FloorPlanCAD Detection Pipeline:
├─ Image encoder (ViT-based)
├─ Vector encoder (stroke transformer)
├─ Late fusion (concat + projection)
└─ CenterNet decoder (heatmap-based)
```

### What We Can Build (VGent-inspired)

```
VGent-style FloorPlanCAD:
├─ Reasoning encoder (Teacher model, frozen)
│  └─ Input: Image + "Find all doors/windows"
│  └─ Output: Spatial reasoning hidden states
│
├─ Detector (Thin-head CenterNet or Dual-Stream)
│  └─ Output: High-recall proposals for queried class
│
└─ Selector decoder (Cross-attention)
   └─ Query: Detector proposals
   └─ Key/Value: Reasoning hidden states
   └─ Output: Confidence scores → final predictions
```

---

## Application to Teacher Reasoning Data

### Current Teacher Data Structure

```json
{
  "image_id": "...",
  "reasoning": "Looking at the image... beds at [x1,y1,x2,y2]...",
  "result": {"beds": [[x1,y1,x2,y2], ...]},
  "model": "deepseek-v4-flash-vision-exp"
}
```

### VGent-inspired Usage Strategy

**Option 1: Reasoning-guided Selector** (Most direct application)

```python
# Training
proposals = detector(image, class_id)  # e.g., 100 box proposals
reasoning_features = teacher_encoder(image, text_query)  # Frozen
scores = selector_decoder(proposals, reasoning_features)  # Trainable
loss = cross_entropy(scores, ground_truth_match)

# Inference
proposals = detector(image, class_id)
reasoning_features = teacher_encoder(image, text_query)  # Fast if frozen
final_boxes = proposals[scores > threshold]
```

**Benefits:**
- ✅ Leverages teacher's spatial reasoning
- ✅ No need to retrain detector
- ✅ Modular: can upgrade detector or teacher independently
- ✅ Fast inference: one forward pass + selection

**Option 2: Reasoning as Attention Bias**

Extract spatial hints from teacher reasoning:
- "top section: windows" → attend more to upper region
- "left side: door" → attend more to left region
- "adjacent to wall" → spatial relationship prior

**Option 3: Reasoning-driven Hard Negative Mining**

Teacher confusion patterns:
- "Could be sofa or chair" → mine sofa/chair pairs
- "Ambiguous between bed and desk" → create hard negatives

---

## Three Enhancements from VGent Paper

### 1. QuadThinker (RL-based Multi-target Reasoning)

**Problem:** Single-target MLLMs struggle with "find all X" queries  
**Solution:** RL training to enhance multi-target recognition

**Application to FloorPlanCAD:**
- Current: Query-based (one class at a time)
- VGent-inspired: "Find all doors, windows, and stairs" in one query
- Benefit: Faster all-class inference if we build multi-class selector

### 2. Mask-aware Label

**Problem:** Detection-segmentation ambiguity (box vs mask)  
**Solution:** Explicit mask-aware supervision

**Application to FloorPlanCAD:**
- Current: Only box supervision
- Could add: Segmentation masks for stuff classes (wall, parking)
- Benefit: Better alignment with source SOTA table (segmentation F1)

### 3. Global Target Recognition

**Problem:** Missing some targets due to low detector recall  
**Solution:** Augment proposals with global recognition module

**Application to FloorPlanCAD:**
- Current: Pure detector proposals
- Could add: Text-image matching to identify missed objects
- Benefit: Better recall on rare classes (elevator, escalator)

---

## Implementation Roadmap

### Phase 1: Reasoning-guided Selector (1 week)

**Goal:** Add selector decoder on top of thin-head detector

**Steps:**
1. Freeze thin-head detector (AP50:95 = 0.3063 baseline)
2. Extract teacher reasoning features (15,663 samples)
3. Build cross-attention selector:
   ```python
   class ReasoningSelector(nn.Module):
       def __init__(self, dim=256):
           self.cross_attn = nn.MultiheadAttention(dim, num_heads=8)
           self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
       
       def forward(self, proposals, reasoning_features):
           # proposals: [B, N_proposals, dim]
           # reasoning_features: [B, L, dim]
           attended = self.cross_attn(proposals, reasoning_features, reasoning_features)
           scores = self.mlp(attended).squeeze(-1)  # [B, N_proposals]
           return scores
   ```
4. Train selector only (detector frozen)
5. Evaluate on val_select

**Expected Gain:** +1-2 AP50:95 (better false positive rejection)

### Phase 2: Multi-class Reasoning (2 weeks)

**Goal:** Extend to multi-target queries ("find all objects")

**Steps:**
1. Train QuadThinker-style multi-target recognition
2. Build class-agnostic proposal generator
3. Selector predicts both: (1) which proposals, (2) which classes
4. Compare latency vs current per-class inference

**Expected Gain:** 2-3× speed for all-class inference

### Phase 3: Global Recognition Module (1 week)

**Goal:** Augment proposals with text-image matching

**Steps:**
1. Add CLIP-style image-text encoder
2. For each class: compute image-text similarity
3. Generate additional proposals in high-similarity regions
4. Feed augmented proposals to selector

**Expected Gain:** +1-2 AP on rare classes (elevator, escalator, ramp)

---

## Success Criteria

### Baseline (Current Thin-head)
- AP50: 0.4916
- AP50:95: 0.3063
- Latency: 5.8ms/image (all-class synchronized)

### Target (VGent-inspired)
- AP50:95: **≥ 0.32** (+1.5 points, reasoning-guided selection)
- Rare-class AP: **≥ 0.15** (elevator, escalator, ramp)
- All-class latency: **≤ 3ms/image** (multi-target query)

### Validation Protocol
1. Train selector on train_new
2. Select checkpoint on val_select
3. Calibrate threshold on val_calibration
4. Replicate with seed 2024
5. Final report on test_set (only if ≥2 seeds confirm improvement)

---

## Key Takeaways

1. **Modularity wins**: Frozen reasoning + trainable selector preserves both strengths
2. **Cross-attention is the bridge**: Proposals ← reasoning features
3. **No autoregressive**: Fast, stable, no hallucination
4. **Multi-target reasoning**: RL training for "find all X" queries
5. **Augmented proposals**: Global recognition supplements detector

**Next Action:** Wait for dual-stream training to complete, then decide:
- **Option A**: Pure dual-stream (image + vector)
- **Option B**: VGent-style selector on top of dual-stream
- **Option C**: Hybrid (dual-stream + reasoning selector)
