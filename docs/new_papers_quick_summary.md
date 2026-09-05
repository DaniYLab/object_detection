# Quick Summary: 4 New Papers

**Date:** 2026-09-05  
**Context:** Preparing Phase 2 after dual-stream training completes

---

## 1. VGent: Visual Grounding via Modular Design

**Relevance:** ⭐⭐⭐⭐⭐ VERY HIGH - Directly applicable to teacher reasoning

**Core Idea:**
- Modular encoder-decoder: Frozen MLLM (reasoning) + Detector (proposals) + Cross-attention selector
- Disentangles high-level reasoning from low-level prediction
- No autoregressive decoding → fast inference, no hallucination

**Key Features:**
- QuadThinker: RL training for multi-target reasoning
- Mask-aware label: Detection-segmentation alignment
- Global target recognition: Augment proposals with text-image matching

**Application to FloorPlanCAD:**
```
Reasoning Encoder (Teacher, frozen) → spatial reasoning features
     ↓
Detector (Thin-head) → box proposals
     ↓
Cross-attention Selector → confidence scores → final predictions
```

**Expected Impact:**
- Quality: +1-2 AP50:95 (better false positive rejection)
- Speed: 2-3× for all-class inference (multi-target query)
- Rare classes: +1-2 AP on elevator, escalator, ramp

**Implementation Priority:** HIGH (1-2 weeks)

---

## 2. PerceptionDLM: Parallel Region Perception with Diffusion

**Relevance:** ⭐⭐⭐⭐ HIGH - Parallel inference alternative

**Core Idea:**
- Multimodal diffusion language model (not autoregressive)
- Parallel region captioning: generate multiple region descriptions simultaneously
- Structured attention masking for efficient multi-region perception

**Key Features:**
- Parallel decoding at both sequence and token levels
- ParaDLC-Bench: new benchmark for multi-region captioning
- Significantly faster than sequential processing

**Application to FloorPlanCAD:**
```
Input: Image + multiple region masks (one per class)
     ↓
PerceptionDLM-style decoder
     ↓
Parallel output: all 35 class predictions in one forward pass
```

**Expected Impact:**
- Speed: 5-10× faster than current sequential class queries
- Quality: Potentially better context-aware predictions
- Architecture: Major rewrite (diffusion decoder vs CenterNet)

**Implementation Priority:** MEDIUM (exploration phase, 2-3 weeks)

---

## 3. Detect Anything via Next Point Prediction

**Already read in previous session** - autoregressive coordinate prediction

**Relevance:** ⭐⭐⭐ MEDIUM

**Core Idea:**
- Predict box coordinates autoregressively: (x1, y1, x2, y2)
- Next-point prediction with vocabulary discretization
- Simpler than LocateAnything's parallel decoding

**Application:** Already considered, deprioritized due to speed concerns

---

## 4. Looking Locally: Object-Centric Vision Transformers

**Relevance:** ⭐⭐ LOW-MEDIUM - Foundation model alternative

**Need to read:** Title suggests object-centric learning, potentially useful for:
- Better feature extraction per object
- Slot attention mechanism
- Foundation model pretraining strategy

**Priority:** LOW (can skip if time-constrained)

---

## Priority Ranking for Phase 2

### Tier 1: Immediate Implementation (1-2 weeks)
1. **VGent-style Reasoning Selector** ← START HERE
   - Use existing teacher reasoning data (15,663 samples)
   - Freeze thin-head detector (validated baseline)
   - Train cross-attention selector only
   - Expected: +1-2 AP50:95, minimal risk

### Tier 2: Exploration (2-3 weeks)
2. **Multi-target Reasoning** (VGent QuadThinker-inspired)
   - Extend selector to multi-class queries
   - RL training for "find all objects" capability
   - Expected: 2-3× speed for all-class inference

3. **PerceptionDLM-style Parallel Decoder** (Research track)
   - Replace CenterNet decoder with diffusion-based parallel decoder
   - Major architecture change, higher risk
   - Expected: 5-10× speed, unknown quality impact

### Tier 3: Optional Enhancements (1 week each)
4. **Global Target Recognition** (VGent-inspired)
   - Augment proposals with CLIP-style image-text matching
   - Expected: +1-2 AP on rare classes

5. **Looking Locally** foundation model
   - Evaluate as alternative backbone
   - Pretraining strategy exploration

---

## Recommended Next Steps

**After dual-stream training completes:**

### Option A: Conservative (Recommended)
1. Evaluate dual-stream results
2. If AP50:95 ≥ 0.28: implement VGent reasoning selector on top
3. Stack improvements: dual-stream + reasoning selector

### Option B: Aggressive
1. Skip dual-stream if results underwhelming
2. Go directly to VGent selector on frozen thin-head
3. Faster iteration, proven baseline

### Option C: Exploration
1. Prototype PerceptionDLM parallel decoder
2. Compare against sequential approaches
3. Higher risk, potentially higher reward

**My recommendation: Option A** - validate dual-stream first, then add reasoning selector for maximum improvement with controlled risk.
