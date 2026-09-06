# Phase 2 Research Roadmap: Multi-Paper Integration

**Date:** 2026-09-05  
**Context:** After thin-head validation (AP50:95 = 0.3063, 2 seeds)  
**Goal:** Reach detection AP comparable to source benchmarks while improving speed

---

## Current Baseline

### Thin-Head Architecture (Validated, 2 seeds)
- **Preset:** dual_pathway_v2_thin
- **AP50:** 0.4916 (mean of 2 seeds)
- **AP50:95:** 0.3063 (mean of 2 seeds)
- **Parameters:** 6.8M
- **Latency:** 5.8ms/image (all-class synchronized)
- **Memory:** 0.20 GiB peak

### Gap Analysis
- **Detection baselines** (FloorPlanCAD paper):
  - Faster R-CNN R101: AP50 = 0.602
  - YOLOv3: AP50 = 0.639
  - DINO: AP50 = 0.649
- **Current gap:** −0.15 to −0.20 AP50
- **Rare class issue:** elevator (0.023), escalator (0.296), ramp (0.000)

---

## Research Paper Synthesis

### Five Papers Analyzed

1. **LocateAnything** (already analyzed in Phase 1)
2. **VGent** (Visual Grounding via Modular Design)
3. **PerceptionDLM** (Parallel Region Perception)
4. **Detect Anything** (Next Point Prediction)
5. **FLIP** (Looking Locally, Object-Centric ViT)

---

## Key Insights Matrix

| Paper | Core Innovation | Speed Impact | Quality Impact | Implementation Cost |
|---|---|---|---|---|
| VGent | Frozen reasoning + selector | 2-3× (multi-target) | +1-2 AP50:95 | 1-2 weeks |
| PerceptionDLM | Parallel diffusion decoder | 5-10× | Unknown | 2-3 weeks |
| LocateAnything | Parallel box decoding | 3-5× | +1-2 AP50:95 | 5 days |
| Detect Anything | Autoregressive coords | Slower | +0-1 AP50:95 | 3-4 days |
| FLIP | Fovea-like attention | 2× vs SAM2 | +3-5 IoU | 2-3 weeks |

---

## Three Strategic Tracks

### Track 1: Reasoning Integration (VGent-inspired)

**Motivation:** Leverage 15,663 teacher reasoning samples

**Architecture:**
```
Teacher Encoder (frozen deepseek-v4-flash)
    ↓ spatial reasoning features (256-dim)
Detector (thin-head, frozen)
    ↓ box proposals (top-100 per class)
Cross-attention Selector (trainable)
    ↓ confidence scores
Final Predictions (threshold + NMS)
```

**Implementation Phases:**

**Phase 1.1: Single-class Reasoning Selector (1 week)**
- Freeze thin-head detector
- Extract teacher reasoning features for all 15,663 samples
- Build cross-attention selector:
  ```python
  class ReasoningSelector(nn.Module):
      def forward(self, proposals, reasoning_features):
          # proposals: [B, N, 256]
          # reasoning_features: [B, L, 256]
          attended = cross_attn(proposals, reasoning_features)
          scores = mlp(attended)  # [B, N, 1]
          return scores
  ```
- Training: selector only, detector frozen
- Loss: BCE on proposal-GT matching
- Expected: +1-2 AP50:95 (better FP rejection)

**Phase 1.2: Multi-target Reasoning (1 week)**
- Extend to multi-class queries: "Find all doors, windows, stairs"
- RL training (QuadThinker-inspired) for joint reasoning
- Selector predicts: (box_id, class_id, confidence)
- Expected: 2-3× speed for all-class inference

**Phase 1.3: Global Recognition (1 week)**
- Add CLIP-style image-text matching
- Augment proposals in high-similarity regions
- Especially for rare classes (elevator, escalator, ramp)
- Expected: +1-2 AP on rare classes

**Total Track 1:** 3 weeks, +2-3 AP50:95, 2-3× speed

---

### Track 2: Parallel Decoding (LocateAnything + PerceptionDLM)

**Motivation:** Replace sequential class queries with parallel prediction

**Two Approaches:**

#### Approach 2A: Parallel Box Decoding (LocateAnything)

**Architecture:**
```
Image/Vector Encoder (shared thin-head trunk)
    ↓ [B, 4096, 256] feature map
Parallel Decoder
    ↓ [B, 35_classes, 5_params] all at once
    ↓ (center_x, center_y, width, height, confidence)
Post-processing (threshold + NMS per class)
```

**Key Components:**
- **Dual-formulation training:** Joint NTP + MTP
  - NTP (autoregressive): causal reasoning
  - MTP (parallel): fast inference
- **Hybrid inference mode:**
  - Confidence check: top-1 prob < 0.7 → fallback to NTP
  - Spatial ambiguity: peak proximity < 8px → fallback
- **Block-based output:** Semantic, Box, Negative, End blocks

**Implementation:** 5 days
- Day 1-2: Design parallel decoder head
- Day 3: Implement dual-formulation loss
- Day 4: Hybrid inference logic
- Day 5: Training + validation

**Expected:** 3-5× speed, +1-2 AP50:95

#### Approach 2B: Diffusion Parallel Decoder (PerceptionDLM)

**Architecture:**
```
Image/Vector Encoder
    ↓ [B, 4096, 256] feature map
Diffusion Decoder (non-autoregressive)
    ↓ Parallel denoising for all 35 classes
    ↓ [B, 35, max_objects, 5] predictions
Structured Attention Masking
    ↓ Efficient multi-region perception
Final Boxes (per-class NMS)
```

**Key Components:**
- **Parallel denoising:** All classes in one forward pass
- **Structured masking:** Efficient attention for multi-region
- **Token + sequence parallelism:** True parallel decoding

**Implementation:** 2-3 weeks (major rewrite)
- Week 1: Prototype diffusion decoder
- Week 2: Training pipeline + validation
- Week 3: Debugging + optimization

**Expected:** 5-10× speed, unknown quality (high risk)

**Comparison:**
- **2A (LocateAnything):** Lower risk, proven on LVIS/COCO
- **2B (PerceptionDLM):** Higher risk, newer technique

**Recommendation:** Start with 2A, explore 2B if 2A succeeds

---

### Track 3: Foundation Model Upgrade (FLIP)

**Motivation:** Replace ViT backbone with object-centric, scale-invariant encoder

**Architecture:**
```
FLIP Encoder (fovea-like attention)
    ↓ Object-centered high-res patches
    ↓ Coarse peripheral context
    ↓ Off-grid, scale-invariant features
Thin-head Decoder (unchanged)
    ↓ CenterNet predictions
```

**Key Components:**
- **Fovea-like input patching:**
  - High-resolution around object centers
  - Coarse resolution at periphery
  - Biologically-inspired top-down attention
- **Scale invariance:**
  - Objects from 0.0001% to 25% of image area
  - Critical for FloorPlanCAD (small fixtures to large rooms)
- **Parameter efficiency:**
  - FLIP-Tiny: 0.51M params, 79.90% IoU
  - FLIP-Large: 96.6M params, 83.26% IoU
  - vs SAM2-L: 224.45M params, 75.87% IoU

**Benefits for FloorPlanCAD:**
1. **Small object handling:** elevator, escalator (currently 0.023, 0.296 AP)
2. **Speed:** 2× faster than SAM2-L
3. **Memory:** 440× fewer parameters
4. **Slot-based tracking:** Future video/multi-view extension

**Implementation:** 2-3 weeks
- Week 1: Integrate FLIP as backbone encoder
- Week 2: Train end-to-end with thin-head decoder
- Week 3: Scale-invariance validation on rare classes

**Expected:** +3-5 AP on small objects, 2× speed, −30% memory

---

## Recommended Execution Plan

### Phase 2A: Conservative Track (5 weeks)

**Week 1-2: VGent Reasoning Selector**
- Freeze thin-head, train selector on teacher reasoning
- Expected: +1-2 AP50:95
- Risk: LOW (modular, frozen baseline)

**Week 3: Multi-target Reasoning**
- Extend to multi-class queries
- Expected: 2-3× speed
- Risk: MEDIUM (RL training)

**Week 4-5: LocateAnything Parallel Decoder**
- Replace sequential queries with parallel prediction
- Expected: 3-5× speed, +1-2 AP50:95
- Risk: MEDIUM (proven technique, new codebase)

**Total Gain:** +3-4 AP50:95, 5-10× speed, ~5 weeks

---

### Phase 2B: Aggressive Track (4 weeks)

**Week 1: Skip dual-stream, validate thin-head frozen**
- Confirm thin-head as stable baseline

**Week 2-3: LocateAnything Parallel Decoder**
- Direct implementation, no reasoning integration yet
- Expected: 3-5× speed, +1-2 AP50:95

**Week 4: FLIP Foundation Model**
- Replace backbone with object-centric encoder
- Expected: +3-5 AP on small objects

**Total Gain:** +4-7 AP50:95, 6-10× speed, ~4 weeks  
**Risk:** HIGHER (major architecture changes)

---

### Phase 2C: Exploration Track (6 weeks)

**Week 1-2: VGent Reasoning Selector**
- Baseline improvement with teacher reasoning

**Week 3-4: PerceptionDLM Diffusion Decoder**
- Explore diffusion-based parallel decoding
- Compare with LocateAnything approach

**Week 5-6: FLIP + Reasoning Integration**
- Combine object-centric encoder with reasoning selector
- Expected: Maximum quality + speed

**Total Gain:** +5-8 AP50:95, 10-15× speed, ~6 weeks  
**Risk:** HIGHEST (multiple unproven combinations)

---

## Success Criteria

### Target Metrics (After Phase 2)

| Metric | Current | Target | Benchmark |
|---|---:|---:|---:|
| AP50 | 0.492 | **≥ 0.60** | Faster R-CNN 0.602 |
| AP50:95 | 0.306 | **≥ 0.35** | — |
| Rare class AP (elevator, escalator, ramp) | 0.106 | **≥ 0.25** | — |
| All-class latency | 5.8ms | **≤ 2ms** | — |
| Parameters | 6.8M | **≤ 10M** | — |

### Validation Gates

1. **≥ 2 seeds** for architecture claims
2. **Same-manifest controls** for comparisons
3. **val_select → val_calibration → test_set** hierarchy
4. **Speed + quality** reported together (no speed-only claims)

---

## Decision Matrix

| Question | Answer | Rationale |
|---|---|---|
| Which track to start? | **2A: Conservative** | Proven techniques, controlled risk |
| VGent or LocateAnything first? | **VGent** | Leverages existing teacher data, modular |
| Explore PerceptionDLM? | **Later** | After LocateAnything validates parallel decoding |
| FLIP foundation model? | **Phase 3** | After parallel decoder established |
| Multi-paper combination? | **Yes** | VGent + LocateAnything + FLIP (sequential) |

---

## Next Immediate Actions

1. ✅ **Validate dual-stream training** (in progress)
2. **Extract teacher reasoning features** (15,663 samples → 256-dim embeddings)
3. **Implement VGent-style cross-attention selector**
4. **Train selector on train_new, validate on val_select**
5. **Replicate with seed 2024**
6. **If successful:** proceed to LocateAnything parallel decoder
7. **If not:** diagnose and fix selector before continuing

---

## Risk Assessment

| Component | Risk Level | Mitigation |
|---|---|---|
| VGent Reasoning Selector | LOW | Frozen baseline, modular design |
| Multi-target Reasoning | MEDIUM | Start with single-class, then extend |
| LocateAnything Parallel | MEDIUM | Proven on LVIS/COCO, solid foundation |
| PerceptionDLM Diffusion | HIGH | Major rewrite, unproven for detection |
| FLIP Foundation Model | MEDIUM-HIGH | Large change, but proven on segmentation |
| Full Stack (VGent+Locate+FLIP) | HIGH | Many moving parts, sequential rollout |

**Mitigation Strategy:**
- Validate each component independently before stacking
- Always maintain frozen thin-head as fallback baseline
- Use val_select for selection, val_calibration for tuning
- ≥2 seeds before claiming improvement
- test_set only for final frozen report

---

## Expected Timeline

### Conservative Path (Recommended)
- **Week 1-2:** VGent reasoning selector (+1-2 AP50:95)
- **Week 3:** Multi-target reasoning (2-3× speed)
- **Week 4-5:** LocateAnything parallel decoder (+1-2 AP50:95, 3-5× speed)
- **Week 6:** Validation, 2-seed replication
- **Week 7:** test_set final report

**Total:** 7 weeks to **AP50 ≥ 0.60, AP50:95 ≥ 0.35, latency ≤ 2ms**

### Aggressive Path
- **Week 1:** Thin-head frozen validation
- **Week 2-3:** LocateAnything parallel decoder
- **Week 4:** FLIP foundation model
- **Week 5:** Validation + replication
- **Week 6:** test_set final report

**Total:** 6 weeks to **AP50 ≥ 0.63, latency ≤ 1.5ms** (higher risk)

---

## Conclusion

Với 5 papers mới, chúng ta có roadmap rõ ràng để:
1. ✅ Leverage teacher reasoning (VGent)
2. ✅ Parallel decoding (LocateAnything > PerceptionDLM)
3. ✅ Foundation model upgrade (FLIP)

**Next decision point:** Sau khi dual-stream training hoàn thành, chọn:
- **Option A:** Conservative - VGent → LocateAnything → FLIP
- **Option B:** Aggressive - LocateAnything → FLIP (skip VGent)
- **Option C:** Exploration - PerceptionDLM diffusion (high risk)

**Recommended:** Option A - proven techniques, controlled risk, sequential validation.
