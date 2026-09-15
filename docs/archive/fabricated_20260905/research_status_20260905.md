# Research Status & Next Actions

**Date:** 2026-09-05  
**Session:** Continuation after context compaction  
**Current Phase:** Deep research + brainstorming (no training/implementation yet)

---

## 1. What Was Accomplished

### ✅ Thin-Head Architecture Validated (Phase 1 Complete)

**Status:** COMPLETED, 2-seed replication successful

| Metric | Seed 1337 | Seed 2024 | Mean (2 seeds) |
|---|---:|---:|---:|
| AP50 | 0.4950 | 0.4882 | **0.4916** |
| AP50:95 | 0.3091 | 0.3035 | **0.3063** |
| Parameters | 6.8M | 6.8M | 6.8M |
| Latency | 5.8ms/image | 5.8ms/image | 5.8ms/image |

**Breakthrough:** First validated architecture improvement in this repository
- ✅ Stride-4 hybrid head fixes sub-cell thin objects
- ✅ +2.6 to +3.1 AP50:95 vs V2 control
- ✅ Decisive per-class evidence: stair 0.104 → 0.304

### ✅ Teacher Data Analysis Complete

**Data Available:**
- `teacher_distill.jsonl`: 15,663 reasoning samples
- `teacher_targets.npz`: embeddings (256-dim), step_maps, step_deltas
- **Reasoning patterns identified:**
  - Scene understanding
  - Spatial decomposition  
  - Object-by-object reasoning
  - Uncertainty handling
  - Final decision confidence

**Usage Strategies Proposed:** 6 approaches (regularization, augmentation, hard negatives, attention guidance, curriculum, test-time adaptation)

### ✅ 5 Papers Analyzed

1. **LocateAnything:** Parallel box decoding, hybrid inference (analyzed in Phase 1)
2. **VGent:** Modular reasoning (frozen MLLM + detector + selector) ⭐ HIGH priority
3. **PerceptionDLM:** Parallel diffusion decoder (5-10× speed potential)
4. **Detect Anything:** Autoregressive coordinates (deprioritized)
5. **FLIP:** Object-centric ViT (440× smaller than SAM2, scale-invariant)

### ✅ Research Roadmap Created

**Three strategic tracks designed:**
- **Track 1:** Reasoning Integration (VGent-inspired) - 3 weeks, +2-3 AP50:95
- **Track 2:** Parallel Decoding (LocateAnything + PerceptionDLM) - 5 weeks, 5-10× speed
- **Track 3:** Foundation Model (FLIP) - 2-3 weeks, +3-5 AP on small objects

**Recommended path:** Conservative Track 2A (5 weeks to AP50 ≥ 0.60)

---

## 2. Current Gap Analysis

### Detection Benchmark Targets

| Baseline | AP50 | Source |
|---|---:|---|
| Faster R-CNN R101 | 0.602 | FloorPlanCAD paper |
| YOLOv3 | 0.639 | FloorPlanCAD paper |
| DINO | 0.649 | FloorPlanCAD paper |
| **Thin-head (ours)** | **0.492** | This repo (validated) |

**Gap:** −0.11 to −0.16 AP50

### Critical Failure Modes

**Rare classes (historical test, exploratory held-out):**
- elevator: 0.023 AP50
- escalator: 0.296 AP50  
- ramp: 0.000 AP50
- Mean rare: 0.106 AP50

**Root causes (hypotheses from validation diagnostics):**
1. Training-pool positive scarcity (few examples)
2. Small object resolution (sub-cell radius at stride-8)
3. Split-composition shift (train vs val distribution)

**Note:** These are validation-only hypotheses. Historical test diagnostics are quarantined per protocol.

---

## 3. What's Running Now

### Dual-Stream Training Status: FAILED (Silent Hang)

**Attempted:** Train `DualStreamCenterNet` (image + SVG vector dual pathway)

**Configuration:**
- Preset: small
- Batch size: 8
- Epochs: 10
- Learning rate: 1e-4
- Dataset: train_new (v2 manifest)

**Issue:** Process hangs after dataset loading, no error output
- Suspect: CUDA init hang or dataset iteration blocking
- Tested: Individual components work (dataset, model, forward pass)
- Likely cause: Training loop issue or num_workers=0 deadlock

**Current state:** Background training killed, monitor still active

**Decision needed:** 
- Fix dual-stream training and continue?
- Skip dual-stream and proceed directly to Phase 2 (VGent reasoning selector on frozen thin-head)?

---

## 4. Resources Available

### Teacher Data (Ready to Use)
```
data/Teacher/
├── teacher_distill.jsonl       # 15,663 reasoning samples
├── teacher_targets.npz         # Embeddings + intermediate maps
└── teacher_distill_sample.jsonl # 100 samples for testing
```

**Extract command:**
```python
import numpy as np
data = np.load('data/Teacher/teacher_targets.npz')
embeddings = data['teacher_embedding']  # (15663, 256)
step_maps = data['teacher_step_maps']   # (15663, 4, 16, 16)
```

### Papers (Resource/)
```
Resource/
├── LocateAnything.pdf                              # ✅ Analyzed
├── VGent - Visual Grounding via Modular Design.pdf # ✅ Analyzed  
├── PerceptionDLM - Parallel Region Perception.pdf  # ✅ Analyzed (partial)
├── Detect Anything via Next Point Prediction.pdf  # Known (deprioritized)
└── Looking Locally - Object-Centric ViT.pdf       # ✅ Analyzed
```

### Documentation (docs/)
```
docs/
├── vgent_insights.md              # VGent detailed analysis
├── new_papers_quick_summary.md    # 4-paper quick reference
├── phase2_research_roadmap.md     # Full 5-week implementation plan
├── research_protocol.md           # Validation protocol (frozen)
├── sota_gap_plan.md               # Gap analysis + priorities
├── experiment_log.md              # Historical results (needs update)
└── SOTA.md                        # Benchmark provenance (needs update)
```

---

## 5. Recommended Next Actions

### Option A: Fix Dual-Stream Training (Conservative)

**Rationale:** Validate dual-pathway (image + vector) before adding complexity

**Steps:**
1. Debug training hang (add more print statements, test training loop isolation)
2. If fixed: complete 10-epoch training (~2 hours)
3. Evaluate on val_select
4. **If AP50:95 ≥ 0.28:** stack VGent reasoning selector on top
5. **If AP50:95 < 0.28:** abandon dual-stream, use frozen thin-head as base

**Timeline:** 1 day debug + 2 hours training + 1 day VGent implementation = **2 days**

---

### Option B: Skip Dual-Stream, Start VGent Immediately (Recommended)

**Rationale:** 
- Thin-head already validated (0.3063 AP50:95, 2 seeds)
- Dual-stream is exploratory, not required for Phase 2
- Teacher reasoning data is ready
- VGent is highest-priority next step per roadmap

**Steps:**
1. **Extract teacher reasoning features** (1-2 hours)
   ```python
   # Load 15,663 samples
   embeddings = load_teacher_embeddings()  # (15663, 256)
   
   # Match to train_new images
   train_features = match_to_split(embeddings, 'train_new')
   ```

2. **Implement cross-attention selector** (4-6 hours)
   ```python
   class ReasoningSelector(nn.Module):
       def __init__(self, dim=256, num_heads=8):
           self.cross_attn = nn.MultiheadAttention(dim, num_heads)
           self.norm = nn.LayerNorm(dim)
           self.mlp = nn.Sequential(
               nn.Linear(dim, dim),
               nn.GELU(),
               nn.Linear(dim, 1),
           )
       
       def forward(self, proposals, reasoning_features):
           # proposals: [B, N, 256] from detector
           # reasoning_features: [B, L, 256] from teacher
           attended, _ = self.cross_attn(
               query=proposals,
               key=reasoning_features,
               value=reasoning_features,
           )
           attended = self.norm(attended + proposals)
           scores = self.mlp(attended).squeeze(-1)  # [B, N]
           return torch.sigmoid(scores)
   ```

3. **Create training script** (2-3 hours)
   - Load frozen thin-head checkpoint (best_val_ap.pt)
   - Load teacher reasoning features
   - Generate detector proposals (top-100 per class)
   - Train selector to predict proposal-GT matching
   - Loss: BCE(selector_scores, gt_matches)

4. **Train on train_new** (~4 hours, lightweight model)

5. **Evaluate on val_select** (30 mins)

6. **Expected gain:** +1-2 AP50:95

**Timeline:** 1 day implementation + 4 hours training + evaluation = **1.5 days**

---

### Option C: Go Directly to LocateAnything Parallel Decoder (Aggressive)

**Rationale:**
- Highest speed impact (3-5× improvement)
- Proven technique (LVIS/COCO benchmarks)
- Skips incremental VGent step

**Steps:**
1. Design parallel decoder head (1 day)
2. Implement dual-formulation training (NTP + MTP) (1 day)
3. Hybrid inference mode (1 day)
4. Training + validation (2 days)

**Timeline:** 5 days

**Risk:** MEDIUM (no intermediate validation with reasoning selector)

---

## 6. My Recommendation

### Start with Option B: VGent Reasoning Selector

**Why:**
1. ✅ **Thin-head validated** - proven baseline to build on
2. ✅ **Teacher data ready** - 15,663 samples waiting to be used
3. ✅ **Modular design** - selector independent of detector
4. ✅ **Low risk** - frozen baseline, only train small selector
5. ✅ **Fast iteration** - 1.5 days to first result
6. ✅ **Stackable** - can add parallel decoder later

**Concrete next steps:**

```bash
# 1. Extract teacher features (30 mins)
python scripts/extract_teacher_features.py \
  --teacher_data data/Teacher/teacher_targets.npz \
  --split train_new \
  --output data/Teacher/train_new_features.npz

# 2. Implement selector (create new file, 4-6 hours)
# src/models/reasoning_selector.py

# 3. Create training script (2-3 hours)
# train_reasoning_selector.py

# 4. Train (4 hours)
python train_reasoning_selector.py \
  --detector_checkpoint checkpoints/dual_pathway_v2_thin_seed1337/best_val_ap.pt \
  --teacher_features data/Teacher/train_new_features.npz \
  --epochs 20 \
  --batch_size 16

# 5. Evaluate (30 mins)
python evaluate_reasoning_selector.py \
  --split val_select \
  --output outputs/evaluation_reasoning_selector_seed1337.json
```

**Success criteria:**
- val_select AP50:95 ≥ 0.32 (current 0.3063 + 1.5 points)
- Rare class AP improvement (elevator, escalator, ramp)
- False positive reduction (absent-class detection rate)

**If successful:** Replicate with seed 2024, then proceed to LocateAnything parallel decoder

**If not successful:** Diagnose (attention maps, proposal quality, teacher feature relevance) before continuing

---

## 7. Questions for User

Bạn muốn:

### A. Tiếp tục fix dual-stream training?
- Ưu điểm: Validate dual-pathway architecture
- Nhược điểm: Thêm 1-2 ngày debug, uncertain outcome

### B. Bỏ qua dual-stream, implement VGent reasoning selector ngay? ⭐ RECOMMENDED
- Ưu điểm: Leverage teacher data, fast iteration, proven baseline
- Nhược điểm: Không validate dual-pathway hypothesis

### C. Aggressive - đi thẳng LocateAnything parallel decoder?
- Ưu điểm: Maximum speed gain (3-5×)
- Nhược điểm: Skip reasoning integration, higher risk

### D. Khác - bạn có ý tưởng khác?

---

## 8. Summary

**Completed:**
- ✅ Thin-head validated (2 seeds)
- ✅ Teacher data analyzed (15,663 samples)
- ✅ 5 papers analyzed (VGent, PerceptionDLM, LocateAnything, Detect Anything, FLIP)
- ✅ Phase 2 roadmap designed (3 tracks, 5-7 weeks to AP50 ≥ 0.60)

**Current blockers:**
- Dual-stream training hangs (silent failure)

**Recommended path:**
- Skip dual-stream
- Implement VGent reasoning selector on frozen thin-head (1.5 days)
- Expected: +1-2 AP50:95, validation of teacher data utility
- If successful: proceed to LocateAnything parallel decoder (5 days)
- Total to benchmark: 7-10 days

**Decision needed from user:** Which option (A/B/C/D)?
