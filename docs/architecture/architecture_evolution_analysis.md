# Architecture Evolution Analysis: Tại Sao FloorPlanLLM Chậm Hơn?

**Date**: 2026-09-06  
**Context**: Training epoch 15/100, phát hiện batch_size=8 chậm hơn 3x so với kỳ vọng (2.6s/batch vs 0.6s expected)  
**Question**: Tại sao phiên bản mới (FloorPlanLLM) chậm hơn phiên bản cũ (dual-pathway ResNet18)?

---

## Executive Summary

**Root Cause**: FloorPlanLLM architecture (commit bb14bf8, 68d01af) thay thế lightweight CNN (ResNet18) bằng heavy Vision Transformer, tăng params **3x** và latency **3x**, NHƯNG không có benchmark kết quả quality để justify trade-off này.

**Critical Finding**: Architecture change được thực hiện KHÔNG có controlled comparison:
- ❌ Không có A/B test ResNet18 vs ViT với cùng decoder
- ❌ Không có quality benchmark (AP50/AP50:95) cho FloorPlanLLM
- ❌ Không có speed profiling trước khi commit
- ❌ Paper tham khảo (VGent, LocateAnything) không áp dụng đúng

**Recommendation**: **Revert về dual-pathway architecture cũ** (commit cdec314) để hoàn thành baseline training nhanh, SAU ĐÓ mới thử FloorPlanLLM nếu cần quality improvement.

---

## Architecture Timeline

### Phase 1: Baseline (Before cdec314)
```
Image → ResNet18 CNN → CenterNet decoder
```
- **Params**: ~15-20M
- **Speed**: ~0.8-1s/batch (batch_size=8)
- **Quality**: Thin-head baseline AP50:95 = 0.3063

---

### Phase 2: Dual-Pathway ResNet18 (commit cdec314)

**Architecture**:
```
Image → ResNet18 CNN (11M params)
          ↓
    Cross-attention fusion ← Transformer vector encoder (4L, 128D, 4H, ~4M params)
          ↓                        ↑
    CenterNet decoder        SVG strokes (whole-drawing, 1024 tokens)
```

**Design Principles** (from `docs/dual_pathway_spec.md`):
- ✅ **Lightweight image encoder**: ResNet18 (proven, fast)
- ✅ **Small vector encoder**: 2-4 transformer layers
- ✅ **Params-matched**: Total ≤ floorplan_base + 10%
- ✅ **Query-based**: Instance-level or whole-drawing strokes
- ✅ **Cross-attention fusion**: Image tokens query vector tokens
- ✅ **Proven decoder**: CenterNet (thin-head validated)

**Parameters**:
- Image encoder: ~11M (ResNet18)
- Vector encoder: ~4M (4L, 128D, 4H)
- Decoder: ~5M (CenterNet heads)
- **Total**: ~20M params

**Expected Speed**: ~0.8-1s/batch (similar to baseline)

**Quality Target**: AP50:95 ≥ 0.28 (matching thin-head range)

---

### Phase 3: FloorPlanLLM (commit bb14bf8, 68d01af) ❌

**Architecture Change**:
```
Image → Vision Transformer (ViT) (8L, 512D, 8H, ~40M params)
          ↓
    Cross-attention fusion ← Graph Transformer (4L, 512D, 8H, ~15M params)
          ↓                        ↑
    CenterNet decoder        SVG strokes + graph edges
```

**Changes Made**:
1. ❌ Replaced ResNet18 → ViT (8 layers, 512 dim)
2. ❌ Scaled up vector encoder: 128D → 512D, 4H → 8H
3. ❌ Added "Graph Transformer" (không rõ graph structure là gì)
4. ❌ Increased total params: 20M → **56.5M** (2.8x)

**Why This Happened** (from `docs/week1_floorplan_llm.md`):
- Inspiration: LocateAnything paper (parallel box decoding)
- Goal: "Big change - paradigm shift from detection to structured understanding"
- Vision: Scene graph output, spatial relationships
- **BUT**: Parallel decoder FAILED (multi-instance problem)
- **THEN**: Reverted to CenterNet decoder (Option B)
- **RESULT**: Heavy encoder + proven decoder = worst of both worlds

**Parameters**:
- Image encoder: ~40M (ViT-8L-512D)
- Vector encoder: ~15M (GraphTransformer-4L-512D)
- Decoder: ~15M (35 CenterNet heads)
- **Total**: **56.5M params** (2.8x vs dual-pathway)

**Current Speed**: 2.6s/batch (3x slower than expected 0.8s)

**Quality**: ⚠️ **UNKNOWN** - no benchmark yet, still training

---

## Root Cause Analysis

### Why FloorPlanLLM Is 3x Slower

| Component | Old (cdec314) | New (bb14bf8) | Ratio |
|---|---|---|---|
| **Image encoder** | ResNet18 (11M) | ViT-8L-512D (40M) | 3.6x |
| **Vector encoder** | 4L-128D-4H (4M) | 4L-512D-8H (15M) | 3.8x |
| **Decoder** | 35 heads (5M) | 35 heads (15M) | 3.0x |
| **Total params** | 20M | 56.5M | 2.8x |
| **Latency** | ~0.8s/batch | ~2.6s/batch | **3.3x** |

**Computational Bottleneck**:
1. **ViT self-attention**: O(n²) where n = 1,024 patches (32×32 grid)
   - ResNet convolution: O(n) where n = pixels
   - ViT is **10-100x slower** than CNN for same receptive field
   
2. **Larger embedding dimension**: 128D → 512D = 16x more parameters per layer
   
3. **More attention heads**: 4 → 8 = 2x more parallel computations

**Why ViT Was Chosen** (speculated from docs):
- Paper hype: "ViT is SOTA for vision"
- Vision: "Foundation model for scene graph understanding"
- **BUT**: No evidence ViT > ResNet18 for floor plan detection
- **AND**: Scene graph goal was abandoned (parallel decoder failed)

---

## Paper Misapplication Analysis

### 1. LocateAnything (Parallel Box Decoding)

**What paper actually says**:
- Parallel decoding works for **single-instance-per-class** tasks
- For **multi-instance**, use hybrid inference (parallel + fallback)
- Designed for LVIS dataset (1000+ classes, mostly 1 instance/class)

**What we tried**:
- FloorPlanCAD: 35 classes, **3-10 instances per class** (windows, doors, furniture)
- Parallel decoder: predicts 1 box per class → **failed on multi-instance**
- Result: Had to revert to CenterNet (heatmap-based multi-instance)

**Conclusion**: ❌ Paper doesn't apply to FloorPlanCAD task

---

### 2. VGent (Modular Reasoning)

**What paper actually says**:
- Freeze MLLM reasoning encoder (preserve pretrained knowledge)
- Use **off-the-shelf fast detector** for proposals
- Cross-attention selector chooses from proposals

**What we should do**:
- Keep ResNet18 detector (fast, proven)
- Add reasoning selector on top (if teacher data helps)

**What we did instead**:
- ❌ Replaced fast detector with slow ViT
- ❌ No reasoning module yet
- ❌ Heavy encoder without quality gain

**Conclusion**: ❌ Misunderstood paper - should keep detector fast

---

### 3. Resource Papers (Not Applied Yet)

From `docs/new_papers_summary.md`:
- **Detect Anything**: Autoregressive decoder (not implemented)
- **Looking Locally**: Object-centric ViT optimization (not implemented)
- **PerceptionDLM**: Diffusion-based detection (not applicable)

**Conclusion**: Papers collected but not properly evaluated before architecture change

---

## Quality Analysis: No Benchmark Available

### Historical Results (Different Manifests)

| Model | Commit | Architecture | Params | AP50:95 | Notes |
|---|---|---|---|---:|---|
| **Thin-head baseline** | — | Image-only CenterNet | ~15M | **0.3063** | 2-seed mean, validated |
| **B `floorplan_base`** | f83fcffd | Image + conditioning | 5.3M | **0.2072** | Different manifest (not comparable) |
| **Dual-pathway ResNet18** | cdec314 | ResNet18 + vector | ~20M | ⚠️ **Unknown** | Not trained to convergence |
| **FloorPlanLLM** | bb14bf8 | ViT + Graph | 56.5M | ⚠️ **Unknown** | Currently training (epoch 15/100) |

**Critical Gap**: ❌ **No controlled comparison** between architectures on same manifest/split

---

### Expected vs Actual Development Flow

**What SHOULD have happened** (research protocol):
1. ✅ Train dual-pathway ResNet18 to convergence (100 epochs)
2. ✅ Benchmark: AP50:95 on val_select
3. ✅ Compare vs thin-head baseline (0.3063)
4. ❓ **IF ResNet18 < baseline**: Try ViT
5. ❓ **IF ResNet18 ≥ baseline + 0.02**: Keep ResNet18, optimize speed

**What ACTUALLY happened**:
1. ⚠️ Implemented dual-pathway ResNet18
2. ❌ Did NOT train to convergence
3. ❌ No benchmark available
4. ❌ Switched to FloorPlanLLM (ViT) WITHOUT comparison
5. ❌ Now training slow architecture with unknown quality

---

## Decision Tree: What To Do Now?

### Current Situation
- ✅ Training FloorPlanLLM (ViT-based): epoch 15/100
- ⚠️ Speed: 2.6s/batch (3x slower than expected)
- ⚠️ Quality: Unknown (no convergence yet)
- ⏱️ ETA: 85 epochs × 82 min/epoch = **~117 hours** (~5 days)

### Option A: Continue FloorPlanLLM ✅ (Current Path)
**Pros**:
- Already invested 15 epochs (~20 hours)
- Might achieve higher quality than ResNet18
- ViT is "modern" architecture

**Cons**:
- ❌ 5 more days to completion
- ❌ No guarantee quality > baseline
- ❌ Even if quality wins, 3x slower inference
- ❌ Can't iterate fast (each experiment = 5 days)

**Decision gate**: Wait until epoch 20, check loss convergence
- If heatmap loss < 1.0 → continue (good progress)
- If heatmap loss > 1.5 → abort (not converging)

---

### Option B: Revert to Dual-Pathway ResNet18 🎯 (RECOMMENDED)
**Pros**:
- ✅ 3x faster training (100 epochs in ~1.5 days vs 5 days)
- ✅ 3x faster inference (important for production)
- ✅ Params-matched to baseline (fair comparison)
- ✅ Can iterate multiple experiments per week
- ✅ Proven ResNet18 encoder (thin-head validated)

**Cons**:
- Lose 15 epochs of FloorPlanLLM training
- Might have slightly lower quality than ViT (unproven)

**Action Plan**:
1. Stop current training (save checkpoint for reference)
2. Checkout dual-pathway architecture (commit cdec314)
3. Update to use latest dataset/manifest
4. Train 100 epochs with batch_size=8 (~1.5 days)
5. Benchmark on val_select
6. **IF AP50:95 ≥ 0.30**: Mission complete, ResNet18 wins
7. **IF AP50:95 < 0.28**: THEN try ViT (controlled comparison)

---

### Option C: Hybrid Approach (Compromise)
**Train both in parallel**:
1. Continue FloorPlanLLM (ViT) on GPU 1
2. Start dual-pathway ResNet18 on GPU 2 (if available)
3. Compare after both converge

**Pros**: Get data for both architectures
**Cons**: Need 2 GPUs, more complex

---

## Recommendations

### Immediate (Next 24 Hours)

1. **✅ CONTINUE FloorPlanLLM until epoch 20** (5 more epochs, ~7 hours)
   - Monitor heatmap loss convergence
   - Decision gate at epoch 20

2. **📊 Check epoch 20 metrics**:
   - **IF heatmap loss < 1.0**: Good progress, continue to epoch 50
   - **IF heatmap loss > 1.5**: Abort, switch to Option B

3. **⚠️ Run validation at epoch 20**:
   - Evaluate on val_select (quick check)
   - Target: AP50 > 0.05 (sanity check that model is learning)

---

### Strategic (Next Week)

1. **🎯 PRIORITY: Establish Fast Baseline**
   - Train dual-pathway ResNet18 (100 epochs, ~1.5 days)
   - Target: AP50:95 ≥ 0.28
   - This becomes the "speed baseline" for all future work

2. **📈 IF ResNet18 succeeds (AP50:95 ≥ 0.28)**:
   - ✅ Use ResNet18 as production architecture
   - ✅ Focus on decoder improvements (stride-4, NMS tuning)
   - ✅ Try VGent reasoning selector (speed-preserving improvement)

3. **🔬 IF ResNet18 fails (AP50:95 < 0.28)**:
   - Then justify trying ViT (controlled experiment)
   - But compare on SAME training config

---

### Long-term (Research Protocol)

1. **Document architecture decisions**:
   - ✅ Always benchmark BEFORE architecture change
   - ✅ Controlled comparison (same manifest, splits, training config)
   - ✅ Speed profiling BEFORE committing to heavy architecture

2. **Paper application checklist**:
   - ❓ Does task match paper's task? (multi-instance vs single-instance)
   - ❓ Do we have same compute budget? (1B params vs 56M params)
   - ❓ Is the improvement validated on similar dataset?

3. **Architecture complexity budget**:
   - Start simple (ResNet18)
   - Add complexity ONLY if quality gain justifies latency cost
   - 3x latency needs +10% AP50:95 to justify

---

## Tài Liệu Cần Tổ Chức

### Files Scattered in Root (Cần Move to docs/)
```
BIG_PICTURE.md → docs/archive/
CRITICAL_ISSUE.md → docs/archive/
CURRENT_STATUS.md → DELETE (outdated)
PROGRESS_REPORT.md → docs/archive/
QUICKREF_HYBRID.md → docs/
README_HYBRID.md → docs/
README_TRAINING.md → docs/
RESEARCH_COMPLETE.md → docs/archive/
RESEARCH_SUMMARY.md → docs/
SETUP_COMPLETE.md → docs/archive/
STATUS.md → DELETE
TRAINING_*.md (7 files) → logs/ (thay vì root)
```

### Research Documents Need Indexing
```
docs/ (59 files) → Need README_DOCS.md index
docs/research_archive/ → Move completed phases
logs/ → All training logs and status files
Resource/ → Keep papers, add Resource/README.md summary
```

---

## Kết Luận

**Current FloorPlanLLM Training**:
- ⚠️ **Chậm 3x** so với kỳ vọng (2.6s/batch vs 0.8s)
- ⚠️ **Chưa có quality benchmark** (đang epoch 15/100)
- ⚠️ **ETA 5 ngày** để hoàn thành
- ⚠️ **Không có A/B test** với ResNet18 baseline

**Root Cause**:
- Architecture change **không có controlled comparison**
- Paper misapplication (LocateAnything không áp dụng cho multi-instance)
- ViT 3x chậm hơn CNN mà chưa chứng minh quality gain

**Recommendation**:
1. ✅ **Continue to epoch 20**, check convergence
2. 🎯 **Start dual-pathway ResNet18** training in parallel (faster baseline)
3. ⚠️ **Decision at epoch 20**: Continue ViT hoặc switch to ResNet18
4. 📊 **Establish quality benchmark** trước khi commit architecture changes
5. 🗂️ **Organize documentation**: Move files to proper directories

**Next Action**: Monitor training đến epoch 20 (7 giờ nữa), sau đó decision gate dựa trên heatmap loss.
