# Tài Liệu Tham Khảo Mới - Resource/

**Ngày:** 2026-09-05  
**Context:** Sau khi hoàn tất Option B (DualStreamCenterNet), có 4 papers mới để tham khảo cho phase tiếp theo.

---

## 1. Detect Anything via Next Point Prediction (34.3 MB)

**Chủ đề:** Sequential detection, autoregressive decoding

**Relevance:**
- Alternative decoder strategy vs CenterNet heatmap
- Autoregressive box generation (point-by-point)
- Có thể giải quyết multi-instance prediction tốt hơn parallel decoder

**Potential application:**
- Replace CenterNet decoder với autoregressive decoder
- Kết hợp với dual-stream encoder hiện tại
- Expected: tốt hơn cho dense scenes (nhiều instances/class)

**Priority:** HIGH - Directly addresses multi-instance detection challenge

---

## 2. Looking Locally - Object-Centric Vision Transformers (4.2 MB)

**Chủ đề:** Object-centric ViT, efficient segmentation

**Relevance:**
- ViT encoder optimization cho object detection
- Local attention mechanisms
- Foundation models cho segmentation

**Potential application:**
- Optimize image encoder trong dual-stream
- Replace hoặc augment hiện tại ViT backbone
- Reduce computational cost while preserving quality

**Priority:** MEDIUM - Encoder optimization sau khi decoder stable

---

## 3. PerceptionDLM - Parallel Region Perception (15.4 MB)

**Chủ đề:** Diffusion models for region perception

**Relevance:**
- Parallel region prediction (tương tự FloorPlanLLM failed attempt)
- Multimodal diffusion language models
- Có thể học được cách làm parallel prediction work

**Potential application:**
- Hiểu tại sao parallel approach failed
- Alternative formulation cho parallel decoder
- Diffusion-based detection thay vì direct regression

**Priority:** LOW - Parallel decoder đã fail, cần validate Option B trước

---

## 4. VGent - Visual Grounding via Modular Design (10.2 MB)

**Chủ đề:** Visual grounding, modular reasoning

**Relevance:**
- Disentangle reasoning và prediction
- Modular design cho vision-language tasks
- Có thể leverage teacher reasoning data

**Potential application:**
- Use teacher_distill.jsonl reasoning data
- Modular architecture: separate reasoning module + prediction module
- Improve interpretability và performance

**Priority:** MEDIUM-HIGH - Directly applicable to available teacher data

---

## Roadmap Integration

### Phase 1: Validate Option B Baseline (CURRENT)
- **Status:** Training DualStreamCenterNet (small, 10 epochs)
- **Target:** AP50:95 ≥ 0.28 to confirm encoder improvement
- **Timeline:** ~30-60 phút

### Phase 2: Quick Wins với Teacher Data (NEXT)
- **Paper:** VGent - modular reasoning
- **Action:** Integrate teacher reasoning as auxiliary supervision
- **Expected gain:** +1-2 AP50:95
- **Timeline:** 1-2 ngày

### Phase 3: Decoder Exploration
**Option A:** Autoregressive decoder (Detect Anything paper)
- Replace CenterNet với next-point prediction
- Better multi-instance handling
- Timeline: 3-5 ngày

**Option B:** Encoder optimization (Looking Locally paper)
- Object-centric ViT improvements
- Reduce cost while preserving quality
- Timeline: 2-3 ngày

### Phase 4: Advanced (if needed)
- PerceptionDLM: diffusion-based detection
- Large-scale data engine
- Timeline: 1-2 tuần

---

## Recommended Next Steps

1. ✅ **Wait for DualStreamCenterNet training** (~30 phút)
2. ✅ **Evaluate on val_select** và so với thin-head baseline
3. **If AP50:95 ≥ 0.28:** Option B validated → move to Phase 2 (VGent + teacher data)
4. **If AP50:95 < 0.28:** Debug hoặc try Detect Anything autoregressive decoder

**Key insight:** Có 4 high-quality papers làm foundation cho 2-3 tháng research tiếp theo. Không thiếu hướng đi!
