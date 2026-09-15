# GLM-OCR & PaddleOCR Literature Survey → FloorPlanCAD Detector

**Date written:** 2026-09-14
**Papers surveyed:** 6
- `docs/research/glm/2603.10910v2.pdf` — GLM-OCR Technical Report (Zhipu AI + Tsinghua, 2026-03)
- `docs/research/paddleocr/2507.05595v1.pdf` — PaddleOCR 3.0 (2025-07)
- `docs/research/paddleocr/2510.14528v4.pdf` — PaddleOCR-VL 0.9B (2025-10)
- `docs/research/paddleocr/2601.21957v2.pdf` — PaddleOCR-VL-1.5 (2026-01)
- `docs/research/paddleocr/2606.03264v1.pdf` — PaddleOCR-VL-1.6 (2026-06)
- `docs/research/paddleocr/2606.13108v1.pdf` — PP-OCRv6 (2026-06)

**Companion to:** `docs/research/deepseek_transfer_report_20260914.md` (14 papers, 2026-09-13/14)
**Plan:** `docs/research/implementation_plan_20260914.md`
**Target model:** `dual_pathway_v2_thin` — 6.79M params, dual-pathway, stride-8 CNN trunk
+ hybrid stride-4 head, query-conditioned CenterNet, 35 classes
**Baseline (caveated):** `val_select` AP50 0.4950 / AP50:95 0.3091, seed 1337, `epoch_030.pt`
**External target:** Faster R-CNN R101 AP50 0.602

> **Standing caveat.** The baseline is measured on a manifest with an 11% train/val
> content leak (see the DeepSeek report §2). **That fix is blocking** and precedes
> everything below.

---

## 0. TL;DR

**One paper matters: PP-OCRv6.** It is a *small detection* model (0.43M–15.5M params) that
beats billion-parameter VLMs on detection by **+39.4 points Hmean**, and it does so with
four mechanisms we can copy. Its ablations isolate each one, and one of them —
**multi-scale training** — targets our measured bottleneck directly.

Its companion finding is the most important strategic datum in either survey: PP-OCRv6
names the failure mode of large VLMs on detection as **imprecise localization** — "overly
tight bounding boxes", misaligned regions, false positives, and specifically **"fails to
detect small-sized characters."** That is our failure mode, in a different domain.

The other five papers are OCR/VLM generation systems. Their machinery is either
inapplicable (autoregressive decode, coordinate-token vocabularies, GRPO) or
**directionally harmful** for us — notably PaddleOCR-VL's `merge size=2` visual-token
reduction, which is a 2×2 spatial average applied to cut decode cost, i.e. exactly the
operation that destroys sub-cell detail our thin objects depend on.

**Three experiments worth running**, all data/training-side, no architecture risk:
1. **Multi-scale training** — highest value; we currently have *no* scale augmentation.
2. **Auxiliary deep supervision at intermediate strides** — free at inference.
3. **SVG-based targeted synthesis for stair/ramp** — we hold exact ground-truth geometry.

---

## 1. PP-OCRv6 (arXiv 2606.13108v1, 2026-06) — the paper that transfers

### 1.1 Core claim

A three-tier OCR family (end-to-end **1.5M / 7.7M / 34.5M** params) where the medium tier
(15.5M detection + 19M recognition) reaches **86.2% detection Hmean** vs **46.8%** for
Gemini-3.1-Pro and 81.6% for their own previous server model. The stated conclusion: the
gain came from **architectural redesign, not scale or data alone** — the previous
1.19M-param mobile baseline was structurally saturated.

### 1.2 Verified relevance to us

| Our property | PP-OCRv6 equivalent |
|---|---|
| Small model, detection task | 0.43M–15.5M params, text detection |
| Measured bottleneck: thin objects at stride 8 | "fails to detect small-sized characters" (their VLM critique) |
| Single-scale training, fixed 512×512 | Trains across scales; dedicated resolution-robustness benchmark |
| Output stride 8, hybrid stride-4 head | FPN strides **4, 8, 16, 32** |
| focal + smooth-L1, no aux heads | Focal + Dice, aux heads at P2/P3/P4 |

### 1.3 The four transferable mechanisms, with their ablation values

Their detection ablation (Table 13, v6_small det, 1/5 data):

| Step | Hmean | Δ |
|---|---:|---:|
| PP-OCRv5_mobile_det baseline | 77.84 | — |
| + LCNetV4 backbone | 79.37 | **+1.53** |
| + RepLKFPN neck | 79.75 | +0.38 (params *drop* 2.53M→2.48M) |
| + Auxiliary deep supervision | 80.28 | +0.53 |
| + **Focal loss** | 81.43 | **+1.15** |
| + full data | 84.12 | +2.69 |
| **Total vs baseline** | **+6.28** | |

**★ (a) Multi-scale / resolution-randomized training.** Their resolution-robustness
benchmark (600 images × 7 scales, 0.35×–2.83×) shows medium-tier mean 86.67 with **CV
5.19%**, against the previous server model's 79.98 / 8.02%. At the extreme 2.83× scale all
v6 models hold >81% (medium 87.94) vs 71.4% and 65.81% for the older models.

**This is our largest gap.** Verified two ways:

*Code path* — `src/data/transforms.py`: `PairedTrainTransform` applies horizontal flip
(p=0.5), vertical flip (p=0.0), 90° rotation (p=0.5), and photometric jitter. The only
resizes present are **fixed** target-size resizes (`TF.resize(image, [height, width])` at
line 284 in the train path, line 200 in eval) plus the corresponding box coordinate
remapping — no scale jitter, no random crop, no random resized crop.

*Empirical* — running `PairedTrainTransform` 200 times on a 1000×1000 canvas with one thin
box (40×5, a stair-like aspect ratio) produces exactly **2 distinct box sizes**, and the
min-side is **always 2.56 px**:

```
distinct (w,h) box sizes over 200 samples: 2
    (2.56, 20.48)      # original orientation
    (20.48, 2.56)      # after 90° rotation
min-side values seen: [2.56]
```

So the model never sees a thin object at a different apparent size. Every training image is
a fixed 512×512 square resize. Our failure is objects below one stride-8 cell; scale
diversity changes apparent size, so a given stair is sometimes above and sometimes below one
cell — which is exactly the supervision that teaches sub-cell cues. It is the standard,
cheap remedy and we are not using it.

**★ (b) Auxiliary deep supervision.** Independent auxiliary heads at **P2, P3, P4**
(strides 4/8/16) predicting the same targets as the main branch, weighted **P2=0.4,
P3=0.3, P4=0.2**, and **removed at inference (zero cost)**. Their stated rationale:
without it, gradients must traverse the whole FPN to reach the backbone.

Verified relevant: we have a hybrid stride-4 head (`head_stride` in `ModelConfig`,
`_resolve_head_features` in `detector.py` upsamples stride-8 trunk features to a 128×128
grid), but **only the final output is supervised**. There is no intermediate supervision.
The agent's read is correct and useful: our stride-4 head already puts the *capacity* where
thin objects live; what is missing is gradient signal at intermediate scales.

**★ (c) Focal + Dice pairing on the detection map.** Their largest loss-side gain
(+1.15) came from adding per-pixel Focal to a Dice-only baseline. Their argument against
OHEM: Dice gives one scalar for global overlap, OHEM ranks pixels but is dominated by
global overlap, so neither gives true per-pixel hard-example gradients — Focal does.

Verified: `src/training/losses.py` contains only `focal_loss`, `l1_loss_masked`,
`centernet_loss`. **No Dice.** We already have the Focal half, so the marginal item is the
Dice term.

**Caveat, stated honestly:** their maps are dense shrink/threshold/binary regions; ours are
sparse Gaussian peaks (and 33.6% of our boxes have gaussian radius 0 at stride 8, a known
pathology). The transfer is not clean. Treat as a medium-confidence single A/B.

**★ (d) RepLKFPN-style large-kernel depthwise refinement.** Per-level refinement block =
7×7 DW main branch + 3 parallel dilated DWs (5×5 d=1, 3×3 d=2, 3×3 d=3) + 1×1 PW + SE,
**fused into a single 7×7 DW at inference**. Per-level receptive field **3×3 → 7×7** while
neck params **drop 172K → 118K**. Rationale: densely-arranged thin structures need RF at
each pyramid level.

Their ablation gain is small (+0.38) — low risk, low reward. Not a headline.

### 1.4 Supporting items

- **CopyPaste augmentation** — increases thin-instance density per image. Our thin classes
  are rare *and* small, which compounds. Cheap, standard.
- **Class-matched teacher discipline for distillation**: their tiny-model recipe is
  CTC KL-divergence on logits from a **vocabulary-matched** teacher. The analogue for us is
  a **class-matched** teacher — same 35-class output layer, no adapter. Relevant if we ever
  revisit distillation (currently out of scope per the plan).
- **Data curation**: difficulty filtering at **confidence 0.95–0.97** (a sweet spot, not
  "keep the hardest"), accuracy verification, diversity sampling via **1000 K-means
  clusters**.
- **BN zero-init on residual compress layers** — makes every block identity at init.
  Cheap stabiliser, useful only if we deepen the trunk.

### 1.5 Their explicit small-vs-large analysis (quoted for the record)

Three named VLM failure modes on detection: **imprecise localization, hallucination,
computational inefficiency**. On localization specifically: VLMs produce "overly tight
bounding boxes", misaligned regions, false positives on non-text; Gemini-3.1-Pro "fails to
detect small-sized characters." Their medium model beats Gemini-3.1-Pro on detection by
**+39.4 points** (86.2 vs 46.8); even their 0.43M tiny detector is **+33.8**.

This is direct external corroboration of the project's architecture philosophy: a small
specialised detector beats a large general model on *localization*. It is also a warning
against ever pivoting this project to an end-to-end VLM for detection.

---

## 2. GLM-OCR (arXiv 2603.10910v2, 2026-03)

### 2.1 Core claim

A 0.9B document model (0.4B CogViT vision encoder + 0.5B GLM decoder) with **Multi-Token
Prediction**: trained to predict 10 tokens per step, generating 5.2 tokens per decoding
step on average at inference → **~50% throughput improvement**, with parameter sharing
across the draft heads to bound memory. OmniDocBench v1.5 **94.6**; OCRBench (Text) 94.0;
UniMERNet 96.5.

### 2.2 The one transferable idea: decompose before you recognise

Their stated design rationale #1 is the useful part, and it is a *decomposition* argument
rather than a modelling trick:

> "small-scale models are highly susceptible to hallucinations and repetitive generation
> when processing documents with complex layouts. By explicitly introducing a layout
> analysis module prior to recognition, we decompose complex layout structures into
> multiple simpler sub-problems, significantly enhancing the overall performance and
> stability of the model. Furthermore, partitioning large, complex pages into smaller,
> independent regions allows for parallel recognition."

Their pipeline is therefore **two-stage**: PP-DocLayout-V3 detects structured regions, then
each region is recognised independently, then outputs are merged.

**Relevance to us: conceptual, not architectural.** The general principle — *decompose a
hard global problem into easier local sub-problems before the learned model sees it* — is
the same principle behind our dual-pathway split (raster + vector) and behind the
`ReasoningSelector` staging idea. It is also, notably, an argument *for* something we have
not done: our model is asked to detect all 35 classes globally with no region
decomposition. Whether that maps to floor plans (e.g. detect room boundaries first, then
objects within) is speculative and not supported by any measurement here.

### 2.3 Their training recipe (for reference; largely N/A)

Five stages: (1) vision encoder training on tens of billions of image-text pairs with MIM +
CLIP objectives **plus knowledge distillation from a larger in-house ViT**; (2) VL
pretraining, then MTP adaptation; (3) SFT; (4) **GRPO RL** with task-aware rewards
(normalized edit distance, CDM, TEDS, field F1) plus structural constraints (repetition
penalty, tag closure, JSON parse validation); (5) deployment.

**Not applicable:** MTP requires an autoregressive decoder (our head is a dense per-pixel
predictor). GRPO requires rollouts and a verifiable reward (we have dense supervision
against exact GT — no reward to discover; the same conclusion as the DeepSeek report §4).
Stage 1's distillation is standard practice, not a transferable trick.

---

## 3. PaddleOCR-VL 0.9B line (2510.14528v4, 2601.21957v2, 2606.03264v1)

### 3.1 Core claims

- **v1 (2510.14528v4):** a 0.9B VLM (NaViT-style native-resolution encoder + ERNIE-4.5-0.3B
  decoder, LLaVA-style 2-layer MLP projector, **no cross-attention**) beats 72B–241B
  general VLMs on document parsing. OmniDocBench v1.5 **92.86**. Uses **`merge size = 2`**
  visual-token reduction (N → N/2 tokens).
- **v1.5 (2601.21957v2):** adds robustness to physical distortion (Real5-OmniDocBench
  **92.05**, skew 77.47 → 91.66) and a unified RT-DETR + mask + reading-order layout
  engine. Adds **1001 dedicated coordinate tokens** `<LOC_0>…<LOC_1000>` for text spotting.
  OmniDocBench v1.5 → **94.50**.
- **v1.6 (2606.03264v1):** architecture **frozen at 0.9B**; the contribution is a
  **region-aware data engine**. OmniDocBench v1.6 **96.33** (SOTA), zero parameter increase.

### 3.2 The one transferable idea: three-expert consensus label verification

PP-OCRv6's and PaddleOCR's shared discipline, stated most concretely in v1.6: verify labels
with **deterministic rules and cross-model agreement**, not human consensus. Their scheme:
three independent expert models; if **≥1 agrees** with the original label → retain; if **all
disagree but ≥2 experts agree with each other** → take the consensus; otherwise → flag for
refinement.

**Directly applicable to our confirmed defect.** We have (a) an 11% byte-identical
train/val leak being fixed, and (b) *contradictory labels on identical images*
(`train_set_1/0263-0005` vs `0326-0005`: same pixels, one labelled `escalator` the other
`door_double` at the **same bbox**). Two independent detectors (our CenterNet + a
Faster R-CNN baseline) can play the expert role. Cheap, and it would catch mislabelled thin
boxes as a by-product.

### 3.3 The idea that is directionally WRONG for us

**`merge size = 2` visual-token reduction.** This superficially looks relevant to our 1,024
SVG stroke tokens. It is not, and it would hurt. It is a 2×2 spatial average whose purpose
is cutting **autoregressive decode** cost. Our stroke tokens are a *regression signal*, not
a decode cost, and our measured bottleneck is sub-cell detail — precisely what a 2×2 average
destroys. **Do not merge stroke tokens.**

Same class of error: their `<LOC_0>…<LOC_1000>` coordinate tokens exist only because their
output is generated text. A regression head emits continuous coordinates natively; a
1001-token vocabulary would be a large precision and parameter regression.

### 3.4 Their hard-case mining loop — worth noting, already covered

v1's §3.3 pattern: split eval into fine-grained categories, measure per-category, locate
weak categories, **synthesise more data for exactly those categories with deterministic
ground truth**. Their synthesis used fonts/CSS/LaTeX because text is renderable.

**Ours is better**: we hold ground-truth geometry as SVG primitives, so we can procedurally
recombine and perturb strokes to generate new floor plans with **exact box labels at zero
annotation cost**, targeted at stair/ramp. See §5.1.

### 3.5 Their UOR (Under-Optimized Region) mining — a data engine, not a mechanism

v1.6's contribution is worth describing accurately because the name invites a
mis-transfer. Three region types, each terminating in **retrieve-more-data or relabel**:
**boundary-fragile** (8 late checkpoints × 16 perturbations = 128 predictions → 8,128
pairwise distances → top 1% flagged), **coverage-sparse** (progressive cosine-similarity
graph splitting, keeping small outlier components), and **unreliable-supervision** (the
three-expert scheme above). Their own finding is load-bearing and deflating: **">10
augmentation operations during training could not fix"** the boundary-fragile samples — it
is an unstable local mapping, not an augmentation gap.

There is **no loss reweighting, no sampler, no training-loop change** here. It is a data
acquisition engine. Their stage ablation also puts it in proportion: 94.93 → CPT **+0.69** →
SFT **+0.63** → RL **+0.08**. The RL stage — the most machinery-heavy part — is worth +0.08.

### 3.6 Corroborating datum

Their instability-mining flagged **CAD drawings** as a non-conventional layout that standard
document models fail on (alongside comics and high-aspect-ratio screenshots). Independent
confirmation that our domain is genuinely out-of-distribution for general-purpose models —
consistent with §1.5's small-specialised-model thesis.

---

## 4. PaddleOCR 3.0 (2507.05595v1) — reference only

Skimmed for pipeline detail. Two items worth recording as citation-chase targets, no
numbers given in the text:

- **PP-OCRv5 detection** uses backbone **PP-HGNetV2**, with **knowledge distillation from
  the GOT-OCR2.0 visual encoder** by aligning feature representations into the small
  student. A validated industrial recipe for our exact shape of problem (small CNN vs large
  model, detection task). Retains **PFHead** (Parallel Fusion Head) and **DSR** (Dynamic
  Scale-aware Refinement) — DSR is the nearest structural analogue to our hybrid stride-4
  head, PFHead to our dual-pathway fusion. **Neither has mechanism detail in this report.**
- **PP-HGNetV2** is a lightweight CNN backbone family designed for edge deployment. If our
  stride-8 trunk is hand-rolled, it is a legitimate reference point. Caveat: a backbone swap
  changes the parameter count and this report gives no detection numbers. Investigate, do
  not adopt blind.

---

## 5. Recommended experiments

Ranked by expected value. All are data/training-side; none changes the architecture, so all
can be measured against a stable baseline.

### 5.1 ★★ Multi-scale training (highest value)

**Gap:** verified — `PairedTrainTransform` has flip/rotate-90/photometric only, **no scale
augmentation**. Every image is a fixed 512×512 square resize.

**Proposal:** add scale randomisation to the paired transform (scale image and boxes
together), e.g. resample the drawing at 0.7×–1.4× before the final square resize. Rationale
is PP-OCRv6's resolution-robustness result (CV 5.19% vs 8.02%; >81% held at 2.83×) plus the
observation that scale diversity is the standard remedy for small-object failure.

**Why it fits us specifically:** our bottleneck is objects whose apparent size falls below
one stride-8 cell. Scale augmentation changes apparent size, so a given stair is sometimes
above and sometimes below one cell — which is precisely the supervision signal that teaches
the model to use sub-cell cues.

**Cost:** localised to `src/data/transforms.py`, no parameter change. **Confidence:**
medium-high — it is the strongest mechanism in the survey and the gap is verified.
**Metric:** class-stratified AP on stair/ramp, plus overall.

### 5.2 ★ Auxiliary deep supervision at intermediate strides

**Gap:** verified — the hybrid stride-4 head exists (`head_stride=4`, features upsampled
128×128) but only the final output is supervised.

**Proposal:** auxiliary CenterNet heads at stride 8 and stride 16 with the same targets,
weighted ~0.4 / 0.3 / 0.2 following PP-OCRv6's P2/P3/P4 scheme, **dropped at inference**.
Their ablation: +0.53 Hmean, zero inference cost, and it fixes the "gradients must traverse
the whole FPN to reach the backbone" problem.

**Prerequisite:** the Phase-2 protocol item from the implementation plan — any new head must
be droppable at inference so compared models have identical inference cost.

**Cost:** moderate (new heads + loss wiring). **Confidence:** medium.

### 5.3 ★ SVG-based targeted synthesis for stair/ramp

**Idea from:** PaddleOCR-VL v1 §3.3 hard-case mining, applied where we have an advantage
they lacked.

**Proposal:** procedurally generate floor plans enriched in stair/ramp geometry by
recombining and perturbing existing stroke primitives, carrying **exact** box labels derived
from the same primitives. No annotation cost.

**Why it is credible here:** `stair` is the repo's most decisive single-class gain when it
works (thin head: 0.104 → 0.304, +0.201, 214 GT, not rare). It is thin (median min-side
≈5.1 px) and has extreme aspect ratio (7.0; ramp 20.0), and 60–100% of its boxes have a side
below one stride-8 cell.

**Caveat, stated honestly:** synthetic plans may not match the real data distribution, and
the `escalator_stair` case (172 px median side — large but zig-zag) suggests part of the
problem is inconsistent annotation, not resolution. Start small and measure on real val.

**Cost:** moderate (generator + metadata). **Confidence:** medium-low on the magnitude,
but it is the only proposed intervention that adds *signal* rather than reshaping existing
signal.

### 5.4 Dice term alongside focal on the heatmap

**Gap:** verified — `src/training/losses.py` has focal, masked smooth-L1, and
`centernet_loss`; **no Dice**.

**Proposal:** add a Dice term to the heatmap loss. PP-OCRv6's largest loss-side gain
(+1.15) came from pairing Focal with Dice; we have only the Focal half.

**Caveat:** their maps are dense regions, ours are sparse Gaussian peaks with a known
radius-0 pathology (33.6% of boxes at stride 8). **Medium confidence — single A/B, expect
it may be a null.**

### 5.5 Three-expert consensus label verification

**Proposal:** two independent detectors as experts; if ≥1 agrees with the label keep it, if
all disagree but the experts agree with each other take the consensus, else flag. Complements
the Phase-1 dedup: dedup removes *duplicated* images, this catches *contradictory labels on
identical images* (the `0263-0005` / `0326-0005` case).

**Cost:** low. **Confidence:** high as a diagnostic; it is a correctness check, not a bet.

---

## 6. Explicitly rejected

| Idea | Source | Why not |
|---|---|---|
| `merge size=2` visual-token reduction | PaddleOCR-VL v1 | 2×2 spatial average to cut decode cost. Our strokes are a regression signal; the average destroys the sub-cell detail we already fail on. **Directionally harmful.** |
| `<LOC_0>…<LOC_1000>` coordinate tokens | PaddleOCR-VL v1.5 | Exists because their output is generated text. A regression head emits continuous coordinates natively; a 1001-token vocabulary is a precision and parameter regression. |
| Multi-Token Prediction | GLM-OCR | Needs an autoregressive decoder. Our head is a dense per-pixel predictor. (V3 reached the same verdict in the DeepSeek survey.) |
| GRPO / RL stages | GLM-OCR, PaddleOCR-VL v1.5/1.6 | Needs rollouts + verifiable reward. We are densely supervised against exact GT — no reward to discover. By their own ablation, RL bought **+0.08**. |
| NaViT native-resolution encoder | PaddleOCR-VL | A ViT patching/packing scheme; our trunk is a CNN on fixed 512×512. And our SVG branch **already is** the resolution-free representation, better than theirs. |
| Distortion-aware augmentation | PaddleOCR-VL v1.5 | Their augmentation works because *text reading* is invariant to skew/warp. Our task is box regression on an already-rectified CAD raster; warping either corrupts labels or teaches a transform we do not need. |
| 4-point / oriented boxes | PaddleOCR-VL v1.5 | The observation (axis-aligned rects cannot tightly enclose inclined shapes) is real for stairs/diagonals, but adopting oriented boxes invalidates the AP50 0.602 comparison and blows the 6.79M budget. |
| UACS / uncertainty-weighted cluster sampling | PaddleOCR-VL v1.5 | "Stochastic decoding divergence" has no detector analogue, and it targets a 5.6M-sample budget. The intent (up-weight hard classes) is three lines of class-balanced sampling. |
| Anti-symmetric reading-order matrix + voting | PaddleOCR-VL v1.5 | Solves a permutation problem. We have no ordering target. |
| Two-stage layout-then-recognise pipeline | GLM-OCR | Conceptually aligned with our dual-pathway decomposition but no measured analogue for floor plans. Speculative; not scheduled. |

---

## 7. Cross-survey synthesis (with the DeepSeek report)

Both surveys independently converge on the same strategic picture:

1. **Small specialised models beat large general models at localization.** PP-OCRv6
   (+39.4 Hmean over Gemini-3.1-Pro with 15.5M detection params), GLM-OCR (0.9B beating
   much larger models), PaddleOCR-VL (0.9B beating 235B Qwen3-VL on text spotting 87.47 vs
   66.82). The stated reason is consistently **data efficiency**: "for compact models final
   performance is more sensitive to data efficiency and distributional balance."
2. **Data > architecture at our scale.** PP-OCRv6's single largest ablation step was
   *full data* (+2.69 of +6.28). PaddleOCR-VL v1.6 gained +1.32 with the architecture
   frozen. Every DeepSeek paper that scaled needed order-of-magnitude more compute than we
   have.
3. **Nothing in 20 papers fixes sub-stride thin objects by architecture.** PP-OCRv6 comes
   closest and does it by *(a)* supervising at stride 4, *(b)* training across scales, and
   *(c)* adding a per-pixel loss term — i.e. **more supervision signal at the fine stride
   and scale diversity**, not a cleverer head. This corroborates our hybrid stride-4 head
   and points the remaining work at data and supervision.

**One caution carried over from the DeepSeek report, restated because it applies here too:**
the `min_overlap` episode in `experiment_log.md` showed a +5.9 AP50 five-epoch gain shrink
to +0.8 at 30 epochs. Every experiment above must be measured at full budget, and every
result must be judged on the re-split manifest — nothing measured before the dedup fix is
interpretable.

---

## 8. Provenance

**Survey date:** 2026-09-14 (extraction + review). Written same day.
**Method:** `pdftotext` extraction to `.colab/glm_txt/` and `.colab/paddle_txt/`; 6 papers
read in full across two parallel review passes, each asked to be explicitly skeptical about
OCR/VLM → small-detector transfer.

| arXiv / doc | Short name | Date |
|---|---|---|
| 2603.10910v2 | GLM-OCR Technical Report | 2026-03 |
| 2507.05595v1 | PaddleOCR 3.0 | 2025-07 |
| 2510.14528v4 | PaddleOCR-VL 0.9B | 2025-10 |
| 2601.21957v2 | PaddleOCR-VL-1.5 | 2026-01 |
| 2606.03264v1 | PaddleOCR-VL-1.6 | 2026-06 |
| 2606.13108v1 | PP-OCRv6 | 2026-06 |

**Code sites verified for this report** (claims marked "verified" were checked here):
- `src/data/transforms.py` — `PairedTrainTransform`: flip h(0.5)/v(0.0), rotate 90°(0.5),
  photometric. **No scale augmentation.**
- `src/training/losses.py` — `focal_loss`, `l1_loss_masked`, `centernet_loss`. **No Dice.**
- `src/models/detector.py` — `_resolve_head_features` (stride-8 → stride-4 bilinear
  upsample for the head), `_decode_raw`, `encode_vector`. **No auxiliary heads.**

**Caveats on this report's numbers:**
- PP-OCRv6 numbers are from its Tables 13 / Table 5 / Table 1 as extracted; the PDF text
  extraction occasionally scrambles table columns (noted where it happened). The ablation
  ordering and magnitudes were consistent across passes.
- GLM-OCR's Figure 1 benchmark values were unreadable in extraction (column interleaving);
  only the abstract-level numbers (OmniDocBench v1.5 94.6) are quoted.
- The transfer verdicts are judgement calls from two reading passes, not measurements.
  §5.1's gap (no scale augmentation) is the only *verified defect* among them; the others
  are proposals.

**Change log:**

| Date | Change |
|---|---|
| 2026-09-14 | Extracted 6 PDFs; two parallel review passes; verified the three code claims; wrote this report |
| 2026-09-14 | Ranked §5; §7 cross-synthesis added against the DeepSeek survey |
