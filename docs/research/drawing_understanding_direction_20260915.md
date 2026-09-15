# Đánh giá hiện trạng và hướng nghiên cứu đọc bản vẽ

**Ngày chốt:** 2026-09-15

**Phạm vi:** FloorPlanCAD hiện tại, scaffold ArchCAD-400K, toàn bộ 25 PDF trong `docs/research/` (14 DeepSeek, 6 OCR/Document AI, 5 grounding/perception), và các công trình floor-plan liên quan được kiểm tra tới ngày chốt.

## 1. Kết luận điều hành

Đề tài hiện tại đã tạo được một baseline **phát hiện ký hiệu có điều kiện theo lớp** từ ảnh raster và primitive vector. Nó chưa phải hệ thống **đọc bản vẽ**. Khoảng trống chính không nằm ở AP hay số epoch, mà ở đầu ra: mô hình hiện chỉ trả box và confidence, trong khi đọc bản vẽ cần một biểu diễn có thể truy vấn và kiểm chứng gồm hình học, ngữ nghĩa, topology và văn bản.

Quyết định hướng đi:

1. Giữ detector hiện tại làm baseline perception và một head phụ.
2. Tạm dừng tối ưu scaffold ArchCAD 21 lớp vì nó loại bỏ `wall`, `beam`, `foundation`, `axis_and_grid`, `glass`, `rebar`—những thành phần quan trọng để hiểu cấu trúc.
3. Xác định đích mới là **Canonical Drawing Graph (CDG)**, rồi đánh giá toàn pipeline từ bản vẽ đến graph và câu trả lời thực thi được.
4. Dùng ArchCAD trước hết để dựng ground truth cấu trúc từ CAD semantics/geometry; chỉ huấn luyện sau khi chứng minh graph và bộ câu hỏi có thể được tạo đúng bằng quy tắc trên annotation thật.
5. Tách perception chính xác khỏi reasoning ngôn ngữ: parser hình học tạo graph/proposal; MLLM/LLM đóng băng hoặc fine-tune nhẹ chỉ suy luận trên graph/proposal. Không dùng sinh tọa độ tự hồi quy làm đường chính ở giai đoạn đầu.

Luồng mục tiêu:

`PNG/SVG/CAD → geometry + symbols + text → canonical graph → query executor / grounded reasoner → answer + evidence`

### Trạng thái triển khai từ 2026-09-15

- Hướng này đã trở thành hướng nghiên cứu chính trong root `README.md` và `docs/research/README.md`.
- Đã triển khai CDG-v0 schema, JSON round-trip, validator, ArchCAD compiler và query executor tại `src/drawing_graph/`.
- Oracle audit seed 1337 trên 100 chunk thật đạt 100/100 graph hợp lệ và 966/966 contract checks đúng với raw annotation/compiler contract.
- Compiler bảo toàn toàn bộ semantic primitives thay vì lọc xuống 21 detector classes.
- CDG-v0 đã thêm enclosed-wall-face và door–wall attachment candidates. Overlay audit cho thấy enclosed face có thể là shaft/core, nên chưa được gọi là room ground truth.
- Endpoint-cycle baseline chỉ tìm được face ở 28/100 mẫu và connectivity candidate ở 2/100 mẫu. Nó không qua cổng coverage để trở thành room oracle; cần architectural subset và opening-closure/raster-assisted extraction.
- Task manifest toàn bộ 41.097 chunk xác định 14.170 architectural-reading candidates (34,48%) và 9.992 floor-plan-reading candidates (24,31%). Audit 100 mẫu floor-plan đạt 72% face coverage và 16% connectivity-candidate coverage, nhưng manual overlay vẫn phát hiện wall-thickness strips bị nhận như region.
- Mọi quan hệ topology suy diễn đã được chuyển sang namespace `candidate_*`; không dùng làm room/connection ground truth cho đến khi qua semantic validator.
- Opening-closure + raster free-space high-precision chỉ phủ 33/100 mẫu và không tạo `occupied_space` candidate nào; automatic room oracle từ ArchCAD semantics local bị đánh dấu **no-go**.
- Đã tạo pack 200 record để annotation thủ công, 70 record có candidate hint. Validator xác nhận format hợp lệ nhưng cổng 100 reviewed floor plans đang ở 0; pack chưa phải evaluation split hay ground truth.
- Đã sinh 1.000 executable QA trên 100 floor-plan candidates: 600 raw-semantic oracle và 400 câu hỏi về heuristic candidates; mọi answer kèm evidence. Đây là feasibility set, chưa phải publication benchmark.
- Audit toàn bộ 205.479 caption QA của release: chỉ 217 câu (0,106%) có dấu hiệu relation/room reasoning, 6 record lỗi kiểu dữ liệu; caption chủ yếu đo count/existence/localization ký hiệu.
- Provenance audit xác nhận năm archive public cùng dùng 41.097 UUID phẳng và JSON sample chỉ chứa `entities`. Paper có drawing-disjoint split gốc, nhưng public 40K release/repo chưa kèm mapping hoặc CADParser, nên không thể tái dựng split đó từ gói local.
- Cổng source-disjoint split chưa qua: cả 100 graph đều thiếu `source_drawing_id`; kết quả hiện tại chỉ là compiler feasibility, chưa phải benchmark generalization.

## 2. “Đọc bản vẽ” được định nghĩa thế nào

Một hệ thống chỉ được coi là đọc được bản vẽ khi nó có thể trả lời đúng, có dẫn chứng hình học, ít nhất bốn tầng sau:

| Tầng | Đầu ra tối thiểu | Ví dụ câu hỏi |
|---|---|---|
| Hình học | wall/beam/axis/opening, junction, polygon/phòng | “Biên phòng này nằm ở đâu?” |
| Ngữ nghĩa | loại phòng, cửa, cột, thiết bị, text, kích thước | “Phòng nào là WC?” |
| Topology | `contains`, `adjacent_to`, `connected_by`, `part_of`, `aligned_with` | “Từ hành lang vào phòng ngủ qua cửa nào?” |
| Suy luận | truy vấn tổng hợp, kiểm tra ràng buộc, xuất dữ liệu | “Có bao nhiêu phòng thoát ra hành lang?” |

Đầu ra box theo lớp chỉ bao phủ một phần nhỏ của tầng ngữ nghĩa. AP50 không đo được phòng có kín hay không, cửa nối đúng hai vùng hay không, text có gắn đúng đối tượng hay không, hoặc câu trả lời có nhất quán với hình học hay không.

## 3. Đánh giá nghiên cứu hiện tại

### 3.1 Những gì đã có bằng chứng

| Hạng mục | Bằng chứng hiện có | Đánh giá |
|---|---|---|
| Detector ảnh + vector | Best run đã ghi nhận: AP50 `0.4950`, AP50:95 `0.3091`, seed 1337, 30 epoch | Có giá trị làm baseline perception |
| Vector pathway | Các ablation cùng protocol cho thấy fusion raster/vector là hướng có ích | Nên giữ, nhưng phải xác nhận lại trên split sạch |
| Multi-scale augmentation | Code và test cho crop/pad đồng bộ ảnh, box, stroke đã hoàn thiện | Hạ tầng tốt cho robustness, chưa chứng minh đọc bản vẽ |
| ArchCAD local | 41,097 ảnh, 380,462 instance đếm được; raster và primitive phong phú | Nguồn dữ liệu mạnh, nhưng scaffold hiện thu hẹp bài toán quá mức |
| Tiny-object diagnosis | `other_door`: median cạnh nhỏ 13 px, 29.62% dưới stride 8 | Hợp lý để thử local high-resolution có chọn lọc |

### 3.2 Những kết quả chưa đủ hiệu lực

- Headline `0.4950 / 0.3091` dùng manifest v2 có duplicate leakage; không được dùng làm số liệu kết luận tổng quát.
- Manifest v3 đã loại trùng giữa train và validation: train 7,536; val-select 1,289; val-calibration 1,336. Tuy nhiên checkpoint v2 đánh giá lại trên v3 vẫn không phải clean baseline vì 350/1,289 ảnh val-select v3 từng nằm trong train v2.
- Test cố định còn 610/5,502 ảnh trùng nội dung với training pool. Cần một test sạch bổ sung nếu muốn công bố khả năng tổng quát hóa.
- Tất cả top run mới dừng ở 30 epoch; đường cong subset còn tăng nhưng chưa có full-role measurement sau epoch 30. Đây là câu hỏi tối ưu detector còn mở, không thay đổi khoảng trống về task definition.
- Split ArchCAD local đang nhóm theo hash ảnh. Dataset gốc có 413,062 chunk sinh từ 5,538 bản vẽ; nếu không giữ provenance bản vẽ gốc, các chunk lân cận của cùng một bản vẽ có thể rơi vào nhiều split. Cần split theo `source_drawing_id`/project trước khi báo cáo benchmark.

### 3.3 Khoảng trống giữa tên gọi và code

- `FloorPlanLLM` hiện chỉ dùng `ParallelBoxDecoder` và trả `boxes`, `confidence`; docstring ghi scene-graph decoder và relationship head là “Future extensions”.
- `ReasoningSelector` chỉ là threshold confidence + khoảng cách giữa box, không có tham số học và không thực hiện reasoning.
- Decoder một box cho mỗi class không biểu diễn được số lượng instance biến thiên; đây là thiết kế không phù hợp với bản vẽ dày đặc có nhiều đối tượng cùng lớp.
- Không có ground truth, loss, decoder hay metric cho room polygon, wall graph, containment, adjacency, connectivity, OCR grounding hoặc dimension.

Vì vậy mức trưởng thành đúng là: **symbol detection research prototype**, chưa phải **drawing understanding system**.

## 4. Bài học từ toàn bộ kho tài liệu tham khảo

### 4.1 Kiểm kê và mức liên quan

| Nhóm local | Số bài | Vai trò đối với mục tiêu đọc bản vẽ | Mức ưu tiên |
|---|---:|---|---|
| `Resource/`: Rex-Omni, LocateAnything, FLIP, PerceptionDLM, VGent | 5 | Grounding, local perception, parallel region processing, tách reasoning khỏi localization | Cao cho kiến trúc tổng thể |
| `paddleocr/` + `glm/`: PaddleOCR 3.0, PaddleOCR-VL, VL-1.5, VL-1.6, PP-OCRv6, GLM-OCR | 6 | Text detection/recognition, document decomposition, hard-region mining | Cao cho text/dimension/legend head |
| `deepseek/`: LLM, MoE, Coder, Math, V2/V3/V3.2/V4/V4.1, R1, Prover, OCR | 14 | Data/protocol discipline, sequence ordering, reasoning layer; phần lớn cơ chế scale không áp dụng cho detector | Thấp–trung bình, có chọn lọc |

Không nên đếm số paper như số “ý tưởng phải đưa vào model”. Chỉ các cơ chế giải quyết đúng bottleneck và có thể kiểm tra bằng ablation mới được chuyển giao.

### 4.2 Grounding và region perception

| Paper | Ý tưởng nên lấy | Điều không nên sao chép trực tiếp |
|---|---|---|
| Detect Anything / Rex-Omni | Task interface thống nhất; geometry-aware reward; point/box grounding | Sinh tọa độ token có rủi ro drift, duplicate và recall; không nên thay detector hình học ngay |
| LocateAnything | Parallel geometric decoding; gom nhiều grounding task | One-query/one-box semantics không tự giải quyết multi-instance và topology |
| Looking Locally / FLIP | Local region đa độ phân giải, giữ chi tiết ở vùng quan trọng | Không cần tăng độ phân giải toàn ảnh; chỉ crop theo proposal/uncertainty |
| PerceptionDLM | Xử lý nhiều region song song sau khi đã có region | Không dùng làm primary geometry parser; paper cũng còn yếu với chữ nhỏ/OCR |
| VGent | Tách frozen MLLM reasoning khỏi proposal localization; decoder chọn proposal | Đây là mẫu kiến trúc phù hợp nhất cho reasoning có grounding; tránh joint training toàn bộ từ đầu |

### 4.3 OCR và Document AI

| Kết luận từ 6 paper | Áp dụng cho drawing understanding |
|---|---|
| Model detection nhỏ, chuyên biệt có thể tốt hơn VLM rất lớn trong localization | Dùng OCR detector/recognizer chuyên biệt cho text nhỏ; không bắt MLLM đọc trực tiếp toàn bản vẽ |
| GLM-OCR tách layout/region trước rồi mới recognize | Tách text-region detection, recognition và semantic linking thành các bước đo độc lập |
| PaddleOCR dùng hard-case/UOR mining và kiểm tra nhãn nhiều nguồn | Tạo vòng lặp mine text nhỏ, dimension, rotated label, legend mismatch; audit trước khi retrain |
| PP-OCRv6 nhấn mạnh multi-scale và intermediate supervision | Thử auxiliary supervision ở các stride cho text/thin symbols; bỏ auxiliary heads khi inference |
| Coordinate token và visual-token merging phục vụ autoregressive document parsing | Không chuyển sang geometry parser: token tọa độ giảm độ chính xác và merging làm mất primitive mảnh |

OCR không chỉ tạo chuỗi ký tự. Mỗi text node phải có polygon/baseline, nội dung, loại (`room_label`, `dimension`, `level`, `legend`) và relation `refers_to` tới geometry/symbol tương ứng. Numeric dimension phải được parse thành value + unit và kiểm tra lại bằng scale/hình học khi có thể.

### 4.4 DeepSeek: chuyển giao có chọn lọc

Ba bài học dùng được:

1. **Data integrity trước architecture:** chính việc kiểm tra theo tư duy data-filtering đã lộ duplicate leakage và label inconsistency của FloorPlanCAD. Đây là lợi ích thực nghiệm lớn nhất từ nhóm paper này.
2. **Stroke ordering/connectivity:** primitive vector hiện mang thứ tự tài liệu và nhiều endpoint liên tiếp nối chuỗi, nhưng encoder không có positional/adjacency signal. Nên thử order embedding hoặc sparse endpoint graph trong structure head.
3. **Ổn định attention nhỏ:** q/k normalization hoặc attention sink chỉ đáng thử nếu đo được instability/attention collapse; không đưa vào mặc định vì chưa có ablation trên bản vẽ.

Các cơ chế không phù hợp với perception parser hiện tại: MoE routing, MLA/KV-cache compression, million-token sparse attention, GRPO/PPO, math verifier và code-specific pretraining. Chúng tối ưu language-model scale hoặc autoregressive decoding—chi phí mà detector/graph parser không có. DeepSeek-R1/Math/Prover chỉ có thể được xem lại ở reasoning layer khi đã có executable graph reward; không dùng để thay exact geometry supervision.

### 4.5 Công trình ngoài repo định hướng lại bài toán

- ArchCAD-400K cung cấp line-grained annotation và DPSS fusion ảnh–primitive, nhưng task gốc vẫn là panoptic symbol spotting. Nó là nền perception, không phải benchmark đọc toàn bộ bản vẽ: <https://arxiv.org/abs/2503.22346>.
- HEAT dự đoán corner và cạnh để tái dựng planar graph; đây là baseline đúng loại đầu ra cho wall/room topology: <https://openaccess.thecvf.com/content/CVPR2022/html/Chen_HEAT_Holistic_Edge_Attention_Transformer_for_Structured_Reconstruction_CVPR_2022_paper.html>.
- RoomFormer biểu diễn đầu ra biến thiên bằng hai cấp query: tập room polygon và chuỗi corner của từng room. Đây là tham chiếu tốt cho structured decoder: <https://openaccess.thecvf.com/content/CVPR2023/papers/Yue_Connecting_the_Dots_Floorplan_Reconstruction_Using_Two-Level_Queries_CVPR_2023_paper.pdf>.
- WAFFLE có gần 20K floor plan “in the wild” cùng metadata đa phương thức; phù hợp để đo OCR, legend và domain robustness hơn là làm nguồn hình học CAD chính xác: <https://arxiv.org/abs/2412.00955>.
- HouseMind dùng discrete room-instance tokens để nối geometry, semantics và spatial hierarchy với MLLM. Đây là bằng chứng mới nhất ủng hộ việc token hóa graph/room instance, không token hóa mọi pixel: <https://arxiv.org/abs/2603.11640>.
- VLM thuần có thể làm map parsing và navigation planning nhưng giảm hiệu năng ở bản đồ lớn/open space. Nên dùng làm reasoning baseline, không dùng làm nguồn ground truth hình học: <https://arxiv.org/abs/2409.12842>.

## 5. Canonical Drawing Graph

Schema v0 nên nhỏ nhưng đủ thực thi:

```text
Drawing
├── GeometryNode: junction | segment | wall | beam | axis | region
├── SemanticNode: room | opening | column | fixture | furniture | annotation
├── TextNode: label | dimension | level | code | legend_item
└── RelationEdge:
    ├── contains / inside
    ├── adjacent_to
    ├── connected_by
    ├── intersects / aligned_with
    ├── part_of
    └── refers_to
```

Mỗi node cần `id`, `type`, geometry chuẩn hóa, confidence, provenance về primitive/pixel nguồn. Mỗi answer phải trả cả giá trị và danh sách node/edge làm evidence. Graph phải có validator với các ràng buộc như polygon kín, door chạm wall/opening, `contains` tương thích hình học và relation hai chiều nhất quán.

## 6. Kiến trúc đề xuất

### 6.1 Perception compiler

`Raster encoder + primitive/vector encoder → shared multiscale features`

Ba head song song:

1. **Structure head:** junction, segment, wall/beam/axis, room/region polygon.
2. **Symbol head:** cửa, cột, thiết bị, nội thất; tận dụng detector hiện tại.
3. **Text head:** text-region detector nhỏ + recognizer chuyên biệt, dimension endpoints, legend-item linkage. Layout/region phải được phát hiện trước recognition và có metric riêng.

Sau đó dùng geometry compiler quyết định để snap endpoint, polygonize region, gắn opening vào wall và tạo candidate relations. Learned relation head chỉ xử lý phần mơ hồ; không bắt mạng học lại các quy tắc hình học chắc chắn.

Vector encoder cần thêm một ablation về cấu trúc chuỗi: (a) không order signal, (b) document-order embedding, (c) endpoint-adjacency sparse graph. Đây là vấn đề trực tiếp hơn MoE hay long-context compression vì nó khôi phục connectivity vốn có trong SVG/CAD.

### 6.2 Grounded reasoner

`CDG + visual proposals + user query → frozen MLLM/LLM adapter → selected nodes/edges → deterministic executor → answer`

Áp dụng nguyên tắc VGent: reasoning model cung cấp semantic reasoning, còn tọa độ/đối tượng được chọn từ proposal hoặc graph đã có. Fine-tune adapter/selector trước; chỉ cân nhắc joint training khi ablation chứng minh cần thiết.

### 6.3 Local refinement có điều kiện

FLIP-style crop chỉ kích hoạt cho chữ nhỏ, symbol nhỏ hoặc vùng uncertainty cao. Với ArchCAD hiện tại, `other_door` là ứng viên rõ nhất. Không dùng global 2×/4× resolution như mặc định vì chi phí tăng mạnh mà collision stride 8 → 4 chỉ giảm 2.87%.

## 7. Benchmark mới

Tách hai scoreboard để không che giấu lỗi:

| Nhóm | Task | Metric chính |
|---|---|---|
| Perception | symbol instance/panoptic | AP50:95, PQ, per-class recall |
| Structure | corner/edge/wall graph | node F1, edge F1, topology validity |
| Region | room polygon/type | polygon IoU, room F1, closed-region rate |
| Text | OCR + grounding + dimension | CER/WER, grounding IoU, numeric accuracy |
| Relation | contains/adjacent/connected-by | macro-F1, graph consistency |
| Reading | executable QA | exact match, execution consistency, evidence F1 |
| Robustness | CAD → FloorPlanCAD/WAFFLE/scan | domain-gap delta, calibration, abstention |

QA phải sinh từ graph bằng chương trình để đáp án chính xác và tái lập được. Nhóm câu hỏi ban đầu: count, locate, containment, adjacency, door connectivity, path, dimension và constraint violation. Cần split template/paraphrase để tránh học thuộc câu chữ.

## 8. Roadmap và cổng quyết định

### P0 — Đóng task/schema/protocol (1 tuần)

- Chốt CDG v0, validator và 30–50 query primitives.
- Xác định đơn vị split là source drawing/project, không phải chunk.
- Đóng băng test và data lineage.

**Go:** 100% query có executable oracle trên graph mẫu.

**No-go:** không truy hồi được provenance bản vẽ gốc cho ArchCAD local; khi đó chỉ dùng local release cho prototyping, không báo cáo generalization.

### P1 — Oracle graph trước ML (1–2 tuần)

- Chọn 100–300 ArchCAD chunk có wall/door/text đa dạng.
- Dựng deterministic compiler từ semantic primitives thành wall/opening/region graph.
- Sinh QA và render overlay để review thủ công.

**Go:** ≥95% graph qua validator; ≥98% answer oracle đúng trên mẫu audit.

**No-go:** semantics hiện có không đủ dựng room/connection; cần bổ sung CAD layer/provenance hoặc thu annotation nhỏ trước khi train.

### P2 — Baseline cấu trúc (2–4 tuần)

- So sánh deterministic geometry, HEAT-style graph và RoomFormer-style polygon decoder.
- Giữ detector hiện tại làm symbol head baseline.
- Cold-train trên split sạch; không reuse checkpoint nhiễm leakage để báo số chính.

**Go:** graph edge F1 và room F1 vượt heuristic với topology-validity không giảm.

**No-go:** nếu model learned không vượt oracle/heuristic đơn giản, tập trung sửa target/annotation thay vì scale model.

### P3 — Multitask drawing parser (3–5 tuần)

- Fusion raster/vector chung, ba head structure/symbol/text.
- Text branch theo pipeline detect → recognize → link; mine riêng text nhỏ, dimension và legend khó.
- Relation candidates từ geometry compiler; train relation head trên ambiguity.
- Ablation bắt buộc: raster-only, vector-only, late fusion, shared fusion; vector no-order vs order embedding vs endpoint graph.

**Go:** tăng reading metrics mà không làm giảm nghiêm trọng structural validity.

**No-go:** nếu loss cạnh tranh, tách encoder/head hoặc train curriculum thay vì tăng trọng số tùy ý.

### P4 — Grounded reasoning (2–4 tuần)

- Graph-to-text serialization hoặc room-instance tokens.
- Frozen MLLM/LLM + proposal/graph selector; answer qua executor.
- So sánh với VLM nhìn ảnh trực tiếp.

**Go:** executable QA và evidence F1 cao hơn direct-VLM, hallucination thấp hơn.

**No-go:** nếu reasoner không hơn graph executor, giữ natural-language layer mỏng và ưu tiên perception.

### P5 — Cross-domain và release

- FloorPlanCAD: symbol/generalization.
- WAFFLE/scanned drawings: OCR, legend, noise.
- Báo cả performance, calibration, failure taxonomy và latency.

## 9. Ba thí nghiệm nên làm ngay

1. **Oracle-CDG feasibility:** đã đạt 100 ArchCAD chunk → graph → 1.000 executable QA, nhưng 400 query topology vẫn chỉ là candidate. Bước kế tiếp là thay chúng bằng human-gold room/connectivity queries.
2. **Source leakage audit:** tìm hoặc phục hồi `source_drawing_id`; đo duplicate/near-neighbor giữa split ở cấp chunk và bản vẽ.
3. **Structure baseline bake-off:** deterministic compiler vs HEAT-style edge model vs RoomFormer-style polygon model trên cùng subset và cùng graph metrics.

Chưa nên làm ngay: train MLLM 3B từ đầu, sinh box bằng token, mở rộng detector 21 lớp rồi coi AP là reading score, hoặc joint-train toàn pipeline trước khi có graph oracle.

## 10. Tiêu chí thành công cuối

Mốc cuối không nên là “AP50 vượt X”. Một release đạt mục tiêu khi:

- tạo được CDG hợp lệ và có provenance từ input;
- phát hiện đủ structure, symbol và text để trả lời bộ executable QA;
- mỗi answer có evidence node/edge và qua consistency check;
- split không rò rỉ source drawing/project;
- kết quả giữ được trên ít nhất một miền ngoài CAD chuẩn hóa;
- hệ thống biết abstain khi graph không đủ bằng chứng.

Tóm tắt: **baseline detector → structural parser → canonical graph → grounded reasoner → executable, evidenced answers**.

## Phụ lục A — Danh mục 25 PDF đã rà soát

### Grounding/perception (5)

1. Detect Anything via Next Point Prediction (Rex-Omni).
2. LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding.
3. Looking Locally: Object-Centric Vision Transformers as Foundation Models for Efficient Segmentation.
4. PerceptionDLM: Parallel Region Perception with Multimodal Diffusion Language Models.
5. VGent: Visual Grounding via Modular Design for Disentangling Reasoning and Prediction.

### OCR/Document AI (6)

1. GLM-OCR Technical Report.
2. PaddleOCR 3.0 Technical Report.
3. PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact VLM.
4. PaddleOCR-VL-1.5: Multi-Task Robust In-the-Wild Document Parsing.
5. PaddleOCR-VL-1.6: Under-Optimized Region Refinement and Progressive Post-Training.
6. PP-OCRv6: Small Specialized OCR Models.

### DeepSeek/general reasoning and efficiency (14)

1. DeepSeek LLM.
2. DeepSeekMoE.
3. DeepSeek-Coder.
4. DeepSeekMath.
5. DeepSeek-V2.
6. DeepSeek-Coder-V2.
7. DeepSeek-Prover-V1.5.
8. DeepSeek-V3.
9. DeepSeek-R1.
10. DeepSeek-OCR.
11. DeepSeekMath-V2.
12. DeepSeek-V3.2.
13. DeepSeek-V4.
14. DeepSeek-V4.1-Flash.
