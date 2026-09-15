# Main Research Direction: Drawing Understanding

Đây là entry point chính cho nghiên cứu. Các tài liệu detector cũ là lịch sử/baseline trừ khi được dẫn lại tại đây hoặc trong experiment log có kiểm chứng.

## Mục tiêu

Biến bản vẽ raster/vector/CAD thành **Canonical Drawing Graph (CDG)** có provenance, sau đó trả lời truy vấn bằng graph evidence. Hệ thống phải hiểu structure, symbols, text và relations; không dùng AP box làm đại diện cho toàn bộ khả năng đọc bản vẽ.

## Nguồn sự thật

1. `drawing_understanding_direction_20260915.md` — quyết định kiến trúc, benchmark và roadmap.
2. `../experiments/experiment_log.md` — số liệu detector đã kiểm chứng và hạn chế leakage.
3. `../../outputs/drawing_graph/oracle_archcad_100_report.json` — kết quả oracle graph hiện tại.
4. `../../outputs/drawing_graph/archcad_task_manifest.json` — task selection toàn corpus.
5. `../../outputs/drawing_graph/oracle_archcad_architectural_100_report.json` — audit trên architectural subset.
6. `../../outputs/drawing_graph/oracle_archcad_floor_plan_100_report.json` — audit trên floor-plan candidate subset chặt hơn.
7. `../../outputs/drawing_graph/raster_regions_floor_plan_100_high_precision_report.json` — cổng go/no-go của raster room heuristic.
8. `../../outputs/drawing_graph/manual_overlay_audit.json` — kết luận kiểm tra trực quan.
9. `../../outputs/drawing_graph/room_gold_pack_v0.validation.json` — tính hợp lệ và tiến độ gold annotation.
10. `../../outputs/drawing_graph/oracle_archcad_floor_plan_100_qa.summary.json` — 1.000 executable QA và supervision boundary.
11. `../../outputs/drawing_graph/archcad_caption_audit.json` — audit 205.479 caption QA của release.
12. `../../outputs/drawing_graph/archcad_provenance_audit.json` — audit mapping chunk → source drawing.
13. `../../outputs/drawing_graph/research_gates_v0.json` — trạng thái machine-readable của ba cổng gold/agreement/provenance.
14. `../../src/drawing_graph/` — contract CDG-v0 thực thi được.

## Trạng thái

| Milestone | Trạng thái | Bằng chứng |
|---|---|---|
| Chuyển hướng detector → drawing understanding | Hoàn thành | README và roadmap này |
| CDG-v0 schema + JSON round-trip | Hoàn thành | `schema.py`, unit tests |
| Validator + provenance gate | Hoàn thành v0 | `validation.py` |
| ArchCAD oracle compiler giữ toàn bộ semantics | Hoàn thành v0 | `archcad.py` |
| Evidence-producing query executor | Hoàn thành v0 | `queries.py` |
| Audit 100 chunk | Hoàn thành | 100/100 graph valid; 966/966 contract checks đúng |
| Source-drawing-disjoint split | Blocked bởi dữ liệu nguồn | JSON local không có `source_drawing_id` |
| Enclosed wall-face candidates | Hoàn thành v0 | 0.96 candidate/chunk; chưa được gọi là room |
| Door–wall attachment candidates | Hoàn thành v0 | 1.52 candidate links/chunk, có contact-point evidence |
| Region connectivity candidates | Baseline coverage thấp | 0.02/chunk; chỉ 2/100 mẫu có candidate |
| Raster-assisted room oracle | No-go | Chỉ 33% coverage; phần lớn là wall-body/void/unclassified candidates |
| Room/connectivity gold pack | Sẵn sàng để review | 200 record hợp lệ; 70 có candidate hint, không phải ground truth |
| Blinded double-annotation pack | Sẵn sàng để review | 40 record (20%); annotation chính không được sao chép |
| Agreement evaluator | Hoàn thành v0 | maximum polygon-IoU matching + door-aware relation F1 |
| Local annotation workbench | Hoàn thành v0 | Canvas polygon + door relation + atomic save/backup/conflict check |
| Executable QA feasibility | Hoàn thành v0 | 1.000 query/100 chunk; 600 raw-semantic oracle, 400 heuristic-only |
| Released caption QA audit | Hoàn thành | 205.479 QA; chỉ 217 relation/room-like, 6 record sai kiểu |
| OCR/dimension/legend branch | Chưa bắt đầu | Thực hiện sau structure oracle |
| Learned structural parser | Chưa bắt đầu | Chỉ bắt đầu sau P1 go/no-go |
| Structure bake-off evaluator | Hoàn thành scaffold | CDG region/connectivity metrics; chưa có gold để chạy |

## Kết quả P1 đầu tiên

Audit seed 1337 trên 100 ArchCAD chunk:

- 100% graph qua structural validator.
- 100% trong 966 executable checks khớp raw annotation/compiler contract. Con số này kiểm tra tính bảo toàn/thực thi, không phải độ đúng semantic của room.
- Trung bình 381.57 nodes, 436.28 edges, 128.14 endpoint junctions và 8.78 instances/chunk.
- Trung bình 0.96 enclosed-wall-face candidates, 1.52 door–wall candidate links, 0.09 instance–region candidates và 0.02 region-connectivity candidates/chunk.
- Coverage: 63/100 mẫu có wall, 22/100 có door, 28/100 có enclosed face và 2/100 có connectivity candidate.
- Bảo toàn các lớp trước đây bị detector lọc bỏ, gồm 2,854 wall primitives, 1,178 concrete-beam, 784 axis/grid, 393 rebar và 182 foundation trong mẫu.
- 100/100 graph cảnh báo thiếu source-drawing provenance. Vì vậy kết quả này chứng minh compiler feasibility, chưa chứng minh khả năng generalization.

## Thứ tự công việc bắt buộc

1. Review 200-record room gold pack theo `room_gold_annotation_protocol.md`; không promote candidate hint thành nhãn.
2. Double-annotate 20% và đo polygon IoU/relation agreement.
3. Phục hồi provenance và dựng split theo source drawing/project.
4. Mở rộng query set trên graph gold thành contains, adjacent, connected-by và path.
5. Chỉ train/so sánh learned structural parser khi gold/provenance gates đã qua.

Protocol thực thi nằm tại `room_gold_annotation_protocol.md`; thiết kế so sánh
deterministic/HEAT-style/RoomFormer-style nằm tại `structure_bakeoff_protocol.md`.

Overlay audit đầu tiên cho thấy face kín có thể là lõi thang/shaft thay vì room. Vì vậy CDG-v0 cố ý dùng nhãn `enclosed_wall_face_candidate` với confidence 0.5. Không được đổi thành room ground truth nếu chưa có classification/audit.

Endpoint-cycle v0 là baseline cần giữ, nhưng **không qua cổng room-oracle coverage**: chỉ 28% mẫu ngẫu nhiên có face và 2% có connectivity candidate. Không scale/training trên target này trước khi xử lý task filtering và boundary gaps.

## Task selection toàn ArchCAD local

Manifest fingerprint `2c83d23d7e48ea128d1e99fc5defb3e903e37e6d8747d7bccff3df2341c56670` trên 41.097 chunk:

- 27.184 (66,15%) có wall.
- 25.317 (61,60%) đủ ít nhất bốn wall primitives để thử wall graph.
- 11.194 (27,24%) có wall graph + door evidence.
- 14.170 (34,48%) được chọn làm architectural-reading candidates bằng wall + door/fixture evidence.
- 9.992 (24,31%) qua bộ lọc floor-plan-reading chặt hơn.
- 34.824 (84,74%) chứa structural-engineering semantics; các subset không loại trừ nhau.

Manifest này chỉ chọn task và được đánh dấu `task_selection_only_not_an_evaluation_split`. Nó không giải quyết source-drawing leakage.

Audit thêm 100 mẫu từ architectural subset cho kết quả: 61% có enclosed face, 18% có connectivity candidate, trung bình 3,08 face và 0,63 connectivity candidates/chunk. Trên floor-plan-reading subset, face coverage là 72% và connectivity-candidate coverage là 16%, trung bình 3,33 và 0,63 candidate/chunk. Coverage tăng nhưng không đồng nghĩa precision: overlay vẫn cho thấy dải giữa hai biên tường bị nhận như region. Vì vậy quan hệ dùng `candidate_bounds`, `candidate_inside`, `candidate_attached_to`, `candidate_connected_by`; không consumer nào được coi chúng là quan hệ thật nếu chưa qua semantic validator.

Opening-closure + raster free-space high-precision chỉ tạo candidate trên 33/100 mẫu floor-plan; 65 region tìm được gồm 29 `unclassified`, 29 `void_or_shaft`, 5 `vertical_core`, 2 `service_space` và **0 `occupied_space`**. Đây là no-go rõ ràng cho automatic room oracle từ semantics local. Pack review thủ công 200 record đã được tạo, 70 record có preannotation hints và validator xác nhận 200/200 record hợp lệ; hiện 0/100 reviewed floor-plan gate, nên chưa được dùng làm evaluation/training labels.

Chạy kiểm định gold pack:

```bash
python scripts/research/validate_room_gold_pack.py outputs/drawing_graph/room_gold_pack_v0.jsonl
```

Mở workbench để annotation:

```bash
python scripts/research/serve_room_gold_annotator.py --open-browser
```

## QA hiện có và giới hạn

Bộ executable QA v0 có 1.000 câu trên 100 floor-plan candidates: 600 câu count trên semantic/instance gốc và 400 câu chỉ hỏi về `candidate_*`. Mỗi record lưu program, answer và evidence node/edge. Nó chứng minh đường graph → executor hoạt động, nhưng chưa phải reading benchmark để công bố vì thiếu source provenance và 400 câu topology chưa có human gold.

Release còn có `caption.zip` với 205.479 cặp QA cho 41.097 chunk. Audit toàn bộ tìm thấy 205.473 cặp string/string hợp lệ, 6 record lỗi kiểu, và chỉ 217 câu (0,106%) mang dấu hiệu relation/room reasoning; phần lớn là count, existence hoặc localization theo tọa độ. Do đó caption chỉ là auxiliary symbol QA sau khi được đối chiếu bằng executor, không thay thế CDG/room gold.

Audit provenance xác nhận cả năm archive public chứa cùng 41.097 UUID phẳng; 100/100 JSON kiểm tra chỉ có top-level key `entities`, không có parent drawing/project key. Paper nói split gốc bảo đảm mỗi drawing chỉ thuộc một split, nhưng initial 40K release và code public không kèm manifest tương ứng; README upstream vẫn liệt kê CADParser trong TODO. Vì vậy SHA-256 grouping hiện tại chỉ loại exact duplicates, không đủ thay thế drawing-disjoint split. Nguồn chính thức: [project page](https://archiai-lab.github.io/ArchCAD.github.io/), [repository](https://github.com/ArchiAI-LAB/ArchCAD), [public dataset files](https://huggingface.co/datasets/jackluoluo/ArchCAD/tree/main/data).

Không ưu tiên lúc này: tăng epoch detector 21 lớp, train MLLM từ đầu, coordinate-token box generation, MoE, KV-cache compression hoặc RL reasoning không có graph verifier.
