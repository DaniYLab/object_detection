# Room/Connectivity Gold Annotation Protocol v0

## Mục đích

Tạo một gold set nhỏ để kiểm tra liệu ArchCAD local có hỗ trợ room/connectivity learning hay không. Pack này phục vụ feasibility và error analysis; chưa phải evaluation split vì thiếu `source_drawing_id`.

## Đơn vị annotation

Mỗi record trỏ tới một PNG và raw JSON, kèm raster candidates chỉ để gợi ý. Annotator phải xem ảnh gốc; không được chấp nhận candidate tự động mà không kiểm tra.

### `view_type`

- `floor_plan`: mặt bằng có thể xác định không gian sử dụng.
- `non_floor_plan`: section, elevation, structural detail hoặc crop không đủ ngữ cảnh.
- `ambiguous`: không thể quyết định chắc chắn.

### Region

Region hợp lệ là polygon free-space, không phải phần thân tường giữa hai boundary lines. Loại cho phép: `room`, `corridor`, `vertical_core`, `shaft`, `void`, `other`.

Mỗi region cần:

```json
{"id": "region:001", "type": "room", "polygon": [[x0, y0], [x1, y1], [x2, y2]]}
```

Polygon dùng tọa độ pixel ảnh 980×980, có tối thiểu ba điểm và không tự cắt.

### Connectivity

Chỉ tạo relation khi nhìn thấy door/opening nối hai region đã annotation:

```json
{
  "relation": "connected_by",
  "source": "region:001",
  "target": "region:002",
  "door_instance_id": "Instance_i_1_12"
}
```

Không suy diễn kết nối qua wall strip, shaft hoặc vùng ngoài crop. Nếu cửa chỉ nối một region với ngoài crop, ghi chú thay vì tạo relation hai-region.

## Review

1. Annotator đặt `review_status=in_progress` trước khi sửa.
2. Chọn `view_type`.
3. Với `floor_plan`, vẽ region polygons rồi gắn door relations.
4. Đặt tên reviewer và `review_status=reviewed`.
5. Chạy validator; record có lỗi không được vào gold set.

### Annotation workbench local

```bash
python scripts/research/serve_room_gold_annotator.py --open-browser
```

Mặc định giao diện chạy tại `http://127.0.0.1:8765/`. Nó hỗ trợ vẽ polygon trên ảnh gốc, bật/tắt candidate hints, gắn `connected_by` bằng door instance thật, chuyển record và lưu review. Lần lưu đầu tiên tạo file backup cạnh JSONL; mỗi lần lưu dùng atomic replace và version check để tránh ghi đè một bản mới hơn.

Validator từ chối polygon trùng đỉnh, diện tích bằng không hoặc tự cắt. Giao diện không tự quyết định region nào là phòng; annotator vẫn phải kiểm tra theo định nghĩa free-space ở trên.

### Double annotation độc lập

Tạo pack mù 20% (40/200 record). Pack thứ hai giữ ảnh, selection metadata và
candidate hints nhưng không sao chép annotation/reviewer của pack chính:

```bash
python scripts/research/build_room_gold_double_pack.py
python scripts/research/serve_room_gold_annotator.py \
  --pack outputs/drawing_graph/room_gold_pack_v0_double.jsonl \
  --port 8766 --open-browser
```

Reviewer thứ hai phải khác reviewer thứ nhất. Sau khi cả hai phía có status
`reviewed`, đo agreement:

```bash
python scripts/research/measure_room_gold_agreement.py
```

Region được ghép one-to-one bằng maximum polygon IoU trong cùng region type.
`connected_by` được so sau khi ánh xạ region và vẫn yêu cầu cùng door instance.
Record chỉ qua gate khi view type khớp, reviewer độc lập, mọi region được ghép
với minimum IoU ≥0,90 và relation F1 ≥0,95.

Lệnh kiểm định và compile graph sau review:

```bash
python scripts/research/validate_room_gold_pack.py outputs/drawing_graph/room_gold_pack_v0.jsonl
python scripts/research/compile_room_gold_graphs.py
```

Compiler chỉ nhận record `reviewed`. Region người duyệt được ghi thành node `gold_region:*` và relation thật dùng `connected_by`; các preannotation tự động vẫn giữ namespace `candidate_*` và không được promote ngầm.

## Cổng quyết định

- Ít nhất 200 record, trong đó ≥100 `floor_plan` được review.
- Double annotation 20% mẫu; polygon IoU ≥0,90 và relation agreement ≥0,95.
- Nếu <50% pack là usable floor plan, phải phục hồi drawing-type/source metadata hoặc dùng dataset room-layout khác; không mở rộng heuristic labeling.
