# Baseline and External Prediction Protocol

## Project-native baseline

`centernet_shared` là baseline nội bộ dùng:

- cùng image encoder và output stride với các conditioned variants;
- một shared pathway;
- không text, không class routing;
- multi-class CenterNet head cho heatmap, size và offset.

Mục đích là giữ data, encoder, target, decoder và AP evaluator giống nhau để đo riêng tác động của conditioning/pathway. Đây không phải reproduction chính thức của CenterNet paper.

## Conditioned baselines

Các preset ablation được định nghĩa trong `src/models/presets.py`. Không so sánh chỉ bằng tên kiến trúc; mọi report phải ghi config đã resolve và parameter count thực tế từ code.

## External models

YOLO, Faster R-CNN hoặc detector khác không cần được nhúng vào training code của repository. Chúng xuất prediction JSON theo schema chung rồi được đánh giá bằng cùng `evaluate.py`, split manifest và AP implementation.

### Prediction JSON schema

```json
{
  "format": "object_detection_predictions",
  "schema_version": 1,
  "box_format": "xyxy",
  "coordinate_space": "image",
  "metadata": {
    "producer": "external-model-name",
    "split_manifest_fingerprint": "...",
    "class_names": ["annotation_text", "..."]
  },
  "predictions": [
    {
      "image_id": "test_set/example",
      "boxes": [[10.0, 20.0, 50.0, 70.0]],
      "scores": [0.92],
      "labels": [4]
    }
  ]
}
```

Quy ước:

- `boxes` dùng `[x0, y0, x1, y1)` trong resized input pixels;
- `labels` dùng alphabetic `CLASS_NAMES` index của repository;
- box, score và label arrays phải cùng độ dài;
- score phải finite trong `[0,1]`;
- label phải thuộc `[0, NUM_CLASSES)`;
- image ID là relative path gồm source folder, không chỉ dùng stem;
- một image không có detection vẫn cần record với ba array rỗng hoặc evaluator phải biết đầy đủ image list từ split manifest.

## Fair-comparison checklist

1. Cùng metadata schema và stuff policy.
2. Cùng train/val/test image IDs.
3. Không tune trên test set.
4. Cùng image resizing convention hoặc map box về cùng input coordinates.
5. Cùng class mapping.
6. Cùng AP implementation và IoU thresholds.
7. Báo parameter count, latency, input resolution và pretrained data.
8. Không dùng prediction threshold khác nhau mà không ghi rõ; AP nên nhận đủ scored predictions sau một ngưỡng sàn thấp.
