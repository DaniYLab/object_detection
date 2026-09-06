# Verification Without Training

Các kiểm tra trong tài liệu này không được chạy training epoch và không gọi `optimizer.step()`.

## Static checks

```bash
python -m compileall -q train.py evaluate.py src scripts tests
```

## Unit and synthetic tests

```bash
python -m pytest -q
```

Nhóm critical có thể chạy riêng:

```bash
python -m pytest -q tests/test_attention.py tests/test_conditioning.py tests/test_models.py
python -m pytest -q tests/test_metadata.py tests/test_splits.py tests/test_targets.py tests/test_transforms.py
python -m pytest -q tests/test_decoder.py tests/test_metrics.py
python -m pytest -q tests/test_losses.py tests/test_reproducibility.py tests/test_checkpoints.py
```

Các test dùng tensor/fixtures nhỏ trên CPU. Backward trong attention/loss test chỉ kiểm tra gradient correctness; nó không cập nhật weight. Checkpoint/conditioner tests dùng injected fake HF backend, không import `transformers`, không download network weights và không gọi `optimizer.step()`.

## Model forward smoke

```bash
python scripts/dev/smoke_models.py \
  --device cpu \
  --image-size 32 \
  --model-dim 16 \
  --depth 1 \
  --all-lightweight-presets
```

Pretrained text preset không được download hoặc chạy trong default smoke suite.

## Dataset read-only checks

Nếu dataset có sẵn:

```bash
python scripts/data/build_dataset.py \
  --data-root ./data/FloorPlanCAD_original \
  --stuff-policy exclude \
  --validate-only

python scripts/data/build_splits.py \
  --data-root ./data/FloorPlanCAD_original \
  --seed 1337 \
  --val-fraction 0.10 \
  --validate-only
```

Không dùng `--force` trong verification thông thường vì flag đó có thể thay metadata hiện có.

## Không được dùng để verify task này

```bash
python train.py --epochs 1
bash run_train.sh
```

Hai command trên thực sự training và nằm ngoài phạm vi triển khai hiện tại.
