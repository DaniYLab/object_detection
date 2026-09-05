#!/bin/bash
# Extended training: 100 epochs with learning rate scheduling

set -e

PYTHON=".venv/Scripts/python.exe"

echo "============================================================"
echo "DualStreamCenterNet Extended Training (100 epochs)"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  - Preset: small (9.8M params)"
echo "  - Epochs: 100"
echo "  - Batch size: 8"
echo "  - Base LR: 1e-4"
echo "  - Warmup: 5 epochs"
echo "  - LR schedule: Cosine decay after warmup"
echo "  - Checkpoints: Every 10 epochs"
echo "  - Validation: Every 5 epochs on val_select"
echo ""
echo "Expected training time: ~8-12 hours on single GPU"
echo "============================================================"
echo ""

$PYTHON train_dual_stream_centernet.py \
  --preset small \
  --epochs 100 \
  --batch_size 8 \
  --lr 1e-4 \
  --warmup_epochs 5 \
  --output_dir checkpoints/dual_stream_100ep \
  --val_every 5 \
  --save_every 10

echo ""
echo "============================================================"
echo "Training Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Check convergence: look for heatmap loss <0.5"
echo "  2. Run diagnostic: python diagnose_dual_stream.py"
echo "  3. Evaluate best checkpoint on val_select"
echo "  4. If AP50 >0.15, proceed with ablation studies"
echo ""
