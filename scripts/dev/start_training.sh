#!/bin/bash
# Simple launcher with unbuffered output

export PYTHONUNBUFFERED=1

echo "Starting 100-epoch training..."
echo "Start time: $(date)"

python -u train_dual_stream_extended.py \
  --preset small \
  --epochs 100 \
  --batch_size 8 \
  --lr 1e-4 \
  --warmup_epochs 5 \
  --output_dir checkpoints/dual_stream_100ep \
  --save_every 10 \
  --val_every 5

echo "Training completed!"
echo "End time: $(date)"
