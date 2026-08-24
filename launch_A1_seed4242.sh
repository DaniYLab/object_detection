#!/bin/bash
# A1 seed 4242: shared_no_condition (architecture control)
.venv/Scripts/python.exe -u train.py \
  --preset shared_no_condition \
  --manifest data/FloorPlanCAD_original/splits.json \
  --ckpt-dir checkpoints/shared_no_condition_seed4242 \
  --seed 4242 \
  --epochs 30 \
  --batch-size 32 \
  --num-workers 6 \
  --lr 5e-4 \
  --neg-queries-per-pos 1 \
  --val-ap-interval 5 \
  --limit-val-ap-images 256 \
  --precision bf16 \
  >> outputs/logs/shared_no_condition_seed4242.log 2>&1
