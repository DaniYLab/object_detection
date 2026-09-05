#!/bin/bash
# B seed 4242: floorplan_base (class_embedding + FiLM fusion)
.venv/Scripts/python.exe -u train.py \
  --preset floorplan_base \
  --manifest data/FloorPlanCAD_original/splits.json \
  --ckpt-dir checkpoints/floorplan_base_seed4242 \
  --seed 4242 \
  --epochs 30 \
  --batch-size 32 \
  --num-workers 6 \
  --lr 5e-4 \
  --neg-queries-per-pos 1 \
  --val-ap-interval 5 \
  --limit-val-ap-images 256 \
  --precision bf16 \
  >> outputs/logs/floorplan_base_seed4242.log 2>&1
