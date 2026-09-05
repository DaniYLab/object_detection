#!/bin/bash
# Full hybrid inference pipeline

set -e

echo "========================================"
echo "Hybrid Inference Pipeline"
echo "========================================"
echo ""

# Phase 1: Train parallel decoder baseline
echo "[STEP] Phase 1: Train parallel decoder (fast baseline)"
python -u train_hybrid_phase1.py \
    --epochs 3 \
    --batch_size 4 \
    --output_dir checkpoints/hybrid_phase1

PHASE1_CHECKPOINT="checkpoints/hybrid_phase1/epoch_3.pt"

if [ ! -f "$PHASE1_CHECKPOINT" ]; then
    echo "[ERROR] Phase 1 checkpoint not found!"
    exit 1
fi

echo ""
echo "[OK] Phase 1 complete: $PHASE1_CHECKPOINT"
echo ""

# Phase 2: Add hybrid inference logic
echo "[STEP] Phase 2: Train with hybrid inference"
python -u train_hybrid_phase2.py \
    --phase1_checkpoint "$PHASE1_CHECKPOINT" \
    --epochs 5 \
    --batch_size 4 \
    --confidence_threshold 0.7 \
    --ambiguity_threshold 8.0 \
    --output_dir checkpoints/hybrid_phase2

PHASE2_CHECKPOINT="checkpoints/hybrid_phase2/epoch_5.pt"

if [ ! -f "$PHASE2_CHECKPOINT" ]; then
    echo "[ERROR] Phase 2 checkpoint not found!"
    exit 1
fi

echo ""
echo "[OK] Phase 2 complete: $PHASE2_CHECKPOINT"
echo ""

# Phase 3: Evaluate on val_select
echo "[STEP] Phase 3: Evaluate on val_select"
python -u evaluate_hybrid.py \
    --checkpoint "$PHASE2_CHECKPOINT" \
    --split val_select \
    --confidence_threshold 0.7 \
    --ambiguity_threshold 8.0 \
    --output outputs/hybrid_val_select.json

echo ""
echo "[OK] Evaluation complete"
echo ""

# Phase 4: Calibrate thresholds on val_calibration
echo "[STEP] Phase 4: Calibrate on val_calibration"
python -u evaluate_hybrid.py \
    --checkpoint "$PHASE2_CHECKPOINT" \
    --split val_calibration \
    --confidence_threshold 0.7 \
    --ambiguity_threshold 8.0 \
    --output outputs/hybrid_val_calibration.json

echo ""
echo "========================================"
echo "Pipeline Complete!"
echo "========================================"
echo ""
echo "Results:"
echo "  Phase 1: $PHASE1_CHECKPOINT"
echo "  Phase 2: $PHASE2_CHECKPOINT"
echo "  Val select: outputs/hybrid_val_select.json"
echo "  Val calibration: outputs/hybrid_val_calibration.json"
echo ""
