#!/bin/bash
# Wait for Phase 1 to complete, then automatically run Phase 2-4

PHASE1_LOG="phase1_full.log"
PHASE1_CHECKPOINT="checkpoints/hybrid_phase1/epoch_3.pt"

echo "=========================================="
echo "Hybrid Pipeline Auto-Runner"
echo "=========================================="
echo ""
echo "Waiting for Phase 1 to complete..."
echo ""

# Wait for Phase 1 completion
while true; do
    if [ -f "$PHASE1_CHECKPOINT" ]; then
        echo "[✓] Phase 1 checkpoint found!"
        break
    fi

    # Check if log is being updated (training still active)
    if [ -f "$PHASE1_LOG" ]; then
        LOG_AGE=$(( $(date +%s) - $(stat -c %Y "$PHASE1_LOG" 2>/dev/null || stat -f %m "$PHASE1_LOG" 2>/dev/null || echo 0) ))
        if [ "$LOG_AGE" -gt 300 ]; then
            echo "[✗] Phase 1 log not updated for 5+ minutes and no checkpoint found"
            echo "Check $PHASE1_LOG for errors"
            exit 1
        fi
    fi

    # Show progress
    LAST_LINE=$(tail -5 "$PHASE1_LOG" | grep "Epoch" | tail -1)
    if [ ! -z "$LAST_LINE" ]; then
        echo -ne "\r$LAST_LINE"
    fi

    sleep 10
done

echo ""
echo ""
echo "=========================================="
echo "Phase 1 Complete! Starting Phase 2..."
echo "=========================================="
echo ""

# Phase 2: Hybrid Inference Training
python -u train_hybrid_phase2.py \
    --phase1_checkpoint "$PHASE1_CHECKPOINT" \
    --epochs 5 \
    --batch_size 4 \
    --confidence_threshold 0.7 \
    --ambiguity_threshold 8.0 \
    --output_dir checkpoints/hybrid_phase2 \
    2>&1 | tee phase2.log

if [ $? -ne 0 ]; then
    echo "[✗] Phase 2 failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Phase 2 Complete! Starting Evaluation..."
echo "=========================================="
echo ""

PHASE2_CHECKPOINT="checkpoints/hybrid_phase2/epoch_5.pt"

# Phase 3: Evaluation
python -u evaluate_hybrid.py \
    --checkpoint "$PHASE2_CHECKPOINT" \
    --split val_select \
    --confidence_threshold 0.7 \
    --ambiguity_threshold 8.0 \
    --output outputs/hybrid_val_select.json \
    2>&1 | tee eval_val_select.log

python -u evaluate_hybrid.py \
    --checkpoint "$PHASE2_CHECKPOINT" \
    --split val_calibration \
    --confidence_threshold 0.7 \
    --ambiguity_threshold 8.0 \
    --output outputs/hybrid_val_calibration.json \
    2>&1 | tee eval_val_calibration.log

echo ""
echo "=========================================="
echo "Evaluation Complete! Generating Analysis..."
echo "=========================================="
echo ""

# Phase 4: Analysis
python analyze_hybrid.py \
    --hybrid outputs/hybrid_val_select.json \
    --output outputs/hybrid_analysis.json

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""

python summarize_hybrid.py

echo ""
echo "All results saved to outputs/"
echo "Run 'python summarize_hybrid.py' to see summary"
echo ""
