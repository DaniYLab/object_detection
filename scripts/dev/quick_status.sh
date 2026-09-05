#!/bin/bash
# Quick status check for research progress

echo "=================================================="
echo "FloorPlanCAD Research Status"
echo "=================================================="
echo "Time: $(date)"
echo ""

# Training progress
echo "[1] Training Status:"
if ps aux | grep "python.*train_dual_stream" | grep -v grep > /dev/null; then
    echo "  ✅ Training process RUNNING"
    LAST_LINE=$(tail -1 training_bs4.log 2>/dev/null)
    if [[ $LAST_LINE == *"Epoch"* ]]; then
        echo "  $LAST_LINE"
    else
        echo "  Starting up..."
    fi
else
    echo "  ❌ Training process NOT FOUND"
fi

echo ""

# Checkpoints
echo "[2] Checkpoints:"
if ls checkpoints/dual_stream_100ep/*.pt 2>/dev/null | wc -l > /dev/null; then
    COUNT=$(ls checkpoints/dual_stream_100ep/*.pt 2>/dev/null | wc -l)
    echo "  Found $COUNT checkpoint(s)"
    ls -lht checkpoints/dual_stream_100ep/*.pt 2>/dev/null | head -3
else
    echo "  No checkpoints yet (will appear at epoch 5, 10, ...)"
fi

echo ""

# Goal progress
echo "[3] Goal Progress (Hoàn thành nghiên cứu):"
echo "  ✅ Phase 0: Architecture implemented"
echo "  🔄 Phase 1: Baseline training (H1) - IN PROGRESS"
echo "  ⏸️  Phase 2: Ablations (H2-H3) - Waiting for H1"
echo "  ⏸️  Phase 3: Final test evaluation - Waiting for H2-H3"
echo ""

echo "=================================================="
echo "Commands:"
echo "  Monitor: tail -f training_bs4.log"
echo "  Status: bash quick_status.sh"
echo "=================================================="
