#!/bin/bash
# Real-time monitoring dashboard for running experiments

echo "========================================"
echo "FloorPlanCAD Experiment Monitor"
echo "========================================"
echo "Time: $(date '+%H:%M:%S')"
echo ""

echo "--- Active Tasks ---"
echo ""

# Task 1: DualStreamCenterNet evaluation
echo "[1] DualStreamCenterNet Evaluation"
if [ -f "C:/Users/PC/AppData/Local/Temp/claude/D--Dat-POC-object-detection/03fca7ed-f8ab-4a86-b05d-eb81ccc06a32/tasks/bs60b8rya.output" ]; then
    tail -5 "C:/Users/PC/AppData/Local/Temp/claude/D--Dat-POC-object-detection/03fca7ed-f8ab-4a86-b05d-eb81ccc06a32/tasks/bs60b8rya.output" 2>/dev/null | grep -E "Inference:|Computing|AP50|Saved" || echo "  Status: Running inference..."
else
    echo "  Status: Not started"
fi
echo ""

# Task 2: Hybrid pipeline
echo "[2] Hybrid Inference Pipeline"
if [ -f "C:/Users/PC/AppData/Local/Temp/claude/D--Dat-POC-object-detection/03fca7ed-f8ab-4a86-b05d-eb81ccc06a32/tasks/bg0qnoyk6.output" ]; then
    tail -3 "C:/Users/PC/AppData/Local/Temp/claude/D--Dat-POC-object-detection/03fca7ed-f8ab-4a86-b05d-eb81ccc06a32/tasks/bg0qnoyk6.output" 2>/dev/null | grep -E "STEP|Phase|Complete|Error" || echo "  Status: Running..."
else
    echo "  Status: Not started"
fi
echo ""

# Check if results exist
echo "--- Completed Results ---"
echo ""

if [ -f "outputs/dual_stream_val_select.json" ]; then
    echo "[OK] DualStreamCenterNet results available"
    cat outputs/dual_stream_val_select.json | grep -E "AP50|AP50:95" | head -2
else
    echo "[ ] DualStreamCenterNet: pending"
fi

if [ -f "outputs/hybrid_analysis.json" ]; then
    echo "[OK] Hybrid analysis available"
else
    echo "[ ] Hybrid analysis: pending"
fi

echo ""
echo "========================================"
echo "Run 'bash monitor_experiments.sh' to refresh"
