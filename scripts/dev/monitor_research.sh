#!/bin/bash
# Monitor both training and audit progress

echo "=========================================="
echo "Research Progress Monitor"
echo "=========================================="
echo ""

# Training progress
TRAIN_OUTPUT="C:/Users/PC/AppData/Local/Temp/claude/D--Dat-POC-object-detection/03fca7ed-f8ab-4a86-b05d-eb81ccc06a32/tasks/by6a393c2.output"

if [ -f "$TRAIN_OUTPUT" ]; then
    echo "[1] 100-Epoch Training Status:"
    tail -10 "$TRAIN_OUTPUT" | grep -E "Epoch|Loss|Heatmap|Summary|Validation" | tail -5 || echo "  Initializing..."
else
    echo "[1] Training: Not started"
fi

echo ""

# Audit progress
AUDIT_OUTPUT="C:/Users/PC/AppData/Local/Temp/claude/D--Dat-POC-object-detection/03fca7ed-f8ab-4a86-b05d-eb81ccc06a32/tasks/b8zeu4gng.output"

if [ -f "$AUDIT_OUTPUT" ]; then
    echo "[2] Stride-8 Collision Audit Status:"
    tail -15 "$AUDIT_OUTPUT" | grep -E "Processing|collision|Decision|OK" | tail -5 || echo "  Running..."
else
    echo "[2] Audit: Not started"
fi

echo ""
echo "=========================================="
echo "Monitoring commands:"
echo "  Training: tail -f $TRAIN_OUTPUT"
echo "  Audit: tail -f $AUDIT_OUTPUT"
echo "=========================================="
