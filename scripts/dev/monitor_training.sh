#!/bin/bash
echo "Monitoring training progress..."
while true; do
    clear
    echo "=== Phase 1 Training Monitor ==="
    echo ""
    if [ -f train_phase1.log ]; then
        tail -30 train_phase1.log
    else
        echo "Log file not found yet..."
    fi
    echo ""
    echo "Press Ctrl+C to stop monitoring"
    sleep 5
done
