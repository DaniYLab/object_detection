#!/bin/bash
echo "=========================================="
echo "Training Progress Check"
echo "=========================================="
echo "Time: $(date '+%H:%M:%S')"
echo ""
echo "Last 10 lines:"
tail -10 training_output.log | grep -E "Epoch|Loss|Summary" || tail -10 training_output.log
echo ""
echo "Checkpoints:"
ls -lht checkpoints/dual_stream_100ep/*.pt 2>/dev/null | head -3 || echo "No checkpoints yet"
echo ""
echo "=========================================="
