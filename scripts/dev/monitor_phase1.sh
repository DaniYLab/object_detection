#!/bin/bash
# Automatic training monitor with progress bar

LOG_FILE="phase1_full.log"
TOTAL_EPOCHS=3
BATCHES_PER_EPOCH=3799

echo "=========================================="
echo "Training Monitor"
echo "=========================================="
echo ""

get_progress() {
    if [ ! -f "$LOG_FILE" ]; then
        echo "0 0 0"
        return
    fi

    # Extract latest epoch and batch
    LAST_LINE=$(tail -20 "$LOG_FILE" | grep "Epoch" | tail -1)

    if [ -z "$LAST_LINE" ]; then
        echo "0 0 0"
        return
    fi

    EPOCH=$(echo "$LAST_LINE" | grep -oP 'Epoch \K[0-9]+' | head -1)
    BATCH=$(echo "$LAST_LINE" | grep -oP 'Batch \K[0-9]+' | head -1)
    LOSS=$(echo "$LAST_LINE" | grep -oP 'Loss \K[0-9]+\.[0-9]+' | head -1)

    echo "${EPOCH:-0} ${BATCH:-0} ${LOSS:-0}"
}

while true; do
    clear
    echo "=========================================="
    echo "Phase 1 Training Monitor"
    echo "=========================================="
    echo ""

    read EPOCH BATCH LOSS <<< $(get_progress)

    if [ "$EPOCH" -eq 0 ] && [ "$BATCH" -eq 0 ]; then
        echo "Status: Initializing..."
    else
        # Calculate progress
        TOTAL_BATCHES=$((TOTAL_EPOCHS * BATCHES_PER_EPOCH))
        COMPLETED_BATCHES=$(( (EPOCH - 1) * BATCHES_PER_EPOCH + BATCH ))
        PROGRESS=$((COMPLETED_BATCHES * 100 / TOTAL_BATCHES))

        echo "Epoch: $EPOCH / $TOTAL_EPOCHS"
        echo "Batch: $BATCH / $BATCHES_PER_EPOCH"
        echo "Loss: $LOSS"
        echo ""
        echo "Overall Progress: $PROGRESS%"

        # Progress bar
        BAR_WIDTH=50
        FILLED=$((PROGRESS * BAR_WIDTH / 100))
        printf "["
        for i in $(seq 1 $FILLED); do printf "="; done
        for i in $(seq $FILLED $((BAR_WIDTH-1))); do printf " "; done
        printf "] $PROGRESS%%\n"

        # Time estimate
        if [ "$BATCH" -gt 100 ]; then
            TIME_LINE=$(tail -20 "$LOG_FILE" | grep "Epoch $EPOCH.*Batch $BATCH" | tail -1)
            TIME_ELAPSED=$(echo "$TIME_LINE" | grep -oP 'Time \K[0-9]+\.[0-9]+' | head -1)
            if [ ! -z "$TIME_ELAPSED" ]; then
                TIME_PER_BATCH=$(echo "scale=2; $TIME_ELAPSED / $BATCH" | bc)
                REMAINING_BATCHES=$((TOTAL_BATCHES - COMPLETED_BATCHES))
                REMAINING_TIME=$(echo "scale=1; $REMAINING_BATCHES * $TIME_PER_BATCH / 60" | bc)
                echo ""
                echo "Estimated remaining: ${REMAINING_TIME} minutes"
            fi
        fi
    fi

    echo ""
    echo "Recent log:"
    echo "----------"
    tail -10 "$LOG_FILE" 2>/dev/null | grep -E "(Epoch|OK|Complete)" || echo "No log yet..."

    echo ""
    echo "Press Ctrl+C to exit monitoring"
    echo "Refreshing in 10s..."

    sleep 10
done
