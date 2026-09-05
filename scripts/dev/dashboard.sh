#!/bin/bash
# Real-time dashboard for hybrid inference experiment

PHASE1_LOG="phase1_full.log"
AUTO_LOG="auto_pipeline.log"

while true; do
    clear

    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║       HYBRID INFERENCE EXPERIMENT - REAL-TIME DASHBOARD            ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""

    # Phase 1 Status
    echo "┌─ PHASE 1: Parallel Decoder Training ────────────────────────────┐"
    if [ -f "$PHASE1_LOG" ]; then
        LAST_LINE=$(tail -20 "$PHASE1_LOG" | grep "Epoch" | tail -1)
        if [ ! -z "$LAST_LINE" ]; then
            EPOCH=$(echo "$LAST_LINE" | grep -oP 'Epoch \K[0-9]+' | head -1)
            BATCH=$(echo "$LAST_LINE" | grep -oP 'Batch \K[0-9]+' | head -1)
            LOSS=$(echo "$LAST_LINE" | grep -oP 'Loss \K[0-9]+\.[0-9]+' | head -1)

            PROGRESS=$(( (BATCH * 100) / 3799 ))
            echo "│ Epoch: $EPOCH/3  |  Batch: $BATCH/3799  |  Loss: $LOSS"
            echo "│ Progress: $PROGRESS% [$BATCH/3799 batches]"

            # Progress bar
            BAR_WIDTH=60
            FILLED=$((PROGRESS * BAR_WIDTH / 100))
            printf "│ ["
            for i in $(seq 1 $FILLED); do printf "█"; done
            for i in $(seq $((FILLED+1)) $BAR_WIDTH); do printf "░"; done
            printf "]\n"
        else
            echo "│ Status: Loading dataset..."
        fi

        # Check if complete
        if [ -f "checkpoints/hybrid_phase1/epoch_3.pt" ]; then
            echo "│ ✓ COMPLETE - Checkpoint saved"
        fi
    else
        echo "│ Status: Not started"
    fi
    echo "└──────────────────────────────────────────────────────────────────┘"
    echo ""

    # Phase 2 Status
    echo "┌─ PHASE 2: Hybrid Inference Training ────────────────────────────┐"
    if [ -f "checkpoints/hybrid_phase2/epoch_5.pt" ]; then
        echo "│ ✓ COMPLETE"
    elif grep -q "Phase 2" "$AUTO_LOG" 2>/dev/null; then
        PHASE2_LINE=$(grep "Epoch" "$AUTO_LOG" | tail -1)
        if [ ! -z "$PHASE2_LINE" ]; then
            echo "│ $PHASE2_LINE"
        else
            echo "│ Status: Starting..."
        fi
    else
        echo "│ Status: Waiting for Phase 1..."
    fi
    echo "└──────────────────────────────────────────────────────────────────┘"
    echo ""

    # Phase 3 Status
    echo "┌─ PHASE 3: Evaluation ────────────────────────────────────────────┐"
    if [ -f "outputs/hybrid_val_select.json" ]; then
        echo "│ ✓ val_select complete"
    else
        echo "│ Status: Waiting..."
    fi
    if [ -f "outputs/hybrid_val_calibration.json" ]; then
        echo "│ ✓ val_calibration complete"
    fi
    echo "└──────────────────────────────────────────────────────────────────┘"
    echo ""

    # Phase 4 Status
    echo "┌─ PHASE 4: Analysis ──────────────────────────────────────────────┐"
    if [ -f "outputs/hybrid_analysis.json" ]; then
        echo "│ ✓ COMPLETE - Run 'python summarize_hybrid.py' for report"
    else
        echo "│ Status: Waiting..."
    fi
    echo "└──────────────────────────────────────────────────────────────────┘"
    echo ""

    # Recent Activity
    echo "┌─ RECENT ACTIVITY ────────────────────────────────────────────────┐"
    if [ -f "$PHASE1_LOG" ]; then
        tail -3 "$PHASE1_LOG" | grep -E "(Epoch|Summary|Complete)" | head -2 | sed 's/^/│ /' || echo "│ Training in progress..."
    fi
    if [ -f "$AUTO_LOG" ]; then
        tail -3 "$AUTO_LOG" | grep -E "(Phase|Complete|Starting)" | head -1 | sed 's/^/│ /' 2>/dev/null
    fi
    echo "└──────────────────────────────────────────────────────────────────┘"
    echo ""

    # Estimates
    echo "┌─ TIME ESTIMATES ─────────────────────────────────────────────────┐"
    if [ -f "$PHASE1_LOG" ] && [ ! -f "checkpoints/hybrid_phase1/epoch_3.pt" ]; then
        LAST_LINE=$(tail -20 "$PHASE1_LOG" | grep "Epoch" | tail -1)
        if [ ! -z "$LAST_LINE" ]; then
            BATCH=$(echo "$LAST_LINE" | grep -oP 'Batch \K[0-9]+' | head -1)
            TIME=$(echo "$LAST_LINE" | grep -oP 'Time \K[0-9]+\.[0-9]+' | head -1)
            if [ ! -z "$BATCH" ] && [ ! -z "$TIME" ] && [ "$BATCH" -gt 0 ]; then
                TIME_PER_BATCH=$(echo "scale=2; $TIME / $BATCH" | bc -l 2>/dev/null || echo "0")
                if [ "$TIME_PER_BATCH" != "0" ]; then
                    REMAINING_BATCHES=$((11397 - BATCH))  # 3 epochs total = 3799*3
                    REMAINING_MIN=$(echo "scale=1; $REMAINING_BATCHES * $TIME_PER_BATCH / 60" | bc -l 2>/dev/null || echo "?")
                    echo "│ Phase 1 remaining: ~$REMAINING_MIN minutes"
                fi
            fi
        fi
    fi

    if [ -f "checkpoints/hybrid_phase1/epoch_3.pt" ] && [ ! -f "checkpoints/hybrid_phase2/epoch_5.pt" ]; then
        echo "│ Phase 2 estimated: ~50 minutes"
    fi

    if [ ! -f "outputs/hybrid_analysis.json" ]; then
        echo "│ Total pipeline: ~2 hours from start"
    else
        echo "│ ✓ Pipeline complete!"
    fi
    echo "└──────────────────────────────────────────────────────────────────┘"
    echo ""

    echo "Press Ctrl+C to exit | Refreshing every 10s..."
    echo ""

    sleep 10
done
