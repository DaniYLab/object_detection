#!/bin/bash
# Resume hybrid pipeline from Phase 2 onwards

set -e

echo "============================================================"
echo "Hybrid Pipeline Resume - Phases 2-4"
echo "============================================================"

PYTHON=".venv/Scripts/python.exe"
PHASE1_CHECKPOINT="checkpoints/hybrid_parallel_phase1/best_val.pt"

# Phase 2: Train hybrid wrapper (confidence-based fallback)
echo ""
echo "[PHASE 2] Training hybrid wrapper with fallback logic..."
echo "------------------------------------------------------------"

$PYTHON train_hybrid_phase2.py \
  --parallel_checkpoint "$PHASE1_CHECKPOINT" \
  --preset small \
  --epochs 5 \
  --batch_size 8 \
  --lr 5e-5 \
  --confidence_threshold 0.7 \
  --spatial_threshold 8 \
  --output_dir checkpoints/hybrid_phase2

PHASE2_CHECKPOINT="checkpoints/hybrid_phase2/best_val.pt"

# Phase 3: Evaluate hybrid model on val_select
echo ""
echo "[PHASE 3] Evaluating hybrid model..."
echo "------------------------------------------------------------"

$PYTHON evaluate_hybrid.py \
  --checkpoint "$PHASE2_CHECKPOINT" \
  --preset small \
  --split val_select \
  --batch_size 8 \
  --output outputs/hybrid_val_select.json

# Phase 4: Analyze hybrid results vs baseline
echo ""
echo "[PHASE 4] Analyzing hybrid performance..."
echo "------------------------------------------------------------"

$PYTHON analyze_hybrid.py \
  --hybrid_results outputs/hybrid_val_select.json \
  --baseline_results outputs/evaluation_val_select_dual_pathway_v2_thin_seed1337.json \
  --output outputs/hybrid_analysis.json

echo ""
echo "============================================================"
echo "Hybrid Pipeline Complete!"
echo "============================================================"
echo ""
echo "Results:"
echo "  - Phase 2 checkpoint: $PHASE2_CHECKPOINT"
echo "  - Evaluation: outputs/hybrid_val_select.json"
echo "  - Analysis: outputs/hybrid_analysis.json"
echo ""
