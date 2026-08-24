#!/usr/bin/env bash
# FloorPlanCAD future-training launcher.
# This file is intentionally not a smoke/verification command.
# Usage: bash run_train.sh

set -euo pipefail

if [[ ! -f .venv/bin/activate ]]; then
    echo "Missing .venv/bin/activate. Run setup_server.sh or create the environment first." >&2
    exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# Paths and reproducibility contract.
DATA_ROOT="${DATA_ROOT:-./data/FloorPlanCAD_original}"
MANIFEST="${MANIFEST:-$DATA_ROOT/splits.json}"
CKPT_DIR="${CKPT_DIR:-./checkpoints}"
LOG_DIR="${LOG_DIR:-./logs}"
SESSION="${SESSION:-floorplan_train}"
PRESET="${PRESET:-floorplan_base}"
SEED="${SEED:-1337}"

# Conservative fixed defaults. Change BATCH_SIZE explicitly for the target server;
# this launcher never infers it from VRAM.
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-1e-5}"
FOCAL_WEIGHT="${FOCAL_WEIGHT:-10.0}"
SIZE_WEIGHT="${SIZE_WEIGHT:-1.0}"
OFFSET_WEIGHT="${OFFSET_WEIGHT:-1.0}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
WARMUP_START_FACTOR="${WARMUP_START_FACTOR:-0.1}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.01}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
SAMPLER="${SAMPLER:-balanced}"
BALANCE_POWER="${BALANCE_POWER:-0.5}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
RESUME="${RESUME:-}"
WEIGHTS_ONLY="${WEIGHTS_ONLY:-0}"
DETERMINISTIC="${DETERMINISTIC:-0}"

if [[ ! -f "$MANIFEST" ]]; then
    echo "Split manifest not found: $MANIFEST" >&2
    echo "Create it with scripts/data/build_splits.py before training." >&2
    exit 1
fi

mkdir -p "$CKPT_DIR" "$LOG_DIR"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU info (informational only; batch size remains $BATCH_SIZE):"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
    echo ""
fi

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$LOG_DIR/train_$TIMESTAMP.log"

CMD=(
    python train.py
    --data-root "$DATA_ROOT"
    --manifest "$MANIFEST"
    --preset "$PRESET"
    --seed "$SEED"
    --ckpt-dir "$CKPT_DIR"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --epochs "$EPOCHS"
    --lr "$LR"
    --focal-weight "$FOCAL_WEIGHT"
    --size-weight "$SIZE_WEIGHT"
    --offset-weight "$OFFSET_WEIGHT"
    --warmup-steps "$WARMUP_STEPS"
    --warmup-start-factor "$WARMUP_START_FACTOR"
    --min-lr-ratio "$MIN_LR_RATIO"
    --grad-clip "$GRAD_CLIP"
    --sampler "$SAMPLER"
    --balance-power "$BALANCE_POWER"
    --log-interval "$LOG_INTERVAL"
)

if [[ "$DETERMINISTIC" == "1" ]]; then
    CMD+=(--deterministic)
fi
if [[ -n "$RESUME" ]]; then
    CMD+=(--resume "$RESUME")
    if [[ "$WEIGHTS_ONLY" == "1" ]]; then
        CMD+=(--weights-only)
    fi
elif [[ "$WEIGHTS_ONLY" == "1" ]]; then
    echo "WEIGHTS_ONLY=1 requires RESUME=/path/to/checkpoint.pt" >&2
    exit 1
fi

printf -v TRAIN_COMMAND '%q ' "${CMD[@]}"
printf -v LOG_PATH_QUOTED '%q' "$LOG_FILE"
SESSION_COMMAND="${TRAIN_COMMAND}2>&1 | tee ${LOG_PATH_QUOTED}; status=\${PIPESTATUS[0]}; echo \"Training command exited with status \$status\"; exec bash"
printf -v TMUX_COMMAND 'bash -lc %q' "$SESSION_COMMAND"

tmux kill-session -t "$SESSION" 2>/dev/null || true

echo "Starting future training run in tmux session: $SESSION"
echo "Preset    : $PRESET"
echo "Manifest  : $MANIFEST"
echo "Seed      : $SEED"
echo "Batch size: $BATCH_SIZE (fixed)"
echo "Log file  : $LOG_FILE"
echo ""
echo "Attach   : tmux attach -t $SESSION"
echo "Detach   : Ctrl+B, D"
echo "Stop     : tmux kill-session -t $SESSION"
echo "Tail log : tail -f $LOG_FILE"
echo ""

tmux new-session -d -s "$SESSION" "$TMUX_COMMAND"
echo "Training process launched. Attach to inspect its output."
