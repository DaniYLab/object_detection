#!/bin/bash
# =============================================================================
# FloorPlanCAD — Training Launcher
# Chạy trong tmux để không bị ngắt khi SSH disconnect
# Usage: bash run_train.sh [config]
# =============================================================================

source .venv/bin/activate

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT="./data/FloorPlanCAD_original"
CKPT_DIR="./checkpoints"
LOG_DIR="./logs"
SESSION="floorplan_train"

# GPU config — tự chỉnh theo server
IMAGE_SIZE=512
BATCH_SIZE=4        # Auto-adjusted by VRAM below; safe default if nvidia-smi fails
MODEL_DIM=512
DEPTH_PER_CLASS=2
EPOCHS=50
LR=1e-5
FOCAL_WEIGHT=10.0
SIZE_WEIGHT=1.0
OFFSET_WEIGHT=1.0
WARMUP_STEPS=500
GRAD_CLIP=1.0
SAMPLER="balanced"
BALANCE_POWER=0.5
FUSION_MODE="film"
NUM_WORKERS=4
LOG_INTERVAL=50

mkdir -p "$CKPT_DIR" "$LOG_DIR"

# ── Kiểm tra GPU ─────────────────────────────────────────────────────────────
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# ── Tự động adjust batch size theo VRAM ──────────────────────────────────────
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if   [ "$VRAM" -ge 79000 ]; then BATCH_SIZE=32   # A100 80GB
elif [ "$VRAM" -ge 39000 ]; then BATCH_SIZE=24   # A100 40GB / A6000
elif [ "$VRAM" -ge 23000 ]; then BATCH_SIZE=16   # RTX 4090 / A5000
elif [ "$VRAM" -ge 15000 ]; then BATCH_SIZE=8    # T4 / RTX 3080
else                              BATCH_SIZE=4    # < 16GB
fi
echo "Auto batch_size = $BATCH_SIZE (VRAM = ${VRAM}MB)"

# ── Launch trong tmux ─────────────────────────────────────────────────────────
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="$LOG_DIR/train_$TIMESTAMP.log"

CMD="python train.py \
    --data_root    $DATA_ROOT \
    --ckpt_dir     $CKPT_DIR \
    --image_size   $IMAGE_SIZE \
    --batch_size   $BATCH_SIZE \
    --num_workers  $NUM_WORKERS \
    --model_dim    $MODEL_DIM \
    --depth_per_class $DEPTH_PER_CLASS \
    --epochs       $EPOCHS \
    --lr           $LR \
    --focal_weight $FOCAL_WEIGHT \
    --size_weight  $SIZE_WEIGHT \
    --offset_weight $OFFSET_WEIGHT \
    --warmup_steps $WARMUP_STEPS \
    --grad_clip    $GRAD_CLIP \
    --sampler      $SAMPLER \
    --balance_power $BALANCE_POWER \
    --fusion_mode  $FUSION_MODE \
    --log_interval $LOG_INTERVAL \
    2>&1 | tee $LOG_FILE"

# Kill session cũ nếu có
tmux kill-session -t "$SESSION" 2>/dev/null || true

echo "Starting training in tmux session: $SESSION"
echo "Log file: $LOG_FILE"
echo ""
echo "Commands:"
echo "  Attach   : tmux attach -t $SESSION"
echo "  Detach   : Ctrl+B, D"
echo "  Kill     : tmux kill-session -t $SESSION"
echo "  Tail log : tail -f $LOG_FILE"
echo ""

tmux new-session -d -s "$SESSION" "bash -c '$CMD; echo DONE; bash'"
sleep 2

# Hiển thị 10 dòng đầu
echo "=== Training started ==="
sleep 3
tail -20 "$LOG_FILE" 2>/dev/null || echo "(Log chưa có — attach vào tmux để xem)"

echo ""
echo "Attach vào tmux: tmux attach -t $SESSION"
