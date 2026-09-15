#!/bin/bash
# Quick start script for FloorPlanNet v2.1 training

echo "=========================================="
echo "FloorPlanNet v2.1 Training Launcher"
echo "=========================================="
echo ""

# Check if DualStream is still running
echo "Checking for running DualStream training..."
DUAL_STREAM_PID=$(ps aux | grep 'train_dual_stream' | grep -v grep | awk '{print $2}')

if [ ! -z "$DUAL_STREAM_PID" ]; then
    echo "Found DualStream training (PID: $DUAL_STREAM_PID)"
    echo "Do you want to stop it? (y/n)"
    read -r response
    if [ "$response" = "y" ]; then
        kill $DUAL_STREAM_PID
        echo "DualStream training stopped."
        sleep 2
    else
        echo "Keeping DualStream running. Warning: GPU may be busy!"
    fi
else
    echo "No DualStream training found."
fi

echo ""
echo "Starting FloorPlanNet v2.1 training..."
echo "Config:"
echo "  - Teacher: dual_pathway_v2_thin_seed1337/best.pt"
echo "  - Total epochs: 70"
echo "  - Distillation: First 20 epochs"
echo "  - Batch size: 12"
echo "  - Mixed precision: Enabled"
echo "  - ETA: 3-4 days"
echo ""

# Create output directory
mkdir -p checkpoints/floorplannet_v2
mkdir -p outputs/logs

# Start training in background
nohup .venv/Scripts/python.exe train_floorplannet_v2.py \
    --teacher checkpoints/dual_pathway_v2_thin_seed1337/best.pt \
    --epochs 70 \
    --distill_epochs 20 \
    --batch_size 12 \
    --lr 5e-4 \
    --warmup_epochs 3 \
    --num_workers 4 \
    --output_dir checkpoints/floorplannet_v2 \
    --save_every 10 \
    > outputs/logs/floorplannet_v2_training.log 2>&1 &

TRAINING_PID=$!

echo "Training started in background (PID: $TRAINING_PID)"
echo ""
echo "Monitor progress:"
echo "  tail -f outputs/logs/floorplannet_v2_training.log"
echo ""
echo "Check GPU usage:"
echo "  nvidia-smi"
echo ""
echo "Stop training:"
echo "  kill $TRAINING_PID"
echo ""
echo "=========================================="
