#!/usr/bin/env bash
# FloorPlanCAD server setup. This script prepares and verifies the environment;
# it does not start training.

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[+]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[x]${NC} $1"; }

echo "============================================================"
echo "  FloorPlanCAD — Server Setup (no training)"
echo "============================================================"

log "Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv tmux htop nvtop tree git curl unzip

log "Checking GPU driver..."
if ! nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; then
    warn "No NVIDIA GPU detected. CPU tests still work, but future training will be slow."
fi

log "Setting up Python environment..."
if [[ ! -d .venv ]]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

# Do not guess a wheel from the locally-installed CUDA toolkit. Override this for
# the server image, e.g. TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128.
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
log "Installing PyTorch from $TORCH_INDEX_URL..."
python -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"
python -m pip install -r requirements.txt -r requirements-dev.txt

python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
PY

if [[ ! -d data/FloorPlanCAD_original ]]; then
    log "Downloading FloorPlanCAD dataset..."
    mkdir -p data
    python scripts/data/download_gdrive.py
else
    log "Dataset directory already exists; skipping download."
fi

log "Building/validating metadata schema v2 (stuff excluded for object benchmark)..."
python scripts/data/build_dataset.py \
    --data-root ./data/FloorPlanCAD_original \
    --stuff-policy exclude

if [[ ! -f data/FloorPlanCAD_original/splits.json ]]; then
    log "Creating deterministic image-level train/val/test manifest..."
    python scripts/data/build_splits.py \
        --data-root ./data/FloorPlanCAD_original \
        --output ./data/FloorPlanCAD_original/splits.json \
        --seed 1337 \
        --val-fraction 0.10
else
    log "Split manifest already exists; validating in memory."
    python scripts/data/build_splits.py \
        --data-root ./data/FloorPlanCAD_original \
        --seed 1337 \
        --val-fraction 0.10 \
        --validate-only
fi

log "Running no-training verification..."
python -m pytest -q
python scripts/dev/smoke_models.py \
    --device cpu \
    --image-size 32 \
    --model-dim 16 \
    --depth 1 \
    --all-lightweight-presets

echo ""
echo "============================================================"
echo -e "  ${GREEN}Setup and verification complete.${NC}"
echo "  No training was started. Review README.md before using run_train.sh."
echo "============================================================"
