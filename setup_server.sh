#!/bin/bash
# =============================================================================
# FloorPlanCAD — SSH Server Setup Script
# Chạy 1 lần duy nhất khi mới thuê server
# Usage: bash setup_server.sh
# =============================================================================

set -e  # Dừng nếu có lỗi

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[+]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[x]${NC} $1"; }

echo "============================================================"
echo "  FloorPlanCAD — Server Setup"
echo "============================================================"

# ── 1. System packages ────────────────────────────────────────────────────────
log "Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq tmux htop nvtop tree git curl unzip

# ── 2. Kiểm tra CUDA ─────────────────────────────────────────────────────────
log "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
    err "No GPU detected!"; exit 1
}
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//' | tr -d '.')
log "CUDA version: $CUDA_VERSION"

# ── 3. Python environment ─────────────────────────────────────────────────────
log "Setting up Python environment..."
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip -q

# Install PyTorch với CUDA phù hợp
# Parse CUDA major version (e.g. "130" → 13, "124" → 12, "118" → 11)
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -c1-2 | sed 's/^0//')
CUDA_MINOR=$(echo $CUDA_VERSION | cut -c3)

if   [[ "$CUDA_MAJOR" -ge 13 ]]; then
    TORCH_CUDA="cu132"
elif [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -ge 4 ]]; then
    TORCH_CUDA="cu124"
elif [[ "$CUDA_MAJOR" -eq 12 ]]; then
    TORCH_CUDA="cu121"
elif [[ "$CUDA_MAJOR" -eq 11 && "$CUDA_MINOR" -ge 8 ]]; then
    TORCH_CUDA="cu118"
else
    warn "Old CUDA $CUDA_MAJOR.$CUDA_MINOR, defaulting to cu118"
    TORCH_CUDA="cu118"
fi

log "Installing PyTorch for $TORCH_CUDA..."
pip install torch torchvision \
    --index-url "https://download.pytorch.org/whl/$TORCH_CUDA" -q


# Install gdown trước (cần để download dataset)
pip install gdown -q

# Install remaining dependencies
pip install -r requirements.txt -q

log "Python environment ready!"
python -c "import torch; print(f'  torch={torch.__version__}, cuda={torch.cuda.is_available()}, gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

# ── 4. Download dataset ───────────────────────────────────────────────────────
log "Downloading FloorPlanCAD dataset..."
mkdir -p data
python scripts/data/download_gdrive.py

log "Building processed dataset (images + metadata)..."
python scripts/data/build_dataset.py

# ── 5. Verify ─────────────────────────────────────────────────────────────────
log "Verifying dataset..."
python - <<'EOF'
import sys; sys.path.insert(0, '.')
from src.data.dataset import FloorPlanDataset, NUM_CLASSES
train_ds = FloorPlanDataset('./data/FloorPlanCAD_dataset', split='train')
val_ds   = FloorPlanDataset('./data/FloorPlanCAD_dataset', split='test')
print(f'  Train: {len(train_ds):,} | Val: {len(val_ds):,} | Classes: {NUM_CLASSES}')
EOF

echo ""
echo "============================================================"
echo -e "  ${GREEN}Setup hoàn tất!${NC}"
echo "  Chạy training: bash run_train.sh"
echo "============================================================"
