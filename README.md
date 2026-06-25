# FloorPlanCAD Object Detection

Multimodal deep learning model for object detection in CAD floor plan drawings.

## Architecture

```
Image (PNG) → VAE Encoder → Image Tokens
Text ("Find {class}") → Text Encoder → Text Tokens
                              ↓
                        Early Fusion (Cross-Attention)
                              ↓
              Class-Conditioned Mamba + Self-Attention Blocks
                              ↓
                      Heatmap Prediction Head [35, H, W]
```

## Dataset

**FloorPlanCAD** — 15,663 CAD floor plan drawings, 35 object classes.

| Split | Samples | Crops |
|-------|---------|-------|
| train | 10,161  | 1,281,903 |
| test  | 5,502   | 549,353   |

### Classes (35)
`annotation_text`, `bathtub`, `bed`, `cabinet`, `chair`, `column`, `counter`,
`dimension_line`, `door_double`, `door_revolving`, `door_single`, `door_sliding`,
`elevator`, `escalator`, `escalator_stair`, `floor_plan_area`, `oven`, `parking`,
`plant`, `ramp`, `refrigerator`, `room_label`, `shower`, `sink`, `sofa`,
`stair`, `symbol_misc`, `table`, `toilet`, `tv`, `wall`, `washing_machine`,
`window`, `window_bay`, `window_blind`

## Project Structure

```
├── train.py                    # Main training script
├── requirements.txt            # Dependencies
│
├── src/
│   ├── data/
│   │   └── dataset.py          # PyTorch Dataset
│   └── models/
│       ├── detector.py         # Full model
│       └── blocks/
│           └── object_learning_block.py  # Mamba + Attention block
│
├── scripts/
│   ├── data/                   # Data pipeline
│   │   ├── download_gdrive.py  # Download from Google Drive
│   │   ├── build_dataset.py    # Build processed dataset
│   │   ├── generate_metadata.py # Generate bbox metadata
│   │   ├── rename_classes.py   # Fix class names
│   │   └── collect_classes.py  # Extract class list
│   └── dev/                    # Debug & visualization
│       ├── test_dataset.py
│       └── viz_heatmap.py
│
└── data/                       # (not tracked — download separately)
    ├── FloorPlanCAD_original/  # Raw dataset backup
    └── FloorPlanCAD_dataset/   # Processed dataset
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Data Preparation

```bash
# 1. Download original dataset
python scripts/data/download_gdrive.py

# 2. Build processed dataset (crops + metadata)
python scripts/data/build_dataset.py

# 3. Collect class names
python scripts/data/collect_classes.py
```

## Training

```bash
python train.py \
  --data_root ./data/FloorPlanCAD_dataset \
  --image_size 512 \
  --model_dim 512 \
  --num_blocks 4 \
  --batch_size 8 \
  --epochs 50 \
  --lr 1e-4
```
