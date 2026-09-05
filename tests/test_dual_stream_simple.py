"""Minimal test for DualStreamCenterNet."""
import sys
print("1. Starting imports...", flush=True)

import torch
print("2. Torch imported", flush=True)

from src.models.dual_stream_centernet import build_dual_stream_centernet
print("3. Model imported", flush=True)

from src.data.dataset import FloorPlanQueryDataset
print("4. Dataset imported", flush=True)

# Build model
print("5. Building model...", flush=True)
model = build_dual_stream_centernet(preset='tiny', num_classes=35)
print(f"6. Model built: {sum(p.numel() for p in model.parameters()):,} params", flush=True)

# Load dataset
print("7. Loading dataset...", flush=True)
dataset = FloorPlanQueryDataset(
    root='data/FloorPlanCAD_original',
    split='train_new',
    manifest_path='data/FloorPlanCAD_original/splits_v2.json',
    vector_branch=True,
)
print(f"8. Dataset loaded: {len(dataset)} samples", flush=True)

# Get one sample
print("9. Getting first sample...", flush=True)
sample = dataset[0]
print(f"10. Sample keys: {list(sample.keys())}", flush=True)

print("\n✓ All tests passed!", flush=True)
