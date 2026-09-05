"""Diagnose DualStreamCenterNet predictions."""

import torch
from torch.utils.data import DataLoader

from src.data.dataset import FloorPlanQueryDataset, query_collate_fn
from src.models.dual_stream_centernet import build_dual_stream_centernet

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_dual_stream_centernet(preset='small', num_classes=35)
checkpoint = torch.load('checkpoints/dual_stream_centernet/epoch_10.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Load one batch
dataset = FloorPlanQueryDataset(
    root='data/FloorPlanCAD_original',
    split='val_select',
    manifest_path='data/FloorPlanCAD_original/splits_v2.json',
    vector_branch=True,
)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=query_collate_fn)

batch = next(iter(dataloader))

# Forward pass
with torch.no_grad():
    image = batch['image'].to(device)
    stroke_tokens = batch['stroke_tokens'].to(device)
    stroke_mask = batch['stroke_mask'].to(device)
    class_id = batch['class_ids'][0]

    output = model(image, stroke_tokens, stroke_mask, class_id=class_id)

    # Check output statistics
    heatmap = output['heatmap']  # Should be logits
    size_map = output['size']
    offset_map = output['offset']

    # Apply sigmoid to get probabilities
    heatmap_probs = torch.sigmoid(heatmap)

    print("=" * 60)
    print("DualStreamCenterNet Diagnostic")
    print("=" * 60)
    print(f"Class ID: {class_id}")
    print(f"Batch size: {image.shape[0]}")
    print(f"Image shape: {image.shape}")
    print(f"Stroke tokens shape: {stroke_tokens.shape}")
    print()
    print("Heatmap (logits):")
    print(f"  Shape: {heatmap.shape}")
    print(f"  Min: {heatmap.min().item():.4f}")
    print(f"  Max: {heatmap.max().item():.4f}")
    print(f"  Mean: {heatmap.mean().item():.4f}")
    print(f"  Std: {heatmap.std().item():.4f}")
    print()
    print("Heatmap (probabilities after sigmoid):")
    print(f"  Min: {heatmap_probs.min().item():.6f}")
    print(f"  Max: {heatmap_probs.max().item():.6f}")
    print(f"  Mean: {heatmap_probs.mean().item():.6f}")
    print(f"  Std: {heatmap_probs.std().item():.6f}")
    print(f"  >0.05: {(heatmap_probs > 0.05).sum().item()} pixels")
    print(f"  >0.1: {(heatmap_probs > 0.1).sum().item()} pixels")
    print(f"  >0.3: {(heatmap_probs > 0.3).sum().item()} pixels")
    print(f"  >0.5: {(heatmap_probs > 0.5).sum().item()} pixels")
    print()
    print("Size map:")
    print(f"  Shape: {size_map.shape}")
    print(f"  Min: {size_map.min().item():.4f}")
    print(f"  Max: {size_map.max().item():.4f}")
    print(f"  Mean: {size_map.mean().item():.4f}")
    print()
    print("Offset map:")
    print(f"  Shape: {offset_map.shape}")
    print(f"  Min: {offset_map.min().item():.4f}")
    print(f"  Max: {offset_map.max().item():.4f}")
    print(f"  Mean: {offset_map.mean().item():.4f}")
    print()

    # Check ground truth
    gt_heatmap = batch['center_heatmap'][0]
    print("Ground truth heatmap:")
    print(f"  Shape: {gt_heatmap.shape}")
    print(f"  Positive pixels: {(gt_heatmap > 0.5).sum().item()}")
    print(f"  Max: {gt_heatmap.max().item():.4f}")
    print()

    # Check if there are any detections at different thresholds
    print("Detection counts at different thresholds:")
    for thresh in [0.01, 0.05, 0.1, 0.3, 0.5]:
        count = (heatmap_probs > thresh).sum().item()
        print(f"  Threshold {thresh:.2f}: {count} pixels across batch")
    print()
    print("=" * 60)
