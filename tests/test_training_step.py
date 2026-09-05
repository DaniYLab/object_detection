import torch
from src.models.hybrid_parallel_detector import build_hybrid_parallel_detector
from src.data.dataset import FloorPlanQueryDataset, query_collate_fn
from torch.utils.data import DataLoader

print("Building model...")
model = build_hybrid_parallel_detector(num_classes=35, use_reasoning=False)
model.eval()

print("Loading dataset...")
dataset = FloorPlanQueryDataset(
    root='data/FloorPlanCAD_original',
    split='train_new',
    manifest_path='data/FloorPlanCAD_original/splits_v2.json',
    vector_branch=False,
)

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,
    collate_fn=query_collate_fn,
)

print("Getting batch...")
batch = next(iter(dataloader))

print("Forward pass...")
image = batch['image']
output = model(image)

print("\nOutput shapes:")
for k, v in output.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {v.shape}")

print("\nTarget shapes:")
print(f"  center_heatmap: {batch['center_heatmap'].shape}")
print(f"  size_map: {batch['size_map'].shape}")
print(f"  offset_map: {batch['offset_map'].shape}")

print("\nComputing loss...")
from src.training.losses import focal_loss, l1_loss_masked

# Prepare targets for all classes
B, num_classes = output['heatmap'].shape[:2]
target_heatmap = torch.zeros(B, num_classes, 64, 64)
target_size = torch.zeros(B, num_classes, 2, 64, 64)
target_offset = torch.zeros(B, num_classes, 2, 64, 64)

# Fill in for actual class
for i in range(B):
    class_id = batch['class_ids'][i]
    target_heatmap[i, class_id] = batch['center_heatmap'][i, 0]
    target_size[i, class_id] = batch['size_map'][i, 0]
    target_offset[i, class_id] = batch['offset_map'][i, 0]

print("Computing focal loss...")
heatmap_loss = focal_loss(
    pred=None,
    target=target_heatmap,
    logits=output['heatmap'],
    alpha=2.0,
    beta=4.0
)
print(f"Heatmap loss: {heatmap_loss.item():.4f}")

print("\nTraining step works!")
