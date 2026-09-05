from src.data.dataset import FloorPlanQueryDataset, query_collate_fn
from torch.utils.data import DataLoader

print("Loading dataset...")
dataset = FloorPlanQueryDataset(
    root='data/FloorPlanCAD_original',
    split='train_new',
    manifest_path='data/FloorPlanCAD_original/splits_v2.json',
    vector_branch=False,
)
print(f"Dataset size: {len(dataset)}")

print("\nGetting one sample...")
sample = dataset[0]
print(f"Sample keys: {sample.keys()}")
print(f"Image shape: {sample['image'].shape}")
print(f"Center heatmap shape: {sample['center_heatmap'].shape}")

print("\nCreating dataloader...")
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,
    collate_fn=query_collate_fn,
)

print("Getting one batch...")
batch = next(iter(dataloader))
print(f"Batch keys: {batch.keys()}")
print(f"Batch image shape: {batch['image'].shape}")

print("OK - Dataset works!")
