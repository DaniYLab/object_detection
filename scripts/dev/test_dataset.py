import sys
sys.path.insert(0, ".")
from src.data.dataset import FloorPlanDataset, collate_fn, CLASS_NAMES
from torch.utils.data import DataLoader

ds = FloorPlanDataset("./data/FloorPlanCAD_dataset", split="train", image_size=512)
print(f"Dataset size : {len(ds):,}")

sample = ds[0]
print(f"image shape  : {sample['image'].shape}")
print(f"heatmap shape: {sample['heatmap'].shape}")
print(f"classes      : {[CLASS_NAMES[i] for i in sample['class_ids'][:5]]}")
print(f"num texts    : {len(sample['texts'])}")
print(f"crops keys   : {list(sample['class_crops'].keys())[:4]}")

dl = DataLoader(ds, batch_size=4, collate_fn=collate_fn, num_workers=0)
batch = next(iter(dl))
print(f"Batch image  : {batch['image'].shape}")
print(f"Batch heatmap: {batch['heatmap'].shape}")
print("Dataset OK!")
