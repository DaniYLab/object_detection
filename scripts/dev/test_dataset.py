"""Quick sanity test for FloorPlanDataset (per-class expanded)."""
import sys
sys.path.insert(0, ".")
from src.data.dataset import FloorPlanDataset, collate_fn, CLASS_NAMES
from torch.utils.data import DataLoader

ds = FloorPlanDataset("./data/FloorPlanCAD_dataset", split="train", image_size=512)
print(f"Dataset size : {len(ds):,}  (expanded: image × class)")

sample = ds[0]
print(f"image shape       : {sample['image'].shape}")
print(f"center_heatmap    : {sample['center_heatmap'].shape}")
print(f"size_map          : {sample['size_map'].shape}")
print(f"mask_map          : {sample['mask_map'].shape}")
print(f"text              : {sample['text']}")
print(f"class_id          : {sample['class_id']} ({CLASS_NAMES[sample['class_id']]})")
print(f"num object centers: {sample['mask_map'].sum().int().item()}")

dl = DataLoader(ds, batch_size=4, collate_fn=collate_fn, num_workers=0)
batch = next(iter(dl))
print(f"\nBatch image       : {batch['image'].shape}")
print(f"Batch heatmap     : {batch['center_heatmap'].shape}")
print(f"Batch texts       : {batch['texts'][:2]}")
print(f"Batch class_ids   : {batch['class_ids'][:4]}")
print("Dataset OK!")
