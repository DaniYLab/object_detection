"""Visualize heatmap overlay on original image."""
import sys
sys.path.insert(0, ".")
import torch
import numpy as np
from PIL import Image, ImageDraw
from src.data.dataset import FloorPlanDataset, CLASS_NAMES

ds = FloorPlanDataset("./data/FloorPlanCAD_dataset", split="train", image_size=512)

# Find sample 0 with metadata
sample = ds[0]
img_t   = sample["image"]           # [3, 512, 512], normalized
heatmap = sample["heatmap"]         # [35, 512, 512]
classes = [CLASS_NAMES[i] for i in sample["class_ids"]]

# Denormalize image
img_np = ((img_t.permute(1,2,0).numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
img_pil = Image.fromarray(img_np)

# Overlay all active class heatmaps
colors = [
    (255,80,80), (80,255,80), (80,80,255), (255,255,80), (255,80,255),
    (80,255,255), (255,160,80), (160,80,255), (80,255,160), (255,80,160),
    (160,255,80), (80,160,255),
]

overlay = img_pil.copy().convert("RGBA")
for i, cid in enumerate(sample["class_ids"][:12]):
    mask = heatmap[cid].numpy()
    if mask.max() < 0.01:
        continue
    color = colors[i % len(colors)]
    mask_img = Image.fromarray((mask * 80).astype(np.uint8), mode="L")
    colored = Image.new("RGBA", img_pil.size, color + (0,))
    colored.putalpha(mask_img)
    overlay = Image.alpha_composite(overlay, colored)

# Draw class labels
draw = ImageDraw.Draw(overlay)
for i, cid in enumerate(sample["class_ids"][:12]):
    mask = heatmap[cid].numpy()
    if mask.max() < 0.01:
        continue
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        continue
    cx, cy = int(xs.mean()), int(ys.mean())
    cls_name = CLASS_NAMES[cid]
    draw.text((cx, cy), cls_name[:10], fill=colors[i % len(colors)] + (255,))

result = overlay.convert("RGB")
result.save("./heatmap_viz.png")
print(f"Saved heatmap_viz.png")
print(f"Active classes: {classes}")

# Check non-zero ratio per class
print("\nHeatmap coverage per class:")
for cid in sample["class_ids"]:
    m = heatmap[cid]
    pct = m.sum().item() / (512*512) * 100
    print(f"  {CLASS_NAMES[cid]:22s}: {pct:.1f}% coverage")
