"""Visualize CenterNet targets: center heatmap + bounding boxes on original image."""
import sys
sys.path.insert(0, ".")
import torch
import numpy as np
from PIL import Image, ImageDraw
from src.data.dataset import FloorPlanDataset, CLASS_NAMES

ds = FloorPlanDataset("./data/FloorPlanCAD_dataset", split="train", image_size=512)

# Show first 4 samples (different classes from same or different images)
n_show = min(4, len(ds))
colors = [
    (255, 80, 80), (80, 255, 80), (80, 80, 255), (255, 255, 80),
]

for idx in range(n_show):
    sample = ds[idx]
    img_t      = sample["image"]           # [3, 512, 512], normalized
    center_hm  = sample["center_heatmap"]  # [1, 512, 512]
    size_map   = sample["size_map"]        # [2, 512, 512]
    mask_map   = sample["mask_map"]        # [1, 512, 512]
    text       = sample["text"]
    class_id   = sample["class_id"]
    sample_id  = sample["sample_id"]

    # Denormalize image
    img_np = ((img_t.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)

    # Overlay heatmap
    hm_np = center_hm[0].numpy()
    overlay = img_pil.copy().convert("RGBA")
    color = colors[idx % len(colors)]
    mask_alpha = (hm_np * 120).astype(np.uint8)
    colored = Image.new("RGBA", img_pil.size, color + (0,))
    colored.putalpha(Image.fromarray(mask_alpha, mode="L"))
    overlay = Image.alpha_composite(overlay, colored)

    # Draw bounding boxes from size_map at center locations
    draw = ImageDraw.Draw(overlay)
    centers = torch.nonzero(mask_map[0] > 0.5)  # [N, 2] — (cy, cx)
    for cy, cx in centers:
        cy, cx = cy.item(), cx.item()
        w = size_map[0, cy, cx].item()
        h = size_map[1, cy, cx].item()
        x0 = cx - w / 2
        y0 = cy - h / 2
        x1 = cx + w / 2
        y1 = cy + h / 2
        draw.rectangle([x0, y0, x1, y1], outline=color + (255,), width=2)
        draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=color + (255,))

    # Add label
    cls_name = CLASS_NAMES[class_id]
    n_obj = len(centers)
    draw.text((10, 10), f"{cls_name} ({n_obj} objects)", fill=(255, 255, 255, 255))

    result = overlay.convert("RGB")
    out_path = f"./outputs/viz_centernet_{idx}_{cls_name}.png"
    result.save(out_path)
    print(f"[{idx}] {sample_id} | {text} | {n_obj} centers → {out_path}")

print(f"\nDone! Saved {n_show} visualizations to ./outputs/")
