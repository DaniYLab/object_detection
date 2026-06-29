"""Visualize CenterNet targets and optional predictions.

Outputs a triptych:
  1. Original + GT boxes
  2. Heatmap only
  3. Grayscale overlay + GT/pred boxes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, ".")
from src.data.dataset import FloorPlanDataset, CLASS_NAMES
from src.models.detector import FloorPlanDetector


def denorm_image(img_t: torch.Tensor) -> Image.Image:
    img_np = ((img_t.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_np).convert("RGB")


def draw_gt(draw: ImageDraw.ImageDraw, sample: dict, stride: int, color=(0, 255, 0)) -> int:
    mask = sample["mask_map"][0]
    size = sample["size_map"]
    offset = sample["offset_map"]
    centers = torch.nonzero(mask > 0.5)
    for k, (cy, cx) in enumerate(centers):
        cy_i, cx_i = int(cy), int(cx)
        off_x = float(offset[0, cy_i, cx_i])
        off_y = float(offset[1, cy_i, cx_i])
        w = float(size[0, cy_i, cx_i]) * stride
        h = float(size[1, cy_i, cx_i]) * stride
        cx_img = (cx_i + off_x) * stride
        cy_img = (cy_i + off_y) * stride
        box = [cx_img - w / 2, cy_img - h / 2, cx_img + w / 2, cy_img + h / 2]
        draw.rectangle(box, outline=color, width=3)
        draw.ellipse([cx_img - 4, cy_img - 4, cx_img + 4, cy_img + 4], fill=(0, 80, 255))
        draw.text((box[0], max(0, box[1] - 14)), f"GT{k}", fill=color)
    return len(centers)


def heatmap_image(hm: torch.Tensor, image_size: int, mode: str = "nearest") -> Image.Image:
    hm_up = F.interpolate(
        hm.unsqueeze(0).unsqueeze(0),
        size=(image_size, image_size),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )[0, 0].numpy()
    heat_np = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    heat_np[..., 0] = (hm_up * 255).clip(0, 255).astype(np.uint8)
    heat_np[..., 1] = (hm_up * 80).clip(0, 255).astype(np.uint8)
    return Image.fromarray(heat_np)


def overlay_heatmap(base: Image.Image, hm: torch.Tensor, image_size: int) -> Image.Image:
    hm_up = F.interpolate(
        hm.unsqueeze(0).unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()
    red = Image.new("RGBA", base.size, (255, 0, 0, 0))
    red.putalpha(Image.fromarray((hm_up * 170).clip(0, 170).astype(np.uint8), mode="L"))
    return Image.alpha_composite(base.convert("RGBA"), red).convert("RGB")


def local_maxima(hm: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    pad = kernel // 2
    pooled = F.max_pool2d(hm.unsqueeze(0).unsqueeze(0), kernel, stride=1, padding=pad)[0, 0]
    return hm * (hm == pooled)


def draw_predictions(
    draw: ImageDraw.ImageDraw,
    preds: dict[str, torch.Tensor],
    class_id: int,
    stride: int,
    threshold: float,
    topk: int,
    color=(255, 80, 80),
) -> int:
    hm = preds["center_heatmap"][0, class_id].detach().cpu()
    size = preds["size_map"][0, class_id * 2 : class_id * 2 + 2].detach().cpu()
    offset = preds["offset_map"][0, class_id * 2 : class_id * 2 + 2].detach().cpu()

    peaks = local_maxima(hm)
    scores, inds = torch.topk(peaks.flatten(), k=min(topk, peaks.numel()))
    drawn = 0
    width = hm.shape[-1]
    for score, ind in zip(scores, inds):
        if float(score) < threshold:
            continue
        cy = int(ind // width)
        cx = int(ind % width)
        off_x = float(offset[0, cy, cx])
        off_y = float(offset[1, cy, cx])
        w = float(size[0, cy, cx]) * stride
        h = float(size[1, cy, cx]) * stride
        cx_img = (cx + off_x) * stride
        cy_img = (cy + off_y) * stride
        box = [cx_img - w / 2, cy_img - h / 2, cx_img + w / 2, cy_img + h / 2]
        draw.rectangle(box, outline=color, width=2)
        draw.text((box[0], max(0, box[1] - 12)), f"{float(score):.2f}", fill=color)
        drawn += 1
    return drawn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./data/FloorPlanCAD_original")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--output_stride", type=int, default=8)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--pred", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--out_dir", default="./outputs")
    args = parser.parse_args()

    ds = FloorPlanDataset(
        args.data_root,
        split=args.split,
        image_size=args.image_size,
        output_stride=args.output_stride,
    )
    sample = ds[args.index]
    cls_name = CLASS_NAMES[sample["class_id"]]
    hm = sample["center_heatmap"][0]

    original = denorm_image(sample["image"])
    original_boxes = original.copy()
    n_gt = draw_gt(ImageDraw.Draw(original_boxes), sample, stride=args.output_stride)

    heat_only = heatmap_image(hm, args.image_size, mode="nearest")

    gray = original.convert("L").convert("RGB")
    overlay = overlay_heatmap(gray, hm, args.image_size)
    draw = ImageDraw.Draw(overlay)
    draw_gt(draw, sample, stride=args.output_stride)

    n_pred = 0
    if args.pred:
        if not args.checkpoint:
            raise ValueError("--pred requires --checkpoint")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model_args = ckpt.get("args", {})
        model = FloorPlanDetector(
            image_size=args.image_size,
            model_dim=int(model_args.get("model_dim", 256)),
            num_classes=len(CLASS_NAMES),
            depth_per_class=int(model_args.get("depth_per_class", 2)),
            fusion_mode=model_args.get("fusion_mode", "film"),
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        with torch.no_grad():
            preds = model(sample["image"].unsqueeze(0))
        n_pred = draw_predictions(
            draw,
            preds,
            class_id=sample["class_id"],
            stride=args.output_stride,
            threshold=args.threshold,
            topk=args.topk,
        )

    # Compose triptych.
    w, h = original.size
    header = 30
    canvas = Image.new("RGB", (w * 3, h + header), (20, 20, 20))
    canvas.paste(original_boxes, (0, header))
    canvas.paste(heat_only, (w, header))
    canvas.paste(overlay, (w * 2, header))
    text = ImageDraw.Draw(canvas)
    text.text((10, 8), "Original + GT boxes (green), centers blue", fill=(255, 255, 255))
    text.text((w + 10, 8), "Heatmap only (red, nearest upsample)", fill=(255, 255, 255))
    text.text((w * 2 + 10, 8), "Grayscale overlay + GT/pred boxes", fill=(255, 255, 255))
    text.text((10, h + header - 18), f"{sample['sample_id']} | {cls_name} | GT={n_gt} | Pred={n_pred}", fill=(255, 255, 255))

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out_dir) / f"viz_centernet_{args.split}_{args.index}_{cls_name}.png"
    canvas.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
