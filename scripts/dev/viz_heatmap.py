"""Visualize CenterNet targets and optional decoded predictions.

This utility performs inference only. It does not run an optimizer or training.
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
from src.data.constants import CLASS_NAMES
from src.data.dataset import FloorPlanDataset
from src.evaluation import CenterNetDecoder
from src.models import ModelConfig, build_model
from src.training.checkpoint import CheckpointError, load_checkpoint, restore_training_state


def denorm_image(image: torch.Tensor) -> Image.Image:
    array = ((image.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).clip(0, 255)
    return Image.fromarray(array.astype(np.uint8)).convert("RGB")


def draw_gt(
    draw: ImageDraw.ImageDraw,
    boxes: torch.Tensor,
    color: tuple[int, int, int] = (0, 255, 0),
) -> int:
    for index, box_tensor in enumerate(boxes):
        box = [float(value) for value in box_tensor]
        draw.rectangle(box, outline=color, width=3)
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        draw.ellipse(
            [center_x - 4, center_y - 4, center_x + 4, center_y + 4],
            fill=(0, 80, 255),
        )
        draw.text((box[0], max(0, box[1] - 14)), f"GT{index}", fill=color)
    return int(boxes.shape[0])


def heatmap_image(heatmap: torch.Tensor, image_size: int, mode: str = "nearest") -> Image.Image:
    upsampled = F.interpolate(
        heatmap.unsqueeze(0).unsqueeze(0),
        size=(image_size, image_size),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )[0, 0].numpy()
    array = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    array[..., 0] = (upsampled * 255).clip(0, 255).astype(np.uint8)
    array[..., 1] = (upsampled * 80).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array)


def overlay_heatmap(base: Image.Image, heatmap: torch.Tensor, image_size: int) -> Image.Image:
    upsampled = F.interpolate(
        heatmap.unsqueeze(0).unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()
    red = Image.new("RGBA", base.size, (255, 0, 0, 0))
    red.putalpha(
        Image.fromarray((upsampled * 170).clip(0, 170).astype(np.uint8), mode="L")
    )
    return Image.alpha_composite(base.convert("RGBA"), red).convert("RGB")


def draw_predictions(
    draw: ImageDraw.ImageDraw,
    prediction: dict,
    class_id: int,
    color: tuple[int, int, int] = (255, 80, 80),
) -> int:
    keep = prediction["labels"].detach().cpu().eq(class_id)
    boxes = prediction["boxes"].detach().cpu()[keep]
    scores = prediction["scores"].detach().cpu()[keep]
    for box_tensor, score_tensor in zip(boxes, scores):
        box = [float(value) for value in box_tensor]
        draw.rectangle(box, outline=color, width=2)
        draw.text(
            (box[0], max(0, box[1] - 12)),
            f"{float(score_tensor):.2f}",
            fill=color,
        )
    return int(boxes.shape[0])


def _checkpoint_model(path: str, image_size: int) -> tuple[torch.nn.Module, ModelConfig, dict]:
    checkpoint = load_checkpoint(path, map_location="cpu")
    config_value = checkpoint.get("model_config")
    if not isinstance(config_value, dict):
        raise CheckpointError("Checkpoint does not contain a serializable model_config")
    config = ModelConfig.from_dict(config_value)
    if config.image_size != image_size:
        raise ValueError(
            f"Visualization image_size={image_size} does not match checkpoint "
            f"image_size={config.image_size}"
        )
    model = build_model(config)
    restore_training_state(
        checkpoint,
        model=model,
        expected_model_config=config,
        expected_class_names=CLASS_NAMES,
        expected_output_stride=config.output_stride,
        weights_only=True,
    )
    model.eval()
    return model, config, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "--data_root", dest="data_root", default="./data/FloorPlanCAD_original")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--image-size", "--image_size", dest="image_size", type=int, default=512)
    parser.add_argument("--output-stride", "--output_stride", dest="output_stride", type=int, default=8)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--pred", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--out-dir", "--out_dir", dest="out_dir", default="./outputs")
    args = parser.parse_args()

    dataset = FloorPlanDataset(
        args.data_root,
        split=args.split,
        image_size=args.image_size,
        output_stride=args.output_stride,
        manifest_path=args.manifest,
    )
    sample = dataset[args.index]
    class_id = sample["class_id"]
    class_name = CLASS_NAMES[class_id]
    heatmap = sample["center_heatmap"][0]

    original = denorm_image(sample["image"])
    original_boxes = original.copy()
    num_gt = draw_gt(ImageDraw.Draw(original_boxes), sample["boxes"])
    heat_only = heatmap_image(heatmap, args.image_size, mode="nearest")

    overlay = overlay_heatmap(original.convert("L").convert("RGB"), heatmap, args.image_size)
    draw = ImageDraw.Draw(overlay)
    draw_gt(draw, sample["boxes"])

    num_predictions = 0
    if args.pred:
        if not args.checkpoint:
            raise ValueError("--pred requires --checkpoint")
        model, config, _checkpoint = _checkpoint_model(args.checkpoint, args.image_size)
        with torch.inference_mode():
            outputs = model(sample["image"].unsqueeze(0))
        decoder = CenterNetDecoder(
            stride=config.output_stride,
            threshold=args.threshold,
            topk=args.topk,
        )
        prediction = decoder(
            outputs,
            [sample["image_id"]],
            image_size=(args.image_size, args.image_size),
        )[0]
        num_predictions = draw_predictions(draw, prediction, class_id)

    width, height = original.size
    header = 30
    canvas = Image.new("RGB", (width * 3, height + header), (20, 20, 20))
    canvas.paste(original_boxes, (0, header))
    canvas.paste(heat_only, (width, header))
    canvas.paste(overlay, (width * 2, header))
    labels = ImageDraw.Draw(canvas)
    labels.text((10, 8), "Original + all GT boxes", fill=(255, 255, 255))
    labels.text((width + 10, 8), "Query target heatmap", fill=(255, 255, 255))
    labels.text((width * 2 + 10, 8), "GT (green) + decoded predictions", fill=(255, 255, 255))
    labels.text(
        (10, height + header - 18),
        f"{sample['image_id']} | {class_name} | GT={num_gt} | Pred={num_predictions}",
        fill=(255, 255, 255),
    )

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"viz_centernet_{args.split}_{args.index}_{class_name}.png"
    canvas.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
