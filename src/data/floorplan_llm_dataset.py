"""FloorPlanLLM Dataset Wrapper.

Extends FloorPlanImageDataset to provide normalized boxes in (cx, cy, w, h) format
for parallel box supervision.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.data.dataset import FloorPlanImageDataset


def normalize_boxes_to_center_form(
    boxes: torch.Tensor,
    image_size: tuple[int, int],
) -> torch.Tensor:
    """Convert corner-format boxes to normalized center format.

    Args:
        boxes: [N, 4] in (x1, y1, x2, y2) pixel coordinates
        image_size: (width, height) in pixels

    Returns:
        [N, 4] in (cx, cy, w, h) normalized to [0, 1]
    """
    if boxes.numel() == 0:
        return boxes  # Empty tensor

    width, height = image_size

    # Convert (x1, y1, x2, y2) → (cx, cy, w, h) in pixels
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1

    # Normalize to [0, 1]
    cx = cx / width
    cy = cy / height
    w = w / width
    h = h / height

    # Stack
    normalized = torch.stack([cx, cy, w, h], dim=1)

    # Clamp to valid range [0, 1]
    normalized = torch.clamp(normalized, min=0.0, max=1.0)

    return normalized


class FloorPlanLLMDataset(FloorPlanImageDataset):
    """Dataset wrapper for FloorPlanLLM training.

    Extends FloorPlanImageDataset to return:
    - image: [3, H, W]
    - stroke_tokens: [N_strokes, 12]
    - stroke_mask: [N_strokes]
    - boxes: [N_objects, 4] normalized (cx, cy, w, h) in [0, 1]
    - classes: [N_objects] class indices
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train_new",
        image_size: int = 512,
        manifest_path: str | Path | None = None,
        vector_n_max: int = 1024,
    ):
        """Initialize dataset.

        Args:
            root: Data root directory
            split: Split name ('train_new', 'val_select', etc.)
            image_size: Image size (default 512)
            manifest_path: Path to manifest file (default 'splits_v2.json')
            vector_n_max: Maximum number of SVG primitives (default 1024)
        """
        # Always enable vector branch for FloorPlanLLM
        super().__init__(
            root=root,
            split=split,
            image_size=image_size,
            manifest_path=manifest_path,
            vector_branch=True,  # Required for SVG input
            vector_n_max=vector_n_max,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get one sample.

        Returns:
            {
                'image': [3, H, W] torch.Tensor (float32),
                'stroke_tokens': [N_strokes, 12] torch.Tensor (float32),
                'stroke_mask': [N_strokes] torch.Tensor (bool),
                'boxes': [N_objects, 4] torch.Tensor (float32) normalized,
                'classes': [N_objects] torch.Tensor (long),
                'sample_id': str,
                'image_id': str,
            }
        """
        # Get base sample from parent
        sample = super().__getitem__(index)

        # Extract stroke tokens and create mask
        stroke_tokens = sample['stroke_tokens']  # [N, 12]

        # Create mask: True for valid (non-zero) primitives
        # A stroke is valid if it has non-zero values (not padding)
        stroke_mask = (stroke_tokens.abs().sum(dim=-1) > 0)  # [N]

        # Extract boxes in corner format (x1, y1, x2, y2)
        boxes_corner = sample['boxes']  # [N, 4] in pixels

        # Normalize to center format (cx, cy, w, h) in [0, 1]
        image_size_hw = sample['image_size']  # (width, height)
        boxes_normalized = normalize_boxes_to_center_form(boxes_corner, image_size_hw)

        # Return FloorPlanLLM-compatible format
        return {
            'image': sample['image'],
            'stroke_tokens': stroke_tokens,
            'stroke_mask': stroke_mask,
            'boxes': boxes_normalized,  # [N, 4] normalized
            'classes': sample['labels'],  # [N] class indices
            'sample_id': sample['sample_id'],
            'image_id': sample['image_id'],
            # Keep originals for debugging
            '_original_boxes': boxes_corner,
            '_original_size': sample['original_size'],
            '_image_size': sample['image_size'],
        }


def test_normalization():
    """Test box normalization."""
    print("Testing box normalization...")

    # Test case 1: Single box
    boxes = torch.tensor([[100.0, 150.0, 200.0, 250.0]])  # (x1, y1, x2, y2)
    image_size = (512, 512)  # (width, height)

    normalized = normalize_boxes_to_center_form(boxes, image_size)

    print(f"Input (corner): {boxes[0].tolist()}")
    print(f"Output (center, normalized): {normalized[0].tolist()}")

    # Expected:
    # cx = (100 + 200) / 2 / 512 = 0.293
    # cy = (150 + 250) / 2 / 512 = 0.391
    # w = (200 - 100) / 512 = 0.195
    # h = (250 - 150) / 512 = 0.195

    expected = torch.tensor([[0.293, 0.391, 0.195, 0.195]])
    if torch.allclose(normalized, expected, atol=0.001):
        print("✓ Normalization correct!")
    else:
        print(f"✗ Expected: {expected[0].tolist()}")

    # Test case 2: Empty boxes
    empty = torch.empty(0, 4)
    normalized_empty = normalize_boxes_to_center_form(empty, image_size)
    assert normalized_empty.shape == (0, 4), "Empty tensor shape should be preserved"
    print("✓ Empty tensor handled correctly")

    # Test case 3: Multiple boxes
    boxes_multi = torch.tensor([
        [0.0, 0.0, 512.0, 512.0],  # Full image
        [256.0, 256.0, 384.0, 384.0],  # Center patch
    ])
    normalized_multi = normalize_boxes_to_center_form(boxes_multi, image_size)
    print(f"\nMultiple boxes:")
    for i, (orig, norm) in enumerate(zip(boxes_multi, normalized_multi)):
        print(f"  Box {i}: {orig.tolist()} → {norm.tolist()}")


if __name__ == '__main__':
    test_normalization()
