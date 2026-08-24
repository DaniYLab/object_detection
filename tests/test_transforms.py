from __future__ import annotations

import random

import torch
from PIL import Image

from src.data.transforms import (
    PairedTrainTransform,
    ResizeNormalize,
    horizontal_flip_boxes,
    rotate_boxes_90,
    vertical_flip_boxes,
)


def test_half_open_flip_box_math() -> None:
    boxes = torch.tensor([[2.0, 3.0, 12.0, 15.0]])
    assert torch.equal(
        horizontal_flip_boxes(boxes, width=40),
        torch.tensor([[28.0, 3.0, 38.0, 15.0]]),
    )
    assert torch.equal(
        vertical_flip_boxes(boxes, height=30),
        torch.tensor([[2.0, 15.0, 12.0, 27.0]]),
    )


def test_rectangular_90_degree_rotation_updates_size_and_boxes() -> None:
    boxes = torch.tensor([[2.0, 5.0, 12.0, 20.0]])
    rotated, new_size = rotate_boxes_90(boxes, (40, 30), k=1)
    assert new_size == (30, 40)
    assert torch.equal(rotated, torch.tensor([[10.0, 2.0, 25.0, 12.0]]))

    restored, restored_size = rotate_boxes_90(rotated, new_size, k=3)
    assert restored_size == (40, 30)
    assert torch.equal(restored, boxes)


def test_forced_train_flip_is_paired_and_photometric_does_not_move_boxes() -> None:
    image = Image.new("RGB", (40, 30), (100, 120, 140))
    boxes = [[2, 5, 12, 20]]
    transform = PairedTrainTransform(
        image_size=(30, 40),
        horizontal_flip_probability=1.0,
        vertical_flip_probability=0.0,
        rotation_probability=0.0,
        photometric_probability=1.0,
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        rng=random.Random(9),
    )
    tensor, transformed_boxes = transform(image, boxes)
    assert tensor.shape == (3, 30, 40)
    assert torch.equal(
        transformed_boxes,
        torch.tensor([[28.0, 5.0, 38.0, 20.0]]),
    )


def test_eval_resize_normalize_is_deterministic() -> None:
    image = Image.new("RGB", (20, 10), (64, 128, 255))
    boxes = [[2, 1, 10, 8]]
    transform = ResizeNormalize(image_size=40)
    tensor_a, boxes_a = transform(image, boxes)
    tensor_b, boxes_b = transform(image, boxes)
    assert torch.equal(tensor_a, tensor_b)
    assert torch.equal(boxes_a, boxes_b)
    assert torch.equal(boxes_a, torch.tensor([[4.0, 4.0, 20.0, 32.0]]))
