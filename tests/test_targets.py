from __future__ import annotations

import math

import pytest
import torch

from src.data.targets import (
    TargetCollisionError,
    build_centernet_targets,
    gaussian_radius,
    generate_centernet_targets,
)


def test_standard_gaussian_radius_and_exact_peak() -> None:
    radius = gaussian_radius((10, 20), min_overlap=0.7)
    assert math.isclose(radius, 3.6779253585061333, rel_tol=1e-6)

    targets = build_centernet_targets(
        [[0, 0, 64, 64]], image_size=(64, 64), output_stride=8
    )
    assert targets["center_heatmap"].shape == (1, 8, 8)
    assert targets["center_heatmap"][0, 4, 4] == 1
    assert targets["mask_map"][0, 4, 4] == 1
    assert torch.equal(targets["size_map"][:, 4, 4], torch.tensor([8.0, 8.0]))
    assert torch.equal(targets["offset_map"][:, 4, 4], torch.tensor([0.0, 0.0]))


def test_half_open_right_bottom_boundary_is_valid() -> None:
    targets, stats = generate_centernet_targets(
        [[48, 48, 64, 64]], image_size=(64, 64), output_stride=8
    )
    assert stats.valid_boxes == 1
    assert stats.out_of_bounds_boxes == 0
    assert targets["mask_map"].sum() == 1
    assert targets["mask_map"][0, 7, 7] == 1


def test_invalid_and_out_of_bounds_boxes_are_counted() -> None:
    _, stats = generate_centernet_targets(
        [
            [1, 1, 1, 2],
            [-1, 0, 3, 3],
            [0, 0, float("nan"), 3],
            [0, 0, 4, 4],
        ],
        image_size=(8, 8),
        output_stride=2,
    )
    assert stats.total_boxes == 4
    assert stats.valid_boxes == 1
    assert stats.invalid_boxes == 2
    assert stats.out_of_bounds_boxes == 1
    assert stats.skipped_boxes == 3


def test_collision_policies_largest_first_and_error() -> None:
    # Both centers map to cell (2, 2), but the second box has larger area.
    boxes = [[16, 16, 24, 24], [8, 8, 32, 32]]
    largest, largest_stats = generate_centernet_targets(
        boxes,
        image_size=(64, 64),
        output_stride=8,
        collision_policy="largest",
    )
    assert largest_stats.collisions == 1
    assert largest_stats.replacements == 1
    assert torch.equal(largest["size_map"][:, 2, 2], torch.tensor([3.0, 3.0]))

    first, first_stats = generate_centernet_targets(
        boxes,
        image_size=(64, 64),
        output_stride=8,
        collision_policy="first",
    )
    assert first_stats.ignored_collisions == 1
    assert torch.equal(first["size_map"][:, 2, 2], torch.tensor([1.0, 1.0]))

    with pytest.raises(TargetCollisionError):
        generate_centernet_targets(
            boxes,
            image_size=(64, 64),
            output_stride=8,
            collision_policy="error",
        )
