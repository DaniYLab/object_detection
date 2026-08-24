"""CenterNet target generation with explicit half-open box semantics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

import torch

CollisionPolicy = Literal["largest", "first", "error"]
COLLISION_POLICIES: tuple[str, ...] = ("largest", "first", "error")


@dataclass
class TargetStats:
    """Counters describing how source boxes were encoded."""

    total_boxes: int = 0
    valid_boxes: int = 0
    encoded_boxes: int = 0
    invalid_boxes: int = 0
    out_of_bounds_boxes: int = 0
    collisions: int = 0
    replacements: int = 0
    ignored_collisions: int = 0

    @property
    def skipped_boxes(self) -> int:
        return self.invalid_boxes + self.out_of_bounds_boxes

    @property
    def num_collisions(self) -> int:
        return self.collisions

    @property
    def unsupervised_boxes(self) -> int:
        """Valid objects without a distinct size/offset cell due to collision."""

        return self.collisions

    @property
    def regression_targets(self) -> int:
        return self.valid_boxes - self.collisions

    @property
    def collision_rate(self) -> float:
        return self.collisions / self.valid_boxes if self.valid_boxes else 0.0

    def to_dict(self) -> dict[str, int | float]:
        result = asdict(self)
        result["skipped_boxes"] = self.skipped_boxes
        result["unsupervised_boxes"] = self.unsupervised_boxes
        result["regression_targets"] = self.regression_targets
        result["collision_rate"] = self.collision_rate
        return result


class TargetCollisionError(ValueError):
    """Raised when two box centers collide under the ``error`` policy."""


def gaussian_radius(
    box_size: Sequence[float],
    min_overlap: float = 0.7,
) -> float:
    """Standard CornerNet/CenterNet IoU-based Gaussian radius.

    ``box_size`` is ``(height, width)`` in output-grid units.
    """

    if len(box_size) != 2:
        raise ValueError("box_size must be (height, width)")
    height, width = (float(value) for value in box_size)
    if not (math.isfinite(height) and math.isfinite(width)) or height <= 0 or width <= 0:
        raise ValueError("box dimensions must be finite and positive")
    if not math.isfinite(min_overlap) or not 0 < min_overlap < 1:
        raise ValueError("min_overlap must be between 0 and 1")

    a1 = 1.0
    b1 = height + width
    c1 = width * height * (1.0 - min_overlap) / (1.0 + min_overlap)
    radius1 = (b1 + math.sqrt(max(0.0, b1 * b1 - 4.0 * a1 * c1))) / 2.0

    a2 = 4.0
    b2 = 2.0 * (height + width)
    c2 = (1.0 - min_overlap) * width * height
    radius2 = (b2 + math.sqrt(max(0.0, b2 * b2 - 4.0 * a2 * c2))) / 2.0

    a3 = 4.0 * min_overlap
    b3 = -2.0 * min_overlap * (height + width)
    c3 = (min_overlap - 1.0) * width * height
    radius3 = (b3 + math.sqrt(max(0.0, b3 * b3 - 4.0 * a3 * c3))) / 2.0

    return min(radius1, radius2, radius3)


def gaussian2d(
    shape: tuple[int, int],
    sigma: float = 1.0,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build a centered 2-D Gaussian using CenterNet's diameter/6 sigma."""

    height, width = shape
    if height <= 0 or width <= 0 or sigma <= 0:
        raise ValueError("Gaussian shape and sigma must be positive")
    y = torch.arange(height, dtype=dtype, device=device) - (height - 1) / 2
    x = torch.arange(width, dtype=dtype, device=device) - (width - 1) / 2
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    gaussian = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    threshold = torch.finfo(gaussian.dtype).eps * gaussian.max()
    gaussian = torch.where(gaussian < threshold, torch.zeros_like(gaussian), gaussian)
    return gaussian


def draw_gaussian(
    heatmap: torch.Tensor,
    center: tuple[int, int],
    radius: int,
    *,
    value: float = 1.0,
) -> torch.Tensor:
    """Draw a max-composited CenterNet Gaussian in place and return heatmap."""

    if heatmap.ndim != 2:
        raise ValueError("heatmap must be two-dimensional")
    radius = max(0, int(radius))
    diameter = 2 * radius + 1
    gaussian = gaussian2d(
        (diameter, diameter),
        sigma=diameter / 6.0,
        dtype=heatmap.dtype,
        device=heatmap.device,
    ) * float(value)

    center_x, center_y = (int(center[0]), int(center[1]))
    height, width = heatmap.shape
    if not 0 <= center_x < width or not 0 <= center_y < height:
        raise ValueError("Gaussian center is outside the heatmap")

    left = min(center_x, radius)
    right = min(width - center_x, radius + 1)
    top = min(center_y, radius)
    bottom = min(height - center_y, radius + 1)
    heatmap_slice = heatmap[
        center_y - top : center_y + bottom,
        center_x - left : center_x + right,
    ]
    gaussian_slice = gaussian[
        radius - top : radius + bottom,
        radius - left : radius + right,
    ]
    torch.maximum(heatmap_slice, gaussian_slice, out=heatmap_slice)
    # Preserve an exact positive target despite floating-point kernel generation.
    heatmap[center_y, center_x] = max(float(heatmap[center_y, center_x]), float(value))
    return heatmap


def _spatial_size(value: int | Sequence[int], name: str) -> tuple[int, int]:
    if isinstance(value, int):
        result = (value, value)
    elif len(value) == 2:
        result = (int(value[0]), int(value[1]))
    else:
        raise ValueError(f"{name} must be an int or (height, width)")
    if result[0] <= 0 or result[1] <= 0:
        raise ValueError(f"{name} dimensions must be positive")
    return result


def _as_box_tensor(boxes: torch.Tensor | Sequence[Sequence[float]]) -> torch.Tensor:
    tensor = torch.as_tensor(boxes, dtype=torch.float32)
    if tensor.numel() == 0:
        return tensor.reshape(0, 4)
    if tensor.ndim != 2 or tensor.shape[1] != 4:
        raise ValueError("boxes must have shape [N, 4]")
    return tensor


def generate_centernet_targets(
    boxes: torch.Tensor | Sequence[Sequence[float]],
    *,
    image_size: int | Sequence[int],
    output_stride: int = 8,
    output_size: int | Sequence[int] | None = None,
    min_overlap: float = 0.7,
    collision_policy: CollisionPolicy = "largest",
) -> tuple[dict[str, torch.Tensor], TargetStats]:
    """Encode input-pixel ``xyxy`` half-open boxes into CenterNet maps.

    ``image_size`` and ``output_size`` use ``(height, width)`` order. Boxes must
    satisfy ``0 <= x0 < x1 <= width`` and ``0 <= y0 < y1 <= height``. Invalid
    boxes are counted and skipped rather than silently clipped.
    """

    image_h, image_w = _spatial_size(image_size, "image_size")
    if not isinstance(output_stride, int) or output_stride <= 0:
        raise ValueError("output_stride must be a positive integer")
    if output_size is None:
        if image_h % output_stride != 0 or image_w % output_stride != 0:
            raise ValueError("image dimensions must be divisible by output_stride")
        output_h, output_w = image_h // output_stride, image_w // output_stride
    else:
        output_h, output_w = _spatial_size(output_size, "output_size")
    if collision_policy not in COLLISION_POLICIES:
        raise ValueError(
            f"Unknown collision_policy {collision_policy!r}; expected one of {COLLISION_POLICIES}"
        )
    if not math.isfinite(min_overlap) or not 0 < min_overlap < 1:
        raise ValueError("min_overlap must be between 0 and 1")

    box_tensor = _as_box_tensor(boxes)
    stats = TargetStats(total_boxes=len(box_tensor))
    heatmap = torch.zeros((1, output_h, output_w), dtype=torch.float32)
    size_map = torch.zeros((2, output_h, output_w), dtype=torch.float32)
    offset_map = torch.zeros((2, output_h, output_w), dtype=torch.float32)
    mask_map = torch.zeros((1, output_h, output_w), dtype=torch.float32)
    area_map = torch.full((output_h, output_w), -1.0, dtype=torch.float32)

    scale_x = output_w / image_w
    scale_y = output_h / image_h
    for box_index, box in enumerate(box_tensor):
        if not bool(torch.isfinite(box).all()):
            stats.invalid_boxes += 1
            continue
        x0, y0, x1, y1 = (float(value) for value in box.tolist())
        if x1 <= x0 or y1 <= y0:
            stats.invalid_boxes += 1
            continue
        if x0 < 0 or y0 < 0 or x1 > image_w or y1 > image_h:
            stats.out_of_bounds_boxes += 1
            continue
        stats.valid_boxes += 1

        output_x0 = x0 * scale_x
        output_y0 = y0 * scale_y
        output_x1 = x1 * scale_x
        output_y1 = y1 * scale_y
        box_w = output_x1 - output_x0
        box_h = output_y1 - output_y0
        center_x_float = (output_x0 + output_x1) / 2.0
        center_y_float = (output_y0 + output_y1) / 2.0
        center_x = math.floor(center_x_float)
        center_y = math.floor(center_y_float)
        # Valid half-open boxes guarantee centers are inside, modulo precision.
        center_x = min(output_w - 1, max(0, center_x))
        center_y = min(output_h - 1, max(0, center_y))

        radius = max(
            0,
            int(gaussian_radius((math.ceil(box_h), math.ceil(box_w)), min_overlap=min_overlap)),
        )
        draw_gaussian(heatmap[0], (center_x, center_y), radius)

        area = box_w * box_h
        occupied = bool(mask_map[0, center_y, center_x].item())
        should_write = not occupied
        if occupied:
            stats.collisions += 1
            if collision_policy == "error":
                raise TargetCollisionError(
                    f"boxes collide at output cell ({center_x}, {center_y}); "
                    f"box index {box_index} conflicts with an earlier box"
                )
            if collision_policy == "largest" and area > float(area_map[center_y, center_x]):
                should_write = True
                stats.replacements += 1
            else:
                stats.ignored_collisions += 1

        if should_write:
            size_map[0, center_y, center_x] = box_w
            size_map[1, center_y, center_x] = box_h
            offset_map[0, center_y, center_x] = center_x_float - center_x
            offset_map[1, center_y, center_x] = center_y_float - center_y
            mask_map[0, center_y, center_x] = 1.0
            area_map[center_y, center_x] = area
        stats.encoded_boxes += 1

    targets = {
        "center_heatmap": heatmap,
        "size_map": size_map,
        "offset_map": offset_map,
        "mask_map": mask_map,
    }
    return targets, stats


def build_centernet_targets(
    boxes: torch.Tensor | Sequence[Sequence[float]],
    *,
    image_size: int | Sequence[int],
    output_stride: int = 8,
    output_size: int | Sequence[int] | None = None,
    min_overlap: float = 0.7,
    collision_policy: CollisionPolicy = "largest",
    return_stats: bool = False,
) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], TargetStats]:
    """Convenience wrapper returning maps, optionally together with statistics."""

    targets, stats = generate_centernet_targets(
        boxes,
        image_size=image_size,
        output_stride=output_stride,
        output_size=output_size,
        min_overlap=min_overlap,
        collision_policy=collision_policy,
    )
    return (targets, stats) if return_stats else targets


@dataclass(frozen=True)
class CenterNetTargetBuilder:
    """Reusable target-builder configuration."""

    image_size: int | tuple[int, int]
    output_stride: int = 8
    output_size: int | tuple[int, int] | None = None
    min_overlap: float = 0.7
    collision_policy: CollisionPolicy = "largest"

    def __call__(
        self,
        boxes: torch.Tensor | Sequence[Sequence[float]],
        *,
        return_stats: bool = False,
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], TargetStats]:
        return build_centernet_targets(
            boxes,
            image_size=self.image_size,
            output_stride=self.output_stride,
            output_size=self.output_size,
            min_overlap=self.min_overlap,
            collision_policy=self.collision_policy,
            return_stats=return_stats,
        )


# Common short names for downstream code.
build_targets = generate_centernet_targets
draw_umich_gaussian = draw_gaussian
