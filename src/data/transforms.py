"""Paired image/box transforms for FloorPlanCAD detection data."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from PIL import Image
from torchvision.transforms import functional as TF

DEFAULT_MEAN = (0.5, 0.5, 0.5)
DEFAULT_STD = (0.5, 0.5, 0.5)


def as_boxes(boxes: torch.Tensor | Sequence[Sequence[float]]) -> torch.Tensor:
    """Convert boxes to a float ``[N, 4]`` tensor."""

    result = torch.as_tensor(boxes, dtype=torch.float32)
    if result.numel() == 0:
        return result.reshape(0, 4)
    if result.ndim != 2 or result.shape[1] != 4:
        raise ValueError("boxes must have shape [N, 4]")
    return result.clone()


def validate_half_open_boxes(
    boxes: torch.Tensor | Sequence[Sequence[float]],
    image_size: tuple[int, int],
) -> None:
    """Raise if boxes are non-finite, empty, or outside ``(width, height)``."""

    tensor = as_boxes(boxes)
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if tensor.numel() == 0:
        return
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("boxes must be finite")
    if not bool((tensor[:, 2] > tensor[:, 0]).all() and (tensor[:, 3] > tensor[:, 1]).all()):
        raise ValueError("boxes must be non-empty half-open xyxy boxes")
    if not bool(
        (tensor[:, 0] >= 0).all()
        and (tensor[:, 1] >= 0).all()
        and (tensor[:, 2] <= width).all()
        and (tensor[:, 3] <= height).all()
    ):
        raise ValueError("boxes must be within image bounds")


def resize_boxes(
    boxes: torch.Tensor | Sequence[Sequence[float]],
    old_size: tuple[int, int],
    new_size: tuple[int, int],
) -> torch.Tensor:
    """Resize half-open boxes; sizes use PIL ``(width, height)`` order."""

    tensor = as_boxes(boxes)
    old_w, old_h = old_size
    new_w, new_h = new_size
    if min(old_w, old_h, new_w, new_h) <= 0:
        raise ValueError("image dimensions must be positive")
    if tensor.numel() == 0:
        return tensor
    scale = torch.tensor(
        [new_w / old_w, new_h / old_h, new_w / old_w, new_h / old_h],
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return tensor * scale


def horizontal_flip_boxes(
    boxes: torch.Tensor | Sequence[Sequence[float]],
    width: int,
) -> torch.Tensor:
    """Flip half-open boxes horizontally within an image width."""

    tensor = as_boxes(boxes)
    if width <= 0:
        raise ValueError("width must be positive")
    if tensor.numel() == 0:
        return tensor
    result = tensor.clone()
    result[:, 0] = width - tensor[:, 2]
    result[:, 2] = width - tensor[:, 0]
    return result


def vertical_flip_boxes(
    boxes: torch.Tensor | Sequence[Sequence[float]],
    height: int,
) -> torch.Tensor:
    """Flip half-open boxes vertically within an image height."""

    tensor = as_boxes(boxes)
    if height <= 0:
        raise ValueError("height must be positive")
    if tensor.numel() == 0:
        return tensor
    result = tensor.clone()
    result[:, 1] = height - tensor[:, 3]
    result[:, 3] = height - tensor[:, 1]
    return result


def rotate_boxes_90(
    boxes: torch.Tensor | Sequence[Sequence[float]],
    image_size: tuple[int, int],
    k: int = 1,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Rotate boxes clockwise by ``k * 90`` degrees.

    Returns ``(boxes, new_size)`` with sizes in ``(width, height)`` order.
    """

    tensor = as_boxes(boxes)
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    k %= 4
    if k == 0:
        return tensor, (width, height)
    result = tensor.clone()
    if k == 1:  # clockwise: (x, y) -> (H-y, x)
        result[:, 0] = height - tensor[:, 3]
        result[:, 1] = tensor[:, 0]
        result[:, 2] = height - tensor[:, 1]
        result[:, 3] = tensor[:, 2]
        return result, (height, width)
    if k == 2:
        result[:, 0] = width - tensor[:, 2]
        result[:, 1] = height - tensor[:, 3]
        result[:, 2] = width - tensor[:, 0]
        result[:, 3] = height - tensor[:, 1]
        return result, (width, height)
    # 270 clockwise / 90 counter-clockwise: (x, y) -> (y, W-x)
    result[:, 0] = tensor[:, 1]
    result[:, 1] = width - tensor[:, 2]
    result[:, 2] = tensor[:, 3]
    result[:, 3] = width - tensor[:, 0]
    return result, (height, width)


def rotate_image_90(image: Image.Image, k: int) -> Image.Image:
    """Rotate a PIL image clockwise by ``k * 90`` degrees."""

    k %= 4
    if k == 0:
        return image
    if k == 1:
        return image.transpose(Image.Transpose.ROTATE_270)
    if k == 2:
        return image.transpose(Image.Transpose.ROTATE_180)
    return image.transpose(Image.Transpose.ROTATE_90)


def _normalize_tensor(
    image: Image.Image,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    tensor = TF.to_tensor(image)
    return TF.normalize(tensor, mean=list(mean), std=list(std))


@dataclass
class ResizeNormalize:
    """Deterministic paired resize and normalization for validation/test."""

    image_size: int | tuple[int, int] = 512
    mean: tuple[float, float, float] = DEFAULT_MEAN
    std: tuple[float, float, float] = DEFAULT_STD
    antialias: bool = True
    supports_boxes: bool = True

    def output_size(self) -> tuple[int, int]:
        if isinstance(self.image_size, int):
            if self.image_size <= 0:
                raise ValueError("image_size must be positive")
            return self.image_size, self.image_size
        if len(self.image_size) != 2:
            raise ValueError("image_size must be an int or (height, width)")
        height, width = (int(value) for value in self.image_size)
        if height <= 0 or width <= 0:
            raise ValueError("image_size dimensions must be positive")
        return height, width

    def __call__(
        self,
        image: Image.Image,
        boxes: torch.Tensor | Sequence[Sequence[float]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = image.convert("RGB")
        old_size = image.size
        height, width = self.output_size()
        resized = TF.resize(image, [height, width], antialias=self.antialias)
        resized_boxes = resize_boxes(boxes, old_size, (width, height))
        validate_half_open_boxes(resized_boxes, (width, height))
        return _normalize_tensor(resized, self.mean, self.std), resized_boxes


@dataclass
class PairedTrainTransform(ResizeNormalize):
    """Random geometry/photometry followed by a fixed resize-normalize.

    Geometry is applied to image and boxes together. Photometric changes never
    alter boxes. Set probabilities to ``0`` or ``1`` and pass a seeded
    ``random.Random`` instance for deterministic synthetic tests.
    """

    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.0
    rotation_probability: float = 0.5
    photometric_probability: float = 0.8
    brightness: float = 0.10
    contrast: float = 0.10
    saturation: float = 0.05
    rng: random.Random | None = None

    def __post_init__(self) -> None:
        for name in (
            "horizontal_flip_probability",
            "vertical_flip_probability",
            "rotation_probability",
            "photometric_probability",
        ):
            value = float(getattr(self, name))
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("brightness", "contrast", "saturation"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

    def _random(self) -> random.Random:
        # The module implements the same random()/uniform()/choice API.
        return self.rng if self.rng is not None else random  # type: ignore[return-value]

    def __call__(
        self,
        image: Image.Image,
        boxes: torch.Tensor | Sequence[Sequence[float]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = image.convert("RGB")
        transformed_boxes = as_boxes(boxes)
        validate_half_open_boxes(transformed_boxes, image.size)
        rng = self._random()

        if rng.random() < self.horizontal_flip_probability:
            transformed_boxes = horizontal_flip_boxes(transformed_boxes, image.width)
            image = TF.hflip(image)
        if rng.random() < self.vertical_flip_probability:
            transformed_boxes = vertical_flip_boxes(transformed_boxes, image.height)
            image = TF.vflip(image)
        if rng.random() < self.rotation_probability:
            k = rng.choice((1, 2, 3))
            transformed_boxes, expected_size = rotate_boxes_90(
                transformed_boxes, image.size, k
            )
            image = rotate_image_90(image, k)
            if image.size != expected_size:  # defensive check around PIL conventions
                raise RuntimeError("image and box rotation sizes disagree")

        if rng.random() < self.photometric_probability:
            if self.brightness:
                image = TF.adjust_brightness(
                    image, rng.uniform(max(0.0, 1.0 - self.brightness), 1.0 + self.brightness)
                )
            if self.contrast:
                image = TF.adjust_contrast(
                    image, rng.uniform(max(0.0, 1.0 - self.contrast), 1.0 + self.contrast)
                )
            if self.saturation:
                image = TF.adjust_saturation(
                    image, rng.uniform(max(0.0, 1.0 - self.saturation), 1.0 + self.saturation)
                )

        old_size = image.size
        height, width = self.output_size()
        image = TF.resize(image, [height, width], antialias=self.antialias)
        transformed_boxes = resize_boxes(transformed_boxes, old_size, (width, height))
        validate_half_open_boxes(transformed_boxes, (width, height))
        return _normalize_tensor(image, self.mean, self.std), transformed_boxes


def build_train_transform(
    image_size: int | tuple[int, int] = 512,
    **kwargs: object,
) -> PairedTrainTransform:
    return PairedTrainTransform(image_size=image_size, **kwargs)


def build_eval_transform(
    image_size: int | tuple[int, int] = 512,
    **kwargs: object,
) -> ResizeNormalize:
    return ResizeNormalize(image_size=image_size, **kwargs)


def apply_transform(
    transform: Callable[..., object],
    image: Image.Image,
    boxes: torch.Tensor | Sequence[Sequence[float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a paired transform, with a resize-only adapter for legacy callables."""

    source_boxes = as_boxes(boxes)
    if getattr(transform, "supports_boxes", False):
        result = transform(image, source_boxes)
    else:
        try:
            result = transform(image, source_boxes)
        except TypeError:
            result = transform(image)

    if isinstance(result, tuple) and len(result) == 2:
        tensor, transformed_boxes = result
        transformed_boxes = as_boxes(transformed_boxes)
    else:
        tensor = result
        if not isinstance(tensor, torch.Tensor) or tensor.ndim < 2:
            raise TypeError("image-only transforms must return a tensor")
        new_h, new_w = int(tensor.shape[-2]), int(tensor.shape[-1])
        transformed_boxes = resize_boxes(source_boxes, image.size, (new_w, new_h))

    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
        raise TypeError("transforms must return a CHW image tensor")
    validate_half_open_boxes(transformed_boxes, (int(tensor.shape[-1]), int(tensor.shape[-2])))
    return tensor, transformed_boxes


# Concise aliases used in configs/tests.
TrainTransform = PairedTrainTransform
EvalTransform = ResizeNormalize
DeterministicTransform = ResizeNormalize
