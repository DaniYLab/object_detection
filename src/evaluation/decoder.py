"""CenterNet output decoding utilities.

The detector has two output layouts:

* query layout: one heatmap and one ``(width, height)`` / ``(dx, dy)``
  pair per image.  ``class_ids`` supplies the queried label for each image.
* all-class layout: ``C`` heatmaps and ``2 * C`` size/offset channels.  The
  regression channels for class ``c`` are ``2*c`` (x/width) and ``2*c + 1``
  (y/height).

Decoded boxes use input-image ``xyxy`` coordinates.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import torch
import torch.nn.functional as F


def _validate_peak_kernel(kernel: int) -> None:
    if not isinstance(kernel, Integral) or isinstance(kernel, bool):
        raise TypeError("peak_kernel must be an integer")
    if kernel <= 0 or kernel % 2 == 0:
        raise ValueError("peak_kernel must be a positive odd integer")


def _peak_mask(heatmap: torch.Tensor, kernel: int) -> torch.Tensor:
    """Return a finite local-maximum mask with the same shape as ``heatmap``."""
    _validate_peak_kernel(kernel)
    if heatmap.ndim not in {2, 3, 4}:
        raise ValueError(
            "heatmap must have shape [H,W], [C,H,W], or [B,C,H,W]"
        )
    if not heatmap.is_floating_point():
        raise TypeError("heatmap must be a floating-point tensor")

    original_ndim = heatmap.ndim
    if original_ndim == 2:
        batched = heatmap.unsqueeze(0).unsqueeze(0)
    elif original_ndim == 3:
        batched = heatmap.unsqueeze(0)
    else:
        batched = heatmap

    finite = torch.isfinite(batched)
    safe = torch.where(finite, batched, torch.full_like(batched, -torch.inf))
    pooled = F.max_pool2d(
        safe,
        kernel_size=kernel,
        stride=1,
        padding=kernel // 2,
    )
    mask = finite & safe.eq(pooled)

    if original_ndim == 2:
        return mask[0, 0]
    if original_ndim == 3:
        return mask[0]
    return mask


def local_peak_suppression(heatmap: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """Keep local maxima and zero all other or non-finite heatmap values.

    Equal-valued plateaus are retained, matching the standard CenterNet
    max-pooling suppression rule.
    """
    return torch.where(_peak_mask(heatmap, kernel), heatmap, torch.zeros_like(heatmap))


def _pair(value: int | float | Sequence[int | float], name: str) -> tuple[float, float]:
    if isinstance(value, Real) and not isinstance(value, bool):
        first = second = float(value)
    elif (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
        and all(isinstance(v, Real) and not isinstance(v, bool) for v in value)
    ):
        first, second = float(value[0]), float(value[1])
    else:
        raise TypeError(f"{name} must be a number or a (height, width) pair")
    if not math.isfinite(first) or not math.isfinite(second) or first <= 0 or second <= 0:
        raise ValueError(f"{name} values must be finite and positive")
    return first, second


def _image_sizes(
    image_size: int | Sequence[int] | Sequence[int | Sequence[int]] | None,
    batch_size: int,
    output_height: int,
    output_width: int,
    stride_y: float,
    stride_x: float,
) -> list[tuple[float, float]]:
    if image_size is None:
        return [(output_height * stride_y, output_width * stride_x)] * batch_size

    # A scalar or numeric pair is one size shared by the whole batch.
    if isinstance(image_size, Real) and not isinstance(image_size, bool):
        return [_pair(image_size, "image_size")] * batch_size
    if (
        isinstance(image_size, Sequence)
        and not isinstance(image_size, (str, bytes))
        and len(image_size) == 2
        and all(isinstance(v, Real) and not isinstance(v, bool) for v in image_size)
    ):
        return [_pair(image_size, "image_size")] * batch_size

    if not isinstance(image_size, Sequence) or isinstance(image_size, (str, bytes)):
        raise TypeError("image_size must be shared or contain one size per image")
    if len(image_size) != batch_size:
        raise ValueError(
            f"image_size has {len(image_size)} entries for a batch of {batch_size}"
        )
    return [_pair(size, f"image_size[{index}]") for index, size in enumerate(image_size)]


def _as_image_ids(image_ids: Sequence[Any] | torch.Tensor | None, batch_size: int) -> list[Any]:
    if image_ids is None:
        return list(range(batch_size))
    if isinstance(image_ids, torch.Tensor):
        if image_ids.ndim != 1:
            raise ValueError("image_ids tensor must be one-dimensional")
        values = image_ids.detach().cpu().tolist()
    elif isinstance(image_ids, Sequence) and not isinstance(image_ids, (str, bytes)):
        values = list(image_ids)
    else:
        raise TypeError("image_ids must be a sequence or one-dimensional tensor")
    if len(values) != batch_size:
        raise ValueError(f"image_ids has {len(values)} entries for a batch of {batch_size}")
    return values


def _label_value(value: Any, name: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must contain scalar labels")
        value = value.item()
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} values must be integers")
    label = int(value)
    if label < 0:
        raise ValueError(f"{name} values must be non-negative")
    return label


def _as_class_ids(
    class_ids: Sequence[int] | torch.Tensor | None,
    batch_size: int,
) -> list[int] | None:
    if class_ids is None:
        return None
    if isinstance(class_ids, torch.Tensor):
        if class_ids.ndim != 1:
            raise ValueError("class_ids tensor must be one-dimensional")
        values: Sequence[Any] = class_ids.detach().cpu().tolist()
    elif isinstance(class_ids, Sequence) and not isinstance(class_ids, (str, bytes)):
        values = class_ids
    else:
        raise TypeError("class_ids must be a sequence or one-dimensional tensor")
    if len(values) != batch_size:
        raise ValueError(f"class_ids has {len(values)} entries for a batch of {batch_size}")
    return [_label_value(value, "class_ids") for value in values]


def _stable_descending_indices(values: torch.Tensor) -> torch.Tensor:
    """Descending indices with flattened-index tie breaking."""
    try:
        return torch.argsort(values, descending=True, stable=True)
    except TypeError:  # pragma: no cover - compatibility with old PyTorch
        # CPU Python sorting supplies the same deterministic tie break.
        order = sorted(
            range(values.numel()),
            key=lambda index: (-float(values[index].detach().cpu()), index),
        )
        return torch.tensor(order, dtype=torch.long, device=values.device)


def decode_centernet(
    outputs: Mapping[str, torch.Tensor],
    image_ids: Sequence[Any] | torch.Tensor | None = None,
    *,
    class_ids: Sequence[int] | torch.Tensor | None = None,
    query_class_ids: Sequence[int] | torch.Tensor | None = None,
    stride: int | float | Sequence[int | float] = 8,
    image_size: int | Sequence[int] | Sequence[int | Sequence[int]] | None = None,
    threshold: float = 0.05,
    score_threshold: float | None = None,
    topk: int = 100,
    peak_kernel: int = 3,
    clip_boxes: bool = True,
) -> list[dict[str, Any]]:
    """Decode query or all-class CenterNet maps into per-image detections.

    Args:
        outputs: Mapping containing ``center_heatmap``, ``size_map`` and
            ``offset_map`` tensors.
        image_ids: Optional identifiers; defaults to batch indices.
        class_ids: Queried class per image for the one-channel layout.
        query_class_ids: Alias for ``class_ids``.
        stride: Output-grid stride, either scalar or ``(y, x)``.
        image_size: Clipping size as ``(height, width)``, a square scalar, or a
            sequence of per-image sizes.  By default it is inferred from the
            output grid and stride.
        threshold: Inclusive minimum center score.
        score_threshold: Alias which, when provided, overrides ``threshold``.
        topk: Maximum retained local peaks per class, before box filtering.
        peak_kernel: Positive odd max-pooling kernel for local suppression.
        clip_boxes: Clip coordinates to each image's bounds.

    Returns:
        A list of dictionaries with ``image_id``, ``boxes`` (``[N,4]`` xyxy),
        ``scores`` (``[N]``), and ``labels`` (``[N]``).
    """
    required = {"center_heatmap", "size_map", "offset_map"}
    missing = required.difference(outputs)
    if missing:
        raise KeyError(f"Missing CenterNet outputs: {sorted(missing)}")

    heatmap = outputs["center_heatmap"]
    size_map = outputs["size_map"]
    offset_map = outputs["offset_map"]
    if not all(isinstance(tensor, torch.Tensor) for tensor in (heatmap, size_map, offset_map)):
        raise TypeError("CenterNet outputs must be torch tensors")
    if not all(tensor.ndim == 4 for tensor in (heatmap, size_map, offset_map)):
        raise ValueError("CenterNet outputs must all have shape [B,C,H,W]")
    if not all(tensor.is_floating_point() for tensor in (heatmap, size_map, offset_map)):
        raise TypeError("CenterNet outputs must be floating-point tensors")
    if heatmap.device != size_map.device or heatmap.device != offset_map.device:
        raise ValueError("CenterNet outputs must be on the same device")
    if heatmap.shape[0] == 0:
        return []

    batch_size, num_heatmap_classes, output_height, output_width = heatmap.shape
    if num_heatmap_classes <= 0 or output_height <= 0 or output_width <= 0:
        raise ValueError("CenterNet output dimensions must be non-zero")
    expected_regression_channels = 2 * num_heatmap_classes
    expected_shape = (batch_size, expected_regression_channels, output_height, output_width)
    if tuple(size_map.shape) != expected_shape:
        raise ValueError(
            f"size_map must have shape {expected_shape}, got {tuple(size_map.shape)}"
        )
    if tuple(offset_map.shape) != expected_shape:
        raise ValueError(
            f"offset_map must have shape {expected_shape}, got {tuple(offset_map.shape)}"
        )

    if class_ids is not None and query_class_ids is not None:
        raise ValueError("Pass only one of class_ids and query_class_ids")
    query_ids = _as_class_ids(
        class_ids if class_ids is not None else query_class_ids,
        batch_size,
    )
    if query_ids is not None and num_heatmap_classes != 1:
        raise ValueError("class_ids are only valid for one-channel query outputs")

    stride_y, stride_x = _pair(stride, "stride")
    sizes = _image_sizes(
        image_size,
        batch_size,
        output_height,
        output_width,
        stride_y,
        stride_x,
    )
    ids = _as_image_ids(image_ids, batch_size)

    if score_threshold is not None:
        threshold = score_threshold
    if not isinstance(threshold, Real) or isinstance(threshold, bool):
        raise TypeError("threshold must be a number")
    threshold = float(threshold)
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if not isinstance(topk, Integral) or isinstance(topk, bool):
        raise TypeError("topk must be an integer")
    if topk < 0:
        raise ValueError("topk must be non-negative")
    _validate_peak_kernel(peak_kernel)

    peak_mask = _peak_mask(heatmap, peak_kernel)
    safe_scores = torch.where(
        peak_mask,
        heatmap,
        torch.full_like(heatmap, -torch.inf),
    )

    decoded: list[dict[str, Any]] = []
    for batch_index in range(batch_size):
        image_height, image_width = sizes[batch_index]
        image_boxes: list[torch.Tensor] = []
        image_scores: list[torch.Tensor] = []
        image_labels: list[torch.Tensor] = []

        for channel in range(num_heatmap_classes):
            flat_scores = safe_scores[batch_index, channel].reshape(-1)
            keep_count = min(int(topk), flat_scores.numel())
            if keep_count == 0:
                continue
            flat_indices = _stable_descending_indices(flat_scores)[:keep_count]
            scores = flat_scores[flat_indices]
            score_keep = torch.isfinite(scores) & scores.ge(threshold)
            if not bool(score_keep.any()):
                continue
            flat_indices = flat_indices[score_keep]
            scores = scores[score_keep]

            ys = torch.div(flat_indices, output_width, rounding_mode="floor")
            xs = flat_indices.remainder(output_width)
            regression_channel = channel * 2
            widths = size_map[batch_index, regression_channel, ys, xs]
            heights = size_map[batch_index, regression_channel + 1, ys, xs]
            offsets_x = offset_map[batch_index, regression_channel, ys, xs]
            offsets_y = offset_map[batch_index, regression_channel + 1, ys, xs]

            centers_x = (xs.to(offsets_x.dtype) + offsets_x) * stride_x
            centers_y = (ys.to(offsets_y.dtype) + offsets_y) * stride_y
            widths = widths * stride_x
            heights = heights * stride_y
            boxes = torch.stack(
                (
                    centers_x - widths / 2,
                    centers_y - heights / 2,
                    centers_x + widths / 2,
                    centers_y + heights / 2,
                ),
                dim=1,
            )
            if clip_boxes:
                boxes[:, 0].clamp_(min=0.0, max=image_width)
                boxes[:, 2].clamp_(min=0.0, max=image_width)
                boxes[:, 1].clamp_(min=0.0, max=image_height)
                boxes[:, 3].clamp_(min=0.0, max=image_height)

            valid = (
                torch.isfinite(boxes).all(dim=1)
                & torch.isfinite(widths)
                & torch.isfinite(heights)
                & widths.gt(0)
                & heights.gt(0)
                & boxes[:, 2].gt(boxes[:, 0])
                & boxes[:, 3].gt(boxes[:, 1])
            )
            if not bool(valid.any()):
                continue

            label = query_ids[batch_index] if query_ids is not None else channel
            image_boxes.append(boxes[valid])
            image_scores.append(scores[valid])
            image_labels.append(
                torch.full(
                    (int(valid.sum().item()),),
                    label,
                    dtype=torch.long,
                    device=heatmap.device,
                )
            )

        if image_boxes:
            boxes_out = torch.cat(image_boxes, dim=0).detach()
            scores_out = torch.cat(image_scores, dim=0).detach()
            labels_out = torch.cat(image_labels, dim=0)
            order = _stable_descending_indices(scores_out)
            boxes_out = boxes_out[order]
            scores_out = scores_out[order]
            labels_out = labels_out[order]
        else:
            boxes_out = heatmap.new_empty((0, 4))
            scores_out = heatmap.new_empty((0,))
            labels_out = torch.empty((0,), dtype=torch.long, device=heatmap.device)

        decoded.append(
            {
                "image_id": ids[batch_index],
                "boxes": boxes_out,
                "scores": scores_out,
                "labels": labels_out,
            }
        )

    return decoded


class CenterNetDecoder:
    """Configurable callable wrapper around :func:`decode_centernet`."""

    def __init__(
        self,
        *,
        stride: int | float | Sequence[int | float] = 8,
        threshold: float = 0.05,
        score_threshold: float | None = None,
        topk: int = 100,
        peak_kernel: int = 3,
        clip_boxes: bool = True,
    ) -> None:
        self.stride = stride
        self.threshold = threshold if score_threshold is None else score_threshold
        self.topk = topk
        self.peak_kernel = peak_kernel
        self.clip_boxes = clip_boxes

    def __call__(
        self,
        outputs: Mapping[str, torch.Tensor],
        image_ids: Sequence[Any] | torch.Tensor | None = None,
        *,
        class_ids: Sequence[int] | torch.Tensor | None = None,
        query_class_ids: Sequence[int] | torch.Tensor | None = None,
        image_size: int | Sequence[int] | Sequence[int | Sequence[int]] | None = None,
    ) -> list[dict[str, Any]]:
        return decode_centernet(
            outputs,
            image_ids,
            class_ids=class_ids,
            query_class_ids=query_class_ids,
            stride=self.stride,
            image_size=image_size,
            threshold=self.threshold,
            topk=self.topk,
            peak_kernel=self.peak_kernel,
            clip_boxes=self.clip_boxes,
        )

    decode = __call__


# Concise compatibility aliases for callers that do not encode model type in the name.
decode_predictions = decode_centernet
local_maximum_suppression = local_peak_suppression


__all__ = [
    "CenterNetDecoder",
    "decode_centernet",
    "decode_predictions",
    "local_maximum_suppression",
    "local_peak_suppression",
]
