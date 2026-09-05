"""CenterNet output decoding utilities."""

import torch
import torch.nn.functional as F


def decode_centernet_output(
    heatmap: torch.Tensor,
    size_map: torch.Tensor,
    offset_map: torch.Tensor,
    score_threshold: float = 0.05,
    top_k: int = 100,
    kernel_size: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decode CenterNet heatmap into boxes and scores.

    Args:
        heatmap: (B, 1, H, W) - center heatmap logits
        size_map: (B, 2, H, W) - width, height predictions
        offset_map: (B, 2, H, W) - local offsets
        score_threshold: Minimum confidence score
        top_k: Maximum detections per image
        kernel_size: NMS kernel size

    Returns:
        boxes: (B, top_k, 4) - [x1, y1, x2, y2] normalized coordinates
        scores: (B, top_k) - confidence scores
    """
    batch_size, _, height, width = heatmap.shape

    # Apply sigmoid to get probabilities
    heatmap = torch.sigmoid(heatmap)

    # NMS: keep local maxima
    heatmap = _nms(heatmap, kernel_size=kernel_size)

    # Get top-k peaks per image
    batch_boxes = []
    batch_scores = []

    for b in range(batch_size):
        # Flatten and get top-k
        heat = heatmap[b, 0]  # (H, W)
        heat_flat = heat.view(-1)
        topk_scores, topk_inds = torch.topk(heat_flat, k=min(top_k, heat_flat.size(0)))

        # Filter by threshold
        keep = topk_scores > score_threshold
        topk_scores = topk_scores[keep]
        topk_inds = topk_inds[keep]

        # Convert to y, x coordinates
        topk_ys = (topk_inds // width).float()
        topk_xs = (topk_inds % width).float()

        # Apply offsets
        offset_y = offset_map[b, 0, topk_ys.long(), topk_xs.long()]
        offset_x = offset_map[b, 1, topk_ys.long(), topk_xs.long()]
        topk_ys = topk_ys + offset_y
        topk_xs = topk_xs + offset_x

        # Get sizes
        size_h = size_map[b, 0, topk_ys.long().clamp(0, height-1), topk_xs.long().clamp(0, width-1)]
        size_w = size_map[b, 1, topk_ys.long().clamp(0, height-1), topk_xs.long().clamp(0, width-1)]

        # Convert to boxes [x1, y1, x2, y2] in feature-map coordinates
        half_h = size_h / 2
        half_w = size_w / 2
        boxes = torch.stack([
            topk_xs - half_w,
            topk_ys - half_h,
            topk_xs + half_w,
            topk_ys + half_h,
        ], dim=1)

        # Normalize to [0, 1]
        boxes[:, [0, 2]] /= width
        boxes[:, [1, 3]] /= height
        boxes = boxes.clamp(0, 1)

        # Pad to top_k
        num_det = boxes.size(0)
        if num_det < top_k:
            pad_boxes = torch.zeros(top_k - num_det, 4, device=boxes.device)
            pad_scores = torch.zeros(top_k - num_det, device=boxes.device)
            boxes = torch.cat([boxes, pad_boxes], dim=0)
            topk_scores = torch.cat([topk_scores, pad_scores], dim=0)
        else:
            boxes = boxes[:top_k]
            topk_scores = topk_scores[:top_k]

        batch_boxes.append(boxes)
        batch_scores.append(topk_scores)

    return torch.stack(batch_boxes), torch.stack(batch_scores)


def _nms(heatmap: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Apply max pooling NMS to keep local maxima.

    Args:
        heatmap: (B, C, H, W)
        kernel_size: Pooling kernel size

    Returns:
        Heatmap with non-maximum suppressed
    """
    pad = (kernel_size - 1) // 2
    hmax = F.max_pool2d(heatmap, kernel_size=kernel_size, stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return heatmap * keep
