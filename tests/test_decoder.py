import pytest
import torch

from src.evaluation.decoder import (
    CenterNetDecoder,
    decode_centernet,
    local_peak_suppression,
)


def test_local_peak_suppression_removes_neighbors_and_nonfinite_values() -> None:
    heatmap = torch.tensor(
        [
            [0.0, 0.2, 0.0, 0.0],
            [0.0, 0.9, 0.8, 0.0],
            [0.0, 0.0, float("nan"), 0.0],
            [0.0, 0.0, 0.0, 0.7],
        ]
    )

    suppressed = local_peak_suppression(heatmap, kernel=3)

    assert suppressed[1, 1].item() == pytest.approx(0.9)
    assert suppressed[1, 2].item() == 0.0
    assert suppressed[3, 3].item() == pytest.approx(0.7)
    assert suppressed[2, 2].item() == 0.0
    assert torch.isfinite(suppressed).all()


def test_decode_all_classes_uses_per_class_regression_channels_and_clips() -> None:
    heatmap = torch.zeros(1, 2, 4, 4)
    size_map = torch.zeros(1, 4, 4, 4)
    offset_map = torch.zeros(1, 4, 4, 4)

    heatmap[0, 0, 1, 1] = 0.9
    size_map[0, 0, 1, 1] = 2.0  # class 0 width
    size_map[0, 1, 1, 1] = 4.0  # class 0 height
    offset_map[0, 0, 1, 1] = 0.25
    offset_map[0, 1, 1, 1] = 0.50

    heatmap[0, 1, 2, 3] = 0.8
    size_map[0, 2, 2, 3] = 3.0  # class 1 width
    size_map[0, 3, 2, 3] = 1.0  # class 1 height
    offset_map[0, 2, 2, 3] = 0.50

    result = decode_centernet(
        {
            "center_heatmap": heatmap,
            "size_map": size_map,
            "offset_map": offset_map,
        },
        image_ids=["floor-a"],
        stride=2,
        image_size=(8, 8),
        threshold=0.5,
        topk=1,
    )[0]

    assert result["image_id"] == "floor-a"
    assert torch.equal(result["labels"], torch.tensor([0, 1]))
    assert torch.allclose(result["scores"], torch.tensor([0.9, 0.8]))
    assert torch.allclose(
        result["boxes"],
        torch.tensor(
            [
                [0.5, 0.0, 4.5, 7.0],
                [4.0, 3.0, 8.0, 5.0],
            ]
        ),
    )


def test_decode_query_outputs_assigns_requested_labels_and_filters_bad_boxes() -> None:
    heatmap = torch.zeros(2, 1, 3, 3)
    size_map = torch.zeros(2, 2, 3, 3)
    offset_map = torch.zeros(2, 2, 3, 3)
    heatmap[:, 0, 1, 1] = torch.tensor([0.75, 0.95])

    size_map[0, :, 1, 1] = torch.tensor([1.0, 2.0])
    offset_map[0, :, 1, 1] = torch.tensor([0.5, 0.25])
    size_map[1, :, 1, 1] = torch.tensor([0.0, 2.0])  # non-positive width

    decoder = CenterNetDecoder(stride=4, score_threshold=0.75, topk=5)
    results = decoder(
        {
            "center_heatmap": heatmap,
            "size_map": size_map,
            "offset_map": offset_map,
        },
        image_ids=[101, 102],
        class_ids=torch.tensor([7, 4]),
        image_size=12,
    )

    assert results[0]["image_id"] == 101
    assert torch.equal(results[0]["labels"], torch.tensor([7]))
    assert torch.allclose(results[0]["scores"], torch.tensor([0.75]))
    assert torch.allclose(results[0]["boxes"], torch.tensor([[4.0, 1.0, 8.0, 9.0]]))

    assert results[1]["image_id"] == 102
    assert results[1]["boxes"].shape == (0, 4)
    assert results[1]["scores"].shape == (0,)
    assert results[1]["labels"].dtype == torch.long


def test_decode_filters_nonfinite_regression_values() -> None:
    heatmap = torch.tensor([[[[0.9]]]])
    size_map = torch.tensor([[[[1.0]], [[1.0]]]])
    offset_map = torch.tensor([[[[float("inf")]], [[0.0]]]])

    result = decode_centernet(
        {
            "center_heatmap": heatmap,
            "size_map": size_map,
            "offset_map": offset_map,
        },
        threshold=0.1,
        stride=8,
    )[0]

    assert result["boxes"].numel() == 0


def test_decoder_rejects_wrong_all_class_regression_shape() -> None:
    outputs = {
        "center_heatmap": torch.zeros(1, 2, 2, 2),
        "size_map": torch.zeros(1, 2, 2, 2),
        "offset_map": torch.zeros(1, 4, 2, 2),
    }
    with pytest.raises(ValueError, match="size_map"):
        decode_centernet(outputs)
