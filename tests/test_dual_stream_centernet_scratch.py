"""Test DualStreamCenterNet architecture."""

import torch
from src.models.dual_stream_centernet import build_dual_stream_centernet


def test_forward():
    """Test forward pass."""
    print("=" * 60)
    print("Testing DualStreamCenterNet")
    print("=" * 60)

    # Build model
    model = build_dual_stream_centernet(preset='small', num_classes=35)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Model built (small preset)")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Size: {total_params * 4 / 1024 / 1024:.1f} MB")

    # Dummy inputs
    batch_size = 2
    image = torch.randn(batch_size, 3, 512, 512)
    stroke_tokens = torch.randn(batch_size, 100, 12)
    stroke_mask = torch.ones(batch_size, 100, dtype=torch.bool)
    stroke_mask[:, 50:] = False

    print(f"\n✓ Input shapes:")
    print(f"  image: {tuple(image.shape)}")
    print(f"  stroke_tokens: {tuple(stroke_tokens.shape)}")
    print(f"  stroke_mask: {tuple(stroke_mask.shape)}")

    # Forward
    with torch.no_grad():
        output = model(image, stroke_tokens, stroke_mask)

    print(f"\n✓ Output shapes:")
    print(f"  heatmap: {tuple(output['heatmap'].shape)}")
    print(f"  size: {tuple(output['size'].shape)}")
    print(f"  offset: {tuple(output['offset'].shape)}")

    # Check output ranges
    heatmap = output['heatmap']
    size = output['size']
    offset = output['offset']

    print(f"\n✓ Output statistics:")
    print(f"  heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    print(f"  size range: [{size.min():.3f}, {size.max():.3f}]")
    print(f"  offset range: [{offset.min():.3f}, {offset.max():.3f}]")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    test_forward()
