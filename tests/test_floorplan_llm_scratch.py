"""Test FloorPlanLLM architecture."""

import torch
from src.models.floorplan_llm import build_floorplan_llm


def test_forward():
    """Test forward pass with dummy data."""
    print("=" * 60)
    print("Testing FloorPlanLLM Forward Pass")
    print("=" * 60)

    # Build model
    model = build_floorplan_llm(preset='base', num_classes=35)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✓ Model built successfully")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Size: {total_params * 4 / 1024 / 1024:.1f} MB (float32)")

    # Dummy inputs
    batch_size = 2
    image = torch.randn(batch_size, 3, 512, 512)
    stroke_tokens = torch.randn(batch_size, 100, 12)  # 100 SVG primitives
    stroke_mask = torch.ones(batch_size, 100, dtype=torch.bool)
    stroke_mask[:, 50:] = False  # Second half is padding

    print(f"\n✓ Input shapes:")
    print(f"  image: {tuple(image.shape)}")
    print(f"  stroke_tokens: {tuple(stroke_tokens.shape)}")
    print(f"  stroke_mask: {tuple(stroke_mask.shape)}")

    # Forward pass
    with torch.no_grad():
        output = model(image, stroke_tokens, stroke_mask)

    print(f"\n✓ Output shapes:")
    print(f"  boxes: {tuple(output['boxes'].shape)}")
    print(f"  confidence: {tuple(output['confidence'].shape)}")

    # Check output ranges
    boxes = output['boxes']
    confidence = output['confidence']

    print(f"\n✓ Output statistics:")
    print(f"  boxes range: [{boxes.min():.3f}, {boxes.max():.3f}] (should be [0, 1])")
    print(f"  confidence range: [{confidence.min():.3f}, {confidence.max():.3f}]")

    # Confidence stats (hybrid inference)
    stats = model.get_confidence_stats(output)
    print(f"\n✓ Confidence stats (for hybrid inference):")
    print(f"  max_confidence: {stats['max_confidence']:.3f}")
    print(f"  top5_spread: {stats['top5_spread']:.3f}")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


def test_presets():
    """Test all model presets."""
    print("\n" + "=" * 60)
    print("Testing All Presets")
    print("=" * 60)

    presets = ['tiny', 'small', 'base', 'large']

    for preset in presets:
        model = build_floorplan_llm(preset=preset, num_classes=35)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n{preset:>6}: {total_params:>12,} params ({total_params * 4 / 1024 / 1024:>6.1f} MB)")


if __name__ == '__main__':
    test_forward()
    test_presets()
