"""Test FloorPlanLLM dataset wrapper with real data."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.floorplan_llm_dataset import FloorPlanLLMDataset


def main():
    print("=" * 60)
    print("Testing FloorPlanLLMDataset with Real Data")
    print("=" * 60)

    # Initialize dataset
    print("\n1. Loading dataset...")
    try:
        dataset = FloorPlanLLMDataset(
            root='data/FloorPlanCAD_original',
            split='train_new',
            manifest_path='data/FloorPlanCAD_original/splits_v2.json',
            image_size=512,
        )
        print(f"✓ Dataset loaded successfully")
        print(f"  Split: {dataset.split_role}")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Manifest fingerprint: {dataset.split_manifest_fingerprint[:16]}...")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return

    # Test first sample
    print("\n2. Testing first sample...")
    try:
        sample = dataset[0]
        print(f"✓ Sample loaded successfully")
        print(f"\n  Sample ID: {sample['sample_id']}")
        print(f"  Image shape: {tuple(sample['image'].shape)}")
        print(f"  Stroke tokens shape: {tuple(sample['stroke_tokens'].shape)}")
        print(f"  Stroke mask shape: {tuple(sample['stroke_mask'].shape)}")
        print(f"  Boxes shape: {tuple(sample['boxes'].shape)}")
        print(f"  Classes shape: {tuple(sample['classes'].shape)}")
    except Exception as e:
        print(f"✗ Failed to load sample: {e}")
        import traceback
        traceback.print_exc()
        return

    # Validate box ranges
    print("\n3. Validating box normalization...")
    boxes = sample['boxes']
    classes = sample['classes']

    if len(boxes) > 0:
        print(f"  Number of objects: {len(boxes)}")
        print(f"  Box range: [{boxes.min():.3f}, {boxes.max():.3f}] (should be [0, 1])")

        # Check individual components
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        print(f"  cx range: [{cx.min():.3f}, {cx.max():.3f}]")
        print(f"  cy range: [{cy.min():.3f}, {cy.max():.3f}]")
        print(f"  w range: [{w.min():.3f}, {w.max():.3f}]")
        print(f"  h range: [{h.min():.3f}, {h.max():.3f}]")

        # Show first 3 objects
        print(f"\n  First 3 objects:")
        for i in range(min(3, len(boxes))):
            cls_idx = classes[i].item()
            box = boxes[i]
            print(f"    [{i}] class={cls_idx}, box=(cx={box[0]:.3f}, cy={box[1]:.3f}, w={box[2]:.3f}, h={box[3]:.3f})")

        # Validation
        assert boxes.min() >= 0.0, "Boxes should be >= 0"
        assert boxes.max() <= 1.0, "Boxes should be <= 1"
        print("\n✓ Box normalization valid!")
    else:
        print("  No objects in this sample (empty image)")

    # Test multiple samples
    print("\n4. Testing multiple samples...")
    num_test = min(10, len(dataset))
    total_objects = 0
    empty_images = 0

    for i in range(num_test):
        sample = dataset[i]
        n_objects = len(sample['boxes'])
        total_objects += n_objects
        if n_objects == 0:
            empty_images += 1

    print(f"  Tested {num_test} samples")
    print(f"  Total objects: {total_objects}")
    print(f"  Average objects per image: {total_objects / num_test:.1f}")
    print(f"  Empty images: {empty_images}")

    # Stroke statistics
    print("\n5. Stroke token statistics...")
    sample = dataset[0]
    stroke_mask = sample['stroke_mask']
    n_valid_strokes = stroke_mask.sum().item()
    n_total_strokes = len(stroke_mask)

    print(f"  Valid strokes: {n_valid_strokes} / {n_total_strokes}")
    print(f"  Padding ratio: {(n_total_strokes - n_valid_strokes) / n_total_strokes:.1%}")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
