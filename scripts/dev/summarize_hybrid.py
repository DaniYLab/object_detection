"""Generate comprehensive summary report of hybrid inference experiment."""

import json
from pathlib import Path
from datetime import datetime


def load_json(path):
    """Load JSON file if exists."""
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return None


def format_seconds(seconds):
    """Format seconds to human readable."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def generate_summary():
    """Generate comprehensive summary."""
    print("=" * 70)
    print("HYBRID INFERENCE EXPERIMENT - SUMMARY REPORT")
    print("=" * 70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Phase 1 results
    print("Phase 1: Parallel Decoder Baseline")
    print("-" * 70)

    phase1_checkpoints = list(Path('checkpoints/hybrid_phase1').glob('epoch_*.pt'))
    if phase1_checkpoints:
        print(f"✓ Checkpoints found: {len(phase1_checkpoints)}")
        for ckpt in sorted(phase1_checkpoints):
            print(f"  - {ckpt.name}")
    else:
        print("✗ No checkpoints found")
    print()

    # Phase 2 results
    print("Phase 2: Hybrid Inference Training")
    print("-" * 70)

    phase2_checkpoints = list(Path('checkpoints/hybrid_phase2').glob('epoch_*.pt'))
    if phase2_checkpoints:
        print(f"✓ Checkpoints found: {len(phase2_checkpoints)}")
        for ckpt in sorted(phase2_checkpoints):
            print(f"  - {ckpt.name}")
    else:
        print("✗ No checkpoints found")
    print()

    # Evaluation results
    print("Phase 3: Evaluation Results")
    print("-" * 70)

    val_select = load_json('outputs/hybrid_val_select.json')
    if val_select:
        stats = val_select['stats']
        total = stats['total']
        fallback = stats['fallback']
        parallel = stats['parallel']
        fallback_rate = fallback / total if total > 0 else 0

        print(f"✓ val_select evaluation complete")
        print(f"  Total samples: {total:,}")
        print(f"  Parallel mode: {parallel:,} ({parallel/total:.1%})")
        print(f"  Fallback mode: {fallback:,} ({fallback_rate:.1%})")
        print()

        # Top classes using fallback
        class_stats = []
        for class_id, data in stats['by_class'].items():
            if data['total'] > 0:
                rate = data['fallback'] / data['total']
                class_stats.append((int(class_id), data['total'], data['fallback'], rate))

        class_stats.sort(key=lambda x: x[3], reverse=True)

        print("  Top 5 classes with highest fallback rate:")
        for class_id, total, fb, rate in class_stats[:5]:
            print(f"    Class {class_id:2d}: {rate:5.1%} ({fb}/{total})")

        print()
        print("  Top 5 classes with lowest fallback rate:")
        for class_id, total, fb, rate in class_stats[-5:]:
            print(f"    Class {class_id:2d}: {rate:5.1%} ({fb}/{total})")
    else:
        print("✗ val_select results not found")

    print()

    val_calib = load_json('outputs/hybrid_val_calibration.json')
    if val_calib:
        stats = val_calib['stats']
        total = stats['total']
        fallback_rate = stats['fallback'] / total if total > 0 else 0
        print(f"✓ val_calibration evaluation complete")
        print(f"  Total samples: {total:,}")
        print(f"  Fallback rate: {fallback_rate:.1%}")
    else:
        print("✗ val_calibration results not found")

    print()

    # Analysis report
    print("Phase 4: Analysis")
    print("-" * 70)

    analysis = load_json('outputs/hybrid_analysis.json')
    if analysis:
        summary = analysis['summary']
        print(f"✓ Analysis complete")
        print(f"  Overall fallback rate: {summary['fallback_rate']:.1%}")
        print(f"  Parallel rate: {summary['parallel_rate']:.1%}")
    else:
        print("✗ Analysis not found")

    print()

    # Comparison with baseline
    print("Comparison with Thin-Head Baseline")
    print("-" * 70)
    print("Thin-head (dual_pathway_v2_thin):")
    print("  Seed 1337: AP50=0.4950, AP50:95=0.3091")
    print("  Seed 2024: AP50=0.4882, AP50:95=0.3035")
    print("  Mean:      AP50=0.4916, AP50:95=0.3063")
    print()
    print("Hybrid inference:")
    print("  [Requires full AP computation with ground truth]")
    print("  Run: python evaluate.py --checkpoint checkpoints/hybrid_phase2/epoch_5.pt")
    print()

    # Next steps
    print("=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Run full AP evaluation:")
    print("   python evaluate.py \\")
    print("     --checkpoint checkpoints/hybrid_phase2/epoch_5.pt \\")
    print("     --split val_select \\")
    print("     --manifest-path data/FloorPlanCAD_original/splits_v2.json")
    print()
    print("2. Compare AP with thin-head baseline")
    print()
    print("3. If AP50:95 ≥ 0.3063: proceed to seed replication")
    print()
    print("4. If fallback_rate < 20%: consider tightening thresholds")
    print()
    print("5. Analyze per-class fallback patterns for insights")
    print()
    print("=" * 70)


if __name__ == '__main__':
    generate_summary()
