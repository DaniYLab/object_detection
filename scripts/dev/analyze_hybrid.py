"""Comprehensive analysis comparing hybrid inference with thin-head baseline."""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np


def load_results(path):
    """Load evaluation results."""
    with open(path) as f:
        return json.load(f)


def compute_ap(predictions, ground_truth, iou_threshold=0.5):
    """Simplified AP computation for analysis."""
    # Group by image and class
    pred_by_image = defaultdict(lambda: defaultdict(list))
    for pred in predictions:
        img_id = pred['image_id']
        class_id = pred['class_id']
        boxes = pred['boxes']
        scores = pred['scores']
        for box, score in zip(boxes, scores):
            pred_by_image[img_id][class_id].append({
                'box': box,
                'score': score,
                'matched': False
            })

    # Sort by score
    for img_id in pred_by_image:
        for class_id in pred_by_image[img_id]:
            pred_by_image[img_id][class_id].sort(key=lambda x: x['score'], reverse=True)

    return pred_by_image


def analyze_fallback_correlation(hybrid_results):
    """Analyze correlation between fallback usage and prediction quality."""
    stats = hybrid_results['stats']
    predictions = hybrid_results['predictions']

    fallback_preds = [p for p in predictions if p.get('fallback_used', False)]
    parallel_preds = [p for p in predictions if not p.get('fallback_used', False)]

    print("\n" + "="*60)
    print("Fallback Analysis")
    print("="*60)
    print(f"Total predictions: {len(predictions)}")
    print(f"Fallback mode: {len(fallback_preds)} ({len(fallback_preds)/len(predictions):.2%})")
    print(f"Parallel mode: {len(parallel_preds)} ({len(parallel_preds)/len(predictions):.2%})")

    # Average predictions per mode
    fallback_avg_boxes = np.mean([len(p['boxes']) for p in fallback_preds]) if fallback_preds else 0
    parallel_avg_boxes = np.mean([len(p['boxes']) for p in parallel_preds]) if parallel_preds else 0

    print(f"\nAverage boxes per prediction:")
    print(f"  Fallback: {fallback_avg_boxes:.2f}")
    print(f"  Parallel: {parallel_avg_boxes:.2f}")

    # Average score per mode
    fallback_scores = [s for p in fallback_preds for s in p['scores']]
    parallel_scores = [s for p in parallel_preds for s in p['scores']]

    if fallback_scores:
        print(f"\nAverage confidence:")
        print(f"  Fallback: {np.mean(fallback_scores):.3f}")
        print(f"  Parallel: {np.mean(parallel_scores):.3f}")


def compare_with_baseline(hybrid_path, baseline_path=None):
    """Compare hybrid with thin-head baseline."""
    hybrid = load_results(hybrid_path)

    print("\n" + "="*60)
    print("Hybrid Inference Results")
    print("="*60)

    analyze_fallback_correlation(hybrid)

    if baseline_path and Path(baseline_path).exists():
        baseline = load_results(baseline_path)
        print("\n" + "="*60)
        print("Comparison with Thin-Head Baseline")
        print("="*60)

        # Compare metrics
        if 'metrics' in baseline:
            print("\nBaseline metrics:")
            for k, v in baseline['metrics'].items():
                print(f"  {k}: {v:.4f}")

        print("\nNote: Full AP comparison requires ground-truth annotations")
        print("Run evaluate.py with --compute-metrics for detailed AP computation")


def generate_report(hybrid_path, output_path):
    """Generate comprehensive analysis report."""
    hybrid = load_results(hybrid_path)

    report = {
        'summary': {
            'total_predictions': len(hybrid['predictions']),
            'fallback_rate': hybrid['stats']['fallback'] / max(1, hybrid['stats']['total']),
            'parallel_rate': hybrid['stats']['parallel'] / max(1, hybrid['stats']['total']),
        },
        'config': hybrid['config'],
        'per_class_fallback': {},
    }

    # Per-class analysis
    for class_id, data in hybrid['stats']['by_class'].items():
        if data['total'] > 0:
            report['per_class_fallback'][class_id] = {
                'total': data['total'],
                'fallback': data['fallback'],
                'rate': data['fallback'] / data['total'],
            }

    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n[OK] Analysis report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hybrid', type=str, required=True, help='Hybrid evaluation results')
    parser.add_argument('--baseline', type=str, help='Baseline evaluation results (optional)')
    parser.add_argument('--output', type=str, default='outputs/hybrid_analysis.json')
    args = parser.parse_args()

    print("="*60)
    print("Hybrid Inference Analysis")
    print("="*60)
    print(f"Hybrid results: {args.hybrid}")
    if args.baseline:
        print(f"Baseline results: {args.baseline}")

    compare_with_baseline(args.hybrid, args.baseline)
    generate_report(args.hybrid, args.output)

    print("\n" + "="*60)
    print("Analysis Complete")
    print("="*60)


if __name__ == '__main__':
    main()
