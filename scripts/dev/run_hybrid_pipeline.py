"""Orchestrator: Run complete hybrid detector pipeline.

Automates:
1. Phase 1: Train parallel decoder (fast baseline)
2. Evaluate Phase 1 (quality + latency)
3. Phase 2: Add selective reasoning (quality improvement)
4. Evaluate Phase 2 (final quality + latency)
5. Compare results and generate report

Total estimated time: 6-8 hours on GPU
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_command(cmd: list[str], description: str) -> dict:
    """Run command and return results."""
    print("\n" + "=" * 60)
    print(f"[STEP] {description}")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print()

    start_time = datetime.now()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = (datetime.now() - start_time).total_seconds()

    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(1)

    print(f"\n[OK] Completed in {elapsed:.1f}s")

    return {
        'command': ' '.join(cmd),
        'elapsed_seconds': elapsed,
        'success': True,
    }


def main():
    parser = argparse.ArgumentParser(description='Run hybrid detector pipeline')
    parser.add_argument('--epochs_phase1', type=int, default=10, help='Epochs for Phase 1')
    parser.add_argument('--epochs_phase2', type=int, default=10, help='Epochs for Phase 2')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--skip_phase1', action='store_true', help='Skip Phase 1 if already trained')
    parser.add_argument('--profile_latency', action='store_true', help='Profile inference speed')
    args = parser.parse_args()

    print("=" * 60)
    print("HYBRID PARALLEL DETECTOR PIPELINE")
    print("=" * 60)
    print(f"Phase 1 epochs: {args.epochs_phase1}")
    print(f"Phase 2 epochs: {args.epochs_phase2}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60)

    results = {
        'start_time': datetime.now().isoformat(),
        'config': vars(args),
        'steps': [],
    }

    # Phase 1: Train parallel decoder
    if not args.skip_phase1:
        phase1_result = run_command(
            [
                sys.executable,
                'train_hybrid_phase1.py',
                '--epochs', str(args.epochs_phase1),
                '--batch_size', str(args.batch_size),
                '--lr', str(args.lr),
            ],
            "Phase 1: Train parallel decoder (fast baseline)"
        )
        results['steps'].append(phase1_result)
    else:
        print("\n[SKIP] Phase 1 training (--skip_phase1)")

    # Evaluate Phase 1
    phase1_checkpoint = 'checkpoints/hybrid_parallel_phase1/best_val.pt'
    if not Path(phase1_checkpoint).exists():
        print(f"\n[ERROR] Phase 1 checkpoint not found: {phase1_checkpoint}")
        sys.exit(1)

    eval_phase1_cmd = [
        sys.executable,
        'evaluate_hybrid_detector.py',
        '--checkpoint', phase1_checkpoint,
        '--phase', 'phase1',
        '--output_path', 'outputs/hybrid_phase1_evaluation.json',
    ]
    if args.profile_latency:
        eval_phase1_cmd.append('--profile_latency')

    eval_phase1_result = run_command(
        eval_phase1_cmd,
        "Evaluate Phase 1 (quality + latency)"
    )
    results['steps'].append(eval_phase1_result)

    # Load Phase 1 results
    with open('outputs/hybrid_phase1_evaluation.json', 'r') as f:
        phase1_metrics = json.load(f)

    print(f"\nPhase 1 Results:")
    print(f"  AP50: {phase1_metrics['metrics']['AP50']:.4f}")
    print(f"  AP50:95: {phase1_metrics['metrics']['AP50:95']:.4f}")
    if phase1_metrics.get('latency'):
        print(f"  Latency: {phase1_metrics['latency']['median_ms']:.2f} ms")
        print(f"  Throughput: {phase1_metrics['latency']['throughput_ims_per_sec']:.1f} img/s")

    # Phase 2: Add selective reasoning
    phase2_result = run_command(
        [
            sys.executable,
            'train_hybrid_phase2.py',
            '--epochs', str(args.epochs_phase2),
            '--batch_size', str(args.batch_size),
            '--lr', str(args.lr),
            '--phase1_weights', phase1_checkpoint,
        ],
        "Phase 2: Add selective reasoning refinement"
    )
    results['steps'].append(phase2_result)

    # Evaluate Phase 2
    phase2_checkpoint = 'checkpoints/hybrid_parallel_phase2/best_val.pt'
    if not Path(phase2_checkpoint).exists():
        print(f"\n[ERROR] Phase 2 checkpoint not found: {phase2_checkpoint}")
        sys.exit(1)

    eval_phase2_cmd = [
        sys.executable,
        'evaluate_hybrid_detector.py',
        '--checkpoint', phase2_checkpoint,
        '--phase', 'phase2',
        '--output_path', 'outputs/hybrid_phase2_evaluation.json',
    ]
    if args.profile_latency:
        eval_phase2_cmd.append('--profile_latency')

    eval_phase2_result = run_command(
        eval_phase2_cmd,
        "Evaluate Phase 2 (final quality + latency)"
    )
    results['steps'].append(eval_phase2_result)

    # Load Phase 2 results
    with open('outputs/hybrid_phase2_evaluation.json', 'r') as f:
        phase2_metrics = json.load(f)

    print(f"\nPhase 2 Results:")
    print(f"  AP50: {phase2_metrics['metrics']['AP50']:.4f}")
    print(f"  AP50:95: {phase2_metrics['metrics']['AP50:95']:.4f}")
    if phase2_metrics.get('latency'):
        print(f"  Latency: {phase2_metrics['latency']['median_ms']:.2f} ms")
        print(f"  Throughput: {phase2_metrics['latency']['throughput_ims_per_sec']:.1f} img/s")

    # Final comparison
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    ap50_gain = phase2_metrics['metrics']['AP50'] - phase1_metrics['metrics']['AP50']
    ap5095_gain = phase2_metrics['metrics']['AP50:95'] - phase1_metrics['metrics']['AP50:95']

    print(f"\nQuality Improvement:")
    print(f"  AP50 gain: {ap50_gain:+.4f}")
    print(f"  AP50:95 gain: {ap5095_gain:+.4f}")

    if phase1_metrics.get('latency') and phase2_metrics.get('latency'):
        latency_overhead = (
            phase2_metrics['latency']['median_ms'] - phase1_metrics['latency']['median_ms']
        )
        print(f"\nLatency Overhead:")
        print(f"  Phase 1: {phase1_metrics['latency']['median_ms']:.2f} ms")
        print(f"  Phase 2: {phase2_metrics['latency']['median_ms']:.2f} ms")
        print(f"  Overhead: {latency_overhead:+.2f} ms ({latency_overhead/phase1_metrics['latency']['median_ms']*100:+.1f}%)")

    # Save final report
    results['end_time'] = datetime.now().isoformat()
    results['phase1_metrics'] = phase1_metrics
    results['phase2_metrics'] = phase2_metrics
    results['comparison'] = {
        'ap50_gain': ap50_gain,
        'ap5095_gain': ap5095_gain,
    }

    report_path = Path('outputs/hybrid_pipeline_report.json')
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Pipeline complete! Report saved to {report_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
