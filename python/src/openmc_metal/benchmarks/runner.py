"""Benchmark suite CLI runner."""

import argparse
import json
import platform
import sys
import time


def run_cpu_baseline_benchmark(c5g7_results: dict) -> dict:
    """Run CPU-only transport baseline and compute GPU speedup.

    Args:
        c5g7_results: Result dict from PincellBenchmark.run(), which contains
                      'particles_per_sec' from the GPU run.

    Returns:
        dict with CPU metrics plus 'cpu_speedup' (GPU / CPU particles/sec).
    """
    from .cpu_baseline import run_cpu_baseline

    # Use 10K particles for a statistically fair baseline comparison.
    # Larger counts would take too long but 10K gives reliable per-particle throughput.
    cpu_num_particles = 10_000
    cpu_num_batches   = 30
    cpu_num_inactive  = 10

    print(f"Running CPU baseline: {cpu_num_particles} particles x {cpu_num_batches} batches ({cpu_num_inactive} inactive) ...")
    cpu_result = run_cpu_baseline(
        num_particles=cpu_num_particles,
        num_batches=cpu_num_batches,
        num_inactive=cpu_num_inactive,
        num_groups=7,
    )

    gpu_pps = c5g7_results.get('particles_per_sec', 0.0)
    cpu_pps = cpu_result['cpu_particles_per_sec']
    speedup = gpu_pps / cpu_pps if cpu_pps > 0 else 0.0

    cpu_result['cpu_speedup'] = speedup
    cpu_result['gpu_particles_per_sec'] = gpu_pps

    print(f"  CPU:       {cpu_pps:,.0f} particles/sec")
    print(f"  GPU:       {gpu_pps:,.0f} particles/sec")
    print(f"  Speedup:   {speedup:.1f}x")

    return cpu_result


def main():
    parser = argparse.ArgumentParser(description='OpenMC-Metal Benchmark Suite')
    parser.add_argument('--xsbench-lookups', type=int, default=10_000_000,
                        help='Number of XSBench lookups (default: 10M)')
    parser.add_argument('--xsbench-runs', type=int, default=5,
                        help='Number of XSBench timed runs (default: 5)')
    parser.add_argument('--particles', type=int, default=100_000,
                        help='Particles per batch (default: 100K)')
    parser.add_argument('--batches', type=int, default=50,
                        help='Total batches (default: 50)')
    parser.add_argument('--inactive', type=int, default=10,
                        help='Inactive batches (default: 10)')
    parser.add_argument('--json', type=str, default='benchmark_results.json',
                        help='Output JSON file path')
    parser.add_argument('--report', type=str, default='benchmark_report.pdf',
                        help='Output PDF report path')
    parser.add_argument('--verbose', action='store_true',
                        help='Print every batch')
    args = parser.parse_args()

    from ..engine import MetalEngine
    from .xsbench import XSBenchmark
    from .pincell import PincellBenchmark

    # Initialize Metal engine
    print("Initializing Metal GPU engine...")
    engine = MetalEngine()
    engine.print_device_info()

    results = {
        'system': {
            'gpu': engine.gpu_name,
            'unified_memory': engine.has_unified_memory,
            'macos_version': platform.mac_ver()[0],
            'python_version': platform.python_version(),
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        }
    }

    # XSBench
    print("\n" + "=" * 60)
    print("PHASE 1: XSBench Microbenchmark")
    print("=" * 60)
    xsbench = XSBenchmark(engine)
    results['xsbench'] = xsbench.run(
        num_lookups=args.xsbench_lookups,
        num_runs=args.xsbench_runs,
    )

    # C5G7 Pincell
    print("\n" + "=" * 60)
    print("PHASE 2: C5G7 Pincell Benchmark")
    print("=" * 60)
    pincell = PincellBenchmark(engine)
    results['c5g7'] = pincell.run(
        num_particles=args.particles,
        num_batches=args.batches,
        num_inactive=args.inactive,
        verbose=args.verbose,
    )

    # CPU Baseline (for GPU speedup comparison)
    print("\n" + "=" * 60)
    print("PHASE 3: CPU Baseline Transport Benchmark")
    print("=" * 60)
    results['cpu_baseline'] = run_cpu_baseline_benchmark(results['c5g7'])

    # Save JSON
    with open(args.json, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.json}")

    # Generate PDF report
    try:
        from ..report.pdf_report import generate_report
        generate_report(results, args.report)
        print(f"Report saved to {args.report}")
    except Exception as e:
        print(f"Warning: PDF report generation failed: {e}")

    return results


if __name__ == '__main__':
    main()
