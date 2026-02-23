"""Matplotlib chart generators for benchmark report."""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for PDF generation
import matplotlib.pyplot as plt
import numpy as np
import tempfile
from pathlib import Path


# Consistent style
COLORS = plt.cm.Set2.colors
FIGURE_DPI = 150

# Published GPU MC speedup data (vs CPU baseline reported in each paper).
# Only speedup values are used — absolute throughput is NOT comparable across
# different benchmark problems and energy treatments.
GPU_MC_SPEEDUP_DATA = [
    # (label, speedup_vs_cpu, tdp_watts, benchmark_type, source_short)
    ("OpenMC\nA100", 9.0, 300, "CE", "[1]"),
    ("OpenMC\nMI250X", 10.0, 560, "CE", "[1]"),
    ("OpenMC\nPVC Aurora", 17.0, 600, "CE", "[1]"),
    ("MC/DC\nV100", 15.0, 300, "MG", "[2]"),
    ("MC/DC\nMI300A", 12.0, 550, "MG", "[2]"),
    ("Shift\nV100", 28.0, 300, "FS", "[3]"),
]

BENCHMARK_COLORS = {
    "CE": COLORS[1],   # Continuous-energy
    "MG": COLORS[2],   # Multi-group
    "FS": COLORS[3],   # Fixed source
}


def performance_comparison_chart(results: dict, output_path: str) -> str:
    """Bar chart: published GPU-to-CPU speedup across MC codes.

    Uses speedup (the metric papers actually report) rather than
    absolute particles/sec which is not comparable across different
    benchmark problems.  Includes This Work (Metal) when CPU baseline
    speedup is available.
    """
    # Build data list: start with This Work if speedup is available
    data = []
    metal_speedup = None
    if 'cpu_baseline' in results:
        metal_speedup = results['cpu_baseline'].get('cpu_speedup', 0)
    if metal_speedup and metal_speedup > 0:
        gpu = results.get('c5g7', {}).get('gpu_name', 'Apple M4 Max')
        data.append((f"This Work\n{gpu}", metal_speedup, 40, "MG", ""))

    data.extend(GPU_MC_SPEEDUP_DATA)

    labels = [d[0] for d in data]
    speedups = [d[1] for d in data]
    bench_types = [d[3] for d in data]
    sources = [d[4] for d in data]
    bar_colors = [BENCHMARK_COLORS[bt] for bt in bench_types]

    # Highlight This Work bar
    if metal_speedup and metal_speedup > 0:
        bar_colors[0] = COLORS[0]  # distinct color for This Work

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(labels)), speedups, color=bar_colors,
                  edgecolor='black', linewidth=0.5)

    # Bold edge for This Work
    if metal_speedup and metal_speedup > 0:
        bars[0].set_edgecolor('red')
        bars[0].set_linewidth(2.0)

    ax.set_ylabel('Speedup vs CPU', fontsize=12)
    ax.set_title('GPU Monte Carlo Speedup vs CPU', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9, ha='center')

    for bar, val, src in zip(bars, speedups, sources):
        label = f'{val:.1f}x {src}'.strip()
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
               label, ha='center', va='bottom', fontsize=8)

    # Legend for benchmark types
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=COLORS[0], edgecolor='red', linewidth=1.5,
              label='This Work (Apple Metal, MG)'),
        Patch(facecolor=BENCHMARK_COLORS["CE"], label='Continuous-Energy (CE)'),
        Patch(facecolor=BENCHMARK_COLORS["MG"], label='Multi-Group (MG)'),
        Patch(facecolor=BENCHMARK_COLORS["FS"], label='Fixed Source (FS)'),
    ]
    ax.legend(handles=legend_items, fontsize=8, loc='upper left')

    ax.set_ylim(bottom=0, top=max(speedups) * 1.25)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    return output_path


def perf_per_watt_chart(results: dict, output_path: str) -> str:
    """Bar chart: GPU speedup per Watt (energy efficiency).

    Uses speedup/TDP as a hardware-normalized efficiency metric.
    Includes This Work (Metal) when CPU baseline speedup is available.
    """
    data = []
    metal_speedup = None
    if 'cpu_baseline' in results:
        metal_speedup = results['cpu_baseline'].get('cpu_speedup', 0)
    if metal_speedup and metal_speedup > 0:
        gpu = results.get('c5g7', {}).get('gpu_name', 'Apple M4 Max')
        data.append((f"This Work\n{gpu}", metal_speedup, 40, "MG", ""))

    data.extend(GPU_MC_SPEEDUP_DATA)

    labels = [d[0] for d in data]
    efficiencies = [d[1] / d[2] * 1000 for d in data]  # speedup per kW
    bench_types = [d[3] for d in data]
    sources = [d[4] for d in data]
    bar_colors = [BENCHMARK_COLORS[bt] for bt in bench_types]

    if metal_speedup and metal_speedup > 0:
        bar_colors[0] = COLORS[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(labels)), efficiencies, color=bar_colors,
                  edgecolor='black', linewidth=0.5)

    if metal_speedup and metal_speedup > 0:
        bars[0].set_edgecolor('red')
        bars[0].set_linewidth(2.0)

    ax.set_ylabel('Speedup per kW (GPU Efficiency)', fontsize=12)
    ax.set_title('GPU Monte Carlo Energy Efficiency', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9, ha='center')

    for bar, val, src in zip(bars, efficiencies, sources):
        label = f'{val:.1f} {src}'.strip()
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
               label, ha='center', va='bottom', fontsize=8)

    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=COLORS[0], edgecolor='red', linewidth=1.5,
              label='This Work (Apple Metal)'),
        Patch(facecolor=BENCHMARK_COLORS["CE"], label='Continuous-Energy'),
        Patch(facecolor=BENCHMARK_COLORS["MG"], label='Multi-Group'),
        Patch(facecolor=BENCHMARK_COLORS["FS"], label='Fixed Source'),
    ]
    ax.legend(handles=legend_items, fontsize=8, loc='upper left')

    ax.set_ylim(bottom=0, top=max(efficiencies) * 1.25)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    return output_path


def keff_convergence_chart(results: dict, output_path: str) -> str:
    """Line chart: k-eff vs batch number with reference line."""
    if 'c5g7' not in results:
        return _empty_chart(output_path, "No C5G7 data")

    keff_history = results['c5g7'].get('batch_keff_history', [])
    reference = results['c5g7'].get('reference_k_eff', 1.33007)
    num_inactive = results['c5g7'].get('num_inactive', 10)

    if not keff_history:
        return _empty_chart(output_path, "No k-eff history")

    fig, ax = plt.subplots(figsize=(10, 5))

    batches = list(range(num_inactive + 1, num_inactive + len(keff_history) + 1))
    ax.plot(batches, keff_history, 'b-', linewidth=1, alpha=0.7, label='Batch k-eff')
    ax.axhline(y=reference, color='r', linestyle='--', linewidth=1.5, label=f'Reference ({reference:.5f})')

    # Running mean
    if len(keff_history) > 1:
        running_mean = []
        cumsum = 0.0
        for i, k in enumerate(keff_history):
            cumsum += k
            running_mean.append(cumsum / (i + 1))
        ax.plot(batches, running_mean, 'g-', linewidth=2, label='Running mean')

    ax.set_xlabel('Batch Number', fontsize=12)
    ax.set_ylabel('k-effective', fontsize=12)
    ax.set_title('K-effective Convergence', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    return output_path


def entropy_convergence_chart(results: dict, output_path: str) -> str:
    """Line chart: Shannon entropy vs batch number."""
    if 'c5g7' not in results:
        return _empty_chart(output_path, "No C5G7 data")

    entropy = results['c5g7'].get('entropy_history', [])
    if not entropy:
        return _empty_chart(output_path, "No entropy data")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(entropy) + 1), entropy, 'b-', linewidth=1.5)
    ax.set_xlabel('Batch Number', fontsize=12)
    ax.set_ylabel('Shannon Entropy (bits)', fontsize=12)
    ax.set_title('Fission Source Convergence (Shannon Entropy)', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    return output_path


def xsbench_throughput_chart(results: dict, output_path: str) -> str:
    """Bar chart: XSBench throughput across runs."""
    if 'xsbench' not in results:
        return _empty_chart(output_path, "No XSBench data")

    runs = results['xsbench'].get('runs', [])
    if not runs:
        return _empty_chart(output_path, "No XSBench runs")

    throughputs = [r['throughput_mlookups_sec'] for r in runs]
    mean_tp = results['xsbench'].get('mean_throughput_mlookups_sec', 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(1, len(throughputs) + 1), throughputs,
                  color=COLORS[0], edgecolor='black', linewidth=0.5)
    ax.axhline(y=mean_tp, color='r', linestyle='--', linewidth=1.5,
               label=f'Mean: {mean_tp:.1f} M/sec')

    ax.set_xlabel('Run', fontsize=12)
    ax.set_ylabel('M lookups/sec', fontsize=12)
    ax.set_title('XSBench Cross-Section Lookup Throughput', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    return output_path


def _empty_chart(output_path: str, message: str) -> str:
    """Generate a placeholder chart with a message."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=14, color='gray')
    ax.set_axis_off()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    return output_path
