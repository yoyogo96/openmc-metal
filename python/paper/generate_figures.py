#!/usr/bin/env python3
"""Generate publication-quality figures for the OpenMC-Metal paper.

Reads benchmark_results.json and produces PNG figures in the figures/ directory.
All figures use consistent styling suitable for two-column LaTeX papers.

Usage:
    python generate_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

COLUMN_WIDTH = 3.5   # inches (single column in two-column layout)
FULL_WIDTH = 7.0     # inches (full page width)

# Colors
C_THIS_WORK  = '#2166ac'   # blue (event-based)
C_PERSISTENT = '#08519c'   # darker blue (persistent kernel)
C_MCDC       = '#b2182b'   # red
C_REFERENCE  = '#d6604d'   # light red
C_RUNNING    = '#4daf4a'   # green
C_BATCH      = '#377eb8'   # blue
C_ENTROPY    = '#984ea3'   # purple
C_UO2        = '#e6ab02'   # gold
C_WATER      = '#66c2a5'   # teal
C_GUIDE      = '#8da0cb'   # light blue
C_FISSION    = '#fc8d62'   # orange

# Persistent kernel fallback constants (assembly, 1M particles)
PERSISTENT_THROUGHPUT_HIST_PER_SEC = 1_776_000
PERSISTENT_TDP_WATTS = 40
PERSISTENT_EFFICIENCY_HIST_PER_SEC_PER_WATT = 44_400


def load_results():
    """Load benchmark_results.json."""
    results_path = Path(__file__).parent.parent / 'benchmark_results.json'
    if not results_path.exists():
        print(f"Error: {results_path} not found")
        sys.exit(1)
    with open(results_path) as f:
        return json.load(f)


def fig_keff_convergence(results, output_dir):
    """Figure 3: K-eff convergence vs batch number."""
    c5g7 = results['c5g7']
    keff_history = c5g7['batch_keff_history']
    reference = c5g7['reference_k_eff']
    num_inactive = c5g7['num_inactive']

    batches = list(range(num_inactive + 1, num_inactive + len(keff_history) + 1))

    # Running mean
    running_mean = np.cumsum(keff_history) / np.arange(1, len(keff_history) + 1)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.4))

    ax.plot(batches, keff_history, '-', color=C_BATCH, linewidth=0.6,
            alpha=0.6, label='Batch $k_\\mathrm{eff}$')
    ax.plot(batches, running_mean, '-', color=C_RUNNING, linewidth=1.5,
            label='Running mean')
    ax.axhline(y=reference, color=C_REFERENCE, linestyle='--', linewidth=1.0,
               label=f'Reference ({reference:.5f})')

    # Mark final value
    mean_k = c5g7['k_eff_mean']
    std_k = c5g7['k_eff_std_dev']
    ax.axhline(y=mean_k, color=C_THIS_WORK, linestyle=':', linewidth=0.8,
               label=f'This work ({mean_k:.4f} $\\pm$ {std_k:.4f})')

    ax.set_xlabel('Batch number')
    ax.set_ylabel('$k_\\mathrm{eff}$')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(alpha=0.2)
    ax.set_ylim(1.310, 1.340)

    path = output_dir / 'fig_keff_convergence.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"  Created {path.name}")


def fig_entropy(results, output_dir):
    """Figure 4: Shannon entropy convergence."""
    c5g7 = results['c5g7']
    entropy = c5g7['entropy_history']

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.2))

    ax.plot(range(1, len(entropy) + 1), entropy, '-', color=C_ENTROPY,
            linewidth=0.8)

    ax.set_xlabel('Batch number')
    ax.set_ylabel('Shannon entropy (bits)')
    ax.grid(alpha=0.2)

    # Mark equilibrium region
    eq_mean = np.mean(entropy[20:])
    ax.axhline(y=eq_mean, color='gray', linestyle='--', linewidth=0.6,
               alpha=0.5, label=f'Mean = {eq_mean:.3f}')
    ax.legend(loc='lower right', framealpha=0.9)

    path = output_dir / 'fig_entropy.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"  Created {path.name}")


def fig_mcdc_comparison(results, output_dir):
    """Figure 5: Absolute throughput bar chart — This Work vs MC/DC."""
    mcdc = results['mcdc_comparison']
    tw = mcdc['this_work']
    v100 = mcdc['mcdc_reference']['single_v100']
    v100x4 = mcdc['mcdc_reference']['four_v100']

    # Persistent kernel value — read from JSON if available, else use constant
    persistent_tp = (mcdc.get('this_work_persistent', {})
                         .get('throughput_hist_per_sec',
                              PERSISTENT_THROUGHPUT_HIST_PER_SEC))

    labels = ['Persistent\nM4 Max', 'Event-based\nM4 Max',
              'MC/DC\n1$\\times$ V100', 'MC/DC\n4$\\times$ V100']
    values = [persistent_tp,
              tw['throughput_hist_per_sec'],
              v100['throughput_hist_per_sec'],
              v100x4['throughput_hist_per_sec']]
    colors = [C_PERSISTENT, C_THIS_WORK, C_MCDC, C_MCDC]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.6))
    bars = ax.bar(range(len(labels)), [v / 1000 for v in values],
                  color=colors, edgecolor='black', linewidth=0.5, width=0.6)

    # Hatching for 4x V100
    bars[3].set_hatch('//')

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                f'{val / 1000:.0f}K', ha='center', va='bottom', fontsize=8,
                fontweight='bold')

    ax.set_ylabel('Histories/sec ($\\times 10^3$)')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(values) / 1000 * 1.25)
    ax.grid(axis='y', alpha=0.2)

    path = output_dir / 'fig_mcdc_comparison.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"  Created {path.name}")


def fig_energy_efficiency(results, output_dir):
    """Figure 6: Performance per watt comparison."""
    mcdc = results['mcdc_comparison']
    tw = mcdc['this_work']
    v100 = mcdc['mcdc_reference']['single_v100']
    v100x4 = mcdc['mcdc_reference']['four_v100']

    # Persistent kernel efficiency — read from JSON if available, else use constant
    persistent_eff = (mcdc.get('this_work_persistent', {})
                          .get('hist_per_sec_per_watt',
                               PERSISTENT_EFFICIENCY_HIST_PER_SEC_PER_WATT))

    labels = ['Persistent\nM4 Max\n(40 W)', 'Event-based\nM4 Max\n(40 W)',
              'MC/DC\nV100\n(300 W)', 'MC/DC\n4$\\times$ V100\n(1200 W)']
    values = [persistent_eff,
              tw['hist_per_sec_per_watt'],
              v100['hist_per_sec_per_watt'],
              v100x4['hist_per_sec_per_watt']]
    colors = [C_PERSISTENT, C_THIS_WORK, C_MCDC, C_MCDC]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.6))
    bars = ax.bar(range(len(labels)), values,
                  color=colors, edgecolor='black', linewidth=0.5, width=0.6)
    bars[3].set_hatch('//')

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f'{val:,}', ha='center', va='bottom', fontsize=8,
                fontweight='bold')

    ax.set_ylabel('Histories/sec/W')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis='y', alpha=0.2)

    # Annotation: efficiency ratio (persistent vs single V100 = 121x)
    ratio = persistent_eff / v100['hist_per_sec_per_watt']
    ax.annotate(f'{ratio:.0f}$\\times$', xy=(2, (values[0] + values[2]) / 2),
                fontsize=10, fontweight='bold', color=C_PERSISTENT,
                ha='center', va='center',
                arrowprops=dict(arrowstyle='->', color=C_PERSISTENT, lw=1.2),
                xytext=(0.5, values[0] * 0.65))

    path = output_dir / 'fig_energy_efficiency.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"  Created {path.name}")


def fig_assembly_layout(output_dir):
    """Figure 7: 17x17 pin assembly layout with material coloring."""
    N = 17
    guide_tubes = {
        (2, 5), (2, 8), (2, 11),
        (3, 3), (3, 13),
        (5, 2), (5, 5), (5, 8), (5, 11), (5, 14),
        (8, 2), (8, 5), (8, 11), (8, 14),
        (11, 2), (11, 5), (11, 8), (11, 11), (11, 14),
        (13, 3), (13, 13),
        (14, 5), (14, 8), (14, 11),
    }
    fission_chamber = (8, 8)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH))

    pitch = 1.26
    r_fuel = 0.54

    for row in range(N):
        for col in range(N):
            cx = col * pitch + pitch / 2
            cy = (N - 1 - row) * pitch + pitch / 2  # flip y for display

            # Moderator square (background)
            rect = plt.Rectangle((col * pitch, (N - 1 - row) * pitch),
                                 pitch, pitch, facecolor=C_WATER,
                                 edgecolor='gray', linewidth=0.3)
            ax.add_patch(rect)

            # Pin circle
            if (row, col) == fission_chamber:
                color = C_FISSION
                label = 'FC'
            elif (row, col) in guide_tubes:
                color = C_GUIDE
                label = 'GT'
            else:
                color = C_UO2
                label = None

            circle = plt.Circle((cx, cy), r_fuel, facecolor=color,
                                edgecolor='black', linewidth=0.3)
            ax.add_patch(circle)

    ax.set_xlim(0, N * pitch)
    ax.set_ylim(0, N * pitch)
    ax.set_aspect('equal')
    ax.set_xlabel('$x$ (cm)')
    ax.set_ylabel('$y$ (cm)')

    # Legend
    from matplotlib.patches import Patch, Circle
    legend_elements = [
        Patch(facecolor=C_UO2, edgecolor='black', linewidth=0.5, label='UO$_2$ fuel (264)'),
        Patch(facecolor=C_GUIDE, edgecolor='black', linewidth=0.5, label='Guide tube (24)'),
        Patch(facecolor=C_FISSION, edgecolor='black', linewidth=0.5, label='Fission chamber (1)'),
        Patch(facecolor=C_WATER, edgecolor='gray', linewidth=0.5, label='Moderator'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=7,
              framealpha=0.9)

    path = output_dir / 'fig_assembly_layout.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"  Created {path.name}")


def fig_architecture_comparison(output_dir):
    """Figure: Event-based vs persistent kernel throughput for pincell and assembly."""
    groups = ['Pincell', 'Assembly']
    event_values   = [46_198,   111_772]
    persist_values = [291_000, 1_776_000]

    x = np.arange(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.6))

    bars_event   = ax.bar(x - width / 2, [v / 1000 for v in event_values],
                          width, label='Event-based', color=C_THIS_WORK,
                          edgecolor='black', linewidth=0.5)
    bars_persist = ax.bar(x + width / 2, [v / 1000 for v in persist_values],
                          width, label='Persistent', color=C_PERSISTENT,
                          edgecolor='black', linewidth=0.5)

    for bar, val in zip(bars_event, event_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                f'{val / 1000:.0f}K', ha='center', va='bottom', fontsize=8,
                fontweight='bold')
    for bar, val in zip(bars_persist, persist_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                f'{val / 1000:.0f}K', ha='center', va='bottom', fontsize=8,
                fontweight='bold')

    ax.set_ylabel('Histories/sec ($\\times 10^3$)')
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, max(persist_values) / 1000 * 1.25)
    ax.legend(framealpha=0.9)
    ax.grid(axis='y', alpha=0.2)

    path = output_dir / 'fig_architecture_comparison.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"  Created {path.name}")


def fig_xsbench(results, output_dir):
    """Supplementary: XSBench throughput across runs."""
    xs = results['xsbench']
    runs = xs['runs']
    throughputs = [r['throughput_mlookups_sec'] for r in runs]
    mean_tp = xs['mean_throughput_mlookups_sec']

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.0))
    bars = ax.bar(range(1, len(throughputs) + 1), throughputs,
                  color=C_THIS_WORK, edgecolor='black', linewidth=0.5,
                  width=0.6)
    ax.axhline(y=mean_tp, color=C_REFERENCE, linestyle='--', linewidth=1.0,
               label=f'Mean: {mean_tp:.0f} M/sec')

    ax.set_xlabel('Run')
    ax.set_ylabel('Throughput (M lookups/sec)')
    ax.legend(framealpha=0.9)
    ax.grid(axis='y', alpha=0.2)
    ax.set_ylim(bottom=10000)

    path = output_dir / 'fig_xsbench.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"  Created {path.name}")


def main():
    results = load_results()
    output_dir = Path(__file__).parent / 'figures'
    output_dir.mkdir(exist_ok=True)

    print("Generating figures for OpenMC-Metal paper...")
    fig_keff_convergence(results, output_dir)
    fig_entropy(results, output_dir)
    fig_mcdc_comparison(results, output_dir)
    fig_energy_efficiency(results, output_dir)
    fig_assembly_layout(output_dir)
    fig_xsbench(results, output_dir)
    fig_architecture_comparison(output_dir)
    print("Done.")


if __name__ == '__main__':
    main()
