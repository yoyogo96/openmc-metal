"""Published GPU Monte Carlo benchmark reference data.

Sources:
    [1] Tramm, J.R. et al. (2024). Performance Analysis of GPU-Accelerated
        Monte Carlo Particle Transport with OpenMC. Proc. M&C 2025.
        Speedup values vs. 52-core CPU node. Benchmark: depleted pincell
        (continuous-energy).
    [2] Morgan, J. et al. (2025). GPU-Accelerated Monte Carlo Dynamic Code
        (MC/DC). arXiv:2501.05440. Speedup vs. single CPU node.
        Benchmark: C5G7 multi-group.
    [3] Biondo, E. et al. (2025). GPU Acceleration of the Shift Monte Carlo
        Particle Transport Code. EPJ Nuclear Sci. Technol. 11, 5.
    [4] Hamilton, S.P. et al. (2018). Multigroup Monte Carlo on GPUs.
        Ann. Nucl. Energy 113, 506-518.
    Apple Inc. specifications - M-series TDP estimates.

Note: absolute particles/sec throughput is NOT comparable across different
benchmark problems and energy treatments. Only speedup vs. CPU is used
as the cross-code comparison metric.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BenchmarkReference:
    """A single published benchmark data point from literature."""
    code: str
    gpu: str
    benchmark: str
    particles_per_second: Optional[float]
    speedup_vs_cpu: Optional[float]
    cpu_cores_equivalent: Optional[int]
    tdp_watts: float
    estimated_cost_usd: float
    source: str

    @property
    def perf_per_watt(self) -> Optional[float]:
        if self.particles_per_second is None:
            return None
        return self.particles_per_second / self.tdp_watts


@dataclass
class BenchmarkResult:
    """Benchmark result from OpenMC-Metal."""
    gpu: str
    benchmark: str
    particles_per_second: float
    speedup_vs_cpu: float
    wall_time_seconds: float
    gpu_time_seconds: float
    num_particles: int
    num_batches: int
    k_eff: Optional[float] = None
    k_eff_std_dev: Optional[float] = None


# Published reference database
REFERENCES = [
    # OpenMC GPU (OpenMP target offload) - Tramm et al. 2024 [1]
    # Speedup vs. 52-core CPU node; benchmark: depleted pincell (CE).
    # Absolute particles/sec not used — not comparable across benchmarks.
    BenchmarkReference(
        code="OpenMC", gpu="Intel PVC (Aurora)",
        benchmark="Depleted Pincell (CE)",
        particles_per_second=None, speedup_vs_cpu=17.0,
        cpu_cores_equivalent=None, tdp_watts=600,
        estimated_cost_usd=15_000,
        source="Tramm et al. 2024, Proc. M&C 2025"
    ),
    BenchmarkReference(
        code="OpenMC", gpu="NVIDIA A100 80GB",
        benchmark="Depleted Pincell (CE)",
        particles_per_second=None, speedup_vs_cpu=9.0,
        cpu_cores_equivalent=None, tdp_watts=300,
        estimated_cost_usd=10_000,
        source="Tramm et al. 2024, Proc. M&C 2025"
    ),
    BenchmarkReference(
        code="OpenMC", gpu="AMD MI250X",
        benchmark="Depleted Pincell (CE)",
        particles_per_second=None, speedup_vs_cpu=10.0,
        cpu_cores_equivalent=None, tdp_watts=560,
        estimated_cost_usd=12_000,
        source="Tramm et al. 2024, Proc. M&C 2025"
    ),

    # MC/DC GPU - Morgan et al. 2025 [2]
    # Speedup vs. single CPU node; benchmark: C5G7 multi-group.
    BenchmarkReference(
        code="MC/DC", gpu="NVIDIA V100",
        benchmark="C5G7 Multi-Group",
        particles_per_second=None, speedup_vs_cpu=15.0,
        cpu_cores_equivalent=415, tdp_watts=300,
        estimated_cost_usd=8_000,
        source="Morgan et al. 2025, arXiv:2501.05440"
    ),
    BenchmarkReference(
        code="MC/DC", gpu="AMD MI300A",
        benchmark="C5G7 Multi-Group",
        particles_per_second=None, speedup_vs_cpu=12.0,
        cpu_cores_equivalent=326, tdp_watts=550,
        estimated_cost_usd=11_000,
        source="Morgan et al. 2025, arXiv:2501.05440"
    ),

    # Shift GPU - Biondo et al. 2025 [3]
    BenchmarkReference(
        code="Shift", gpu="NVIDIA V100",
        benchmark="Fixed Source",
        particles_per_second=None, speedup_vs_cpu=28.0,
        cpu_cores_equivalent=None, tdp_watts=300,
        estimated_cost_usd=8_000,
        source="Biondo et al. 2025, EPJ Nuclear Sci. Technol. 11, 5"
    ),
    BenchmarkReference(
        code="Shift", gpu="AMD MI250X (Frontier)",
        benchmark="Eigenvalue",
        particles_per_second=None, speedup_vs_cpu=5.0,
        cpu_cores_equivalent=None, tdp_watts=560,
        estimated_cost_usd=12_000,
        source="Biondo et al. 2025, EPJ Nuclear Sci. Technol. 11, 5"
    ),

    # Hamilton et al. 2018 [4]
    BenchmarkReference(
        code="Hamilton", gpu="NVIDIA P100",
        benchmark="MG Eigenvalue",
        particles_per_second=None, speedup_vs_cpu=None,
        cpu_cores_equivalent=80, tdp_watts=250,
        estimated_cost_usd=5_000,
        source="Hamilton et al. 2018, Ann. Nucl. Energy 113, 506-518"
    ),

    # Apple Silicon TDPs (for perf/watt comparison)
    BenchmarkReference(
        code="Reference", gpu="Apple M1 Pro (16 GPU cores)",
        benchmark="N/A",
        particles_per_second=None, speedup_vs_cpu=None,
        cpu_cores_equivalent=None, tdp_watts=30,
        estimated_cost_usd=2_000,
        source="Apple specifications"
    ),
    BenchmarkReference(
        code="Reference", gpu="Apple M4 Max (40 GPU cores)",
        benchmark="N/A",
        particles_per_second=None, speedup_vs_cpu=None,
        cpu_cores_equivalent=None, tdp_watts=40,
        estimated_cost_usd=4_000,
        source="Apple specifications"
    ),
]


# MC/DC absolute throughput from Morgan et al. 2025 (arXiv:2501.05440)
# C5G7 7-group assembly benchmark, 1M particles/batch, 150 batches
MCDC_THROUGHPUT = {
    'single_v100': 109_000,      # ~109K histories/sec per GPU
    'four_v100': 437_000,        # ~437K histories/sec (4x V100)
    'single_mi300a': 130_000,    # estimated from speedup ratios
    'source': 'Morgan et al. 2025, arXiv:2501.05440',
    'benchmark': 'C5G7 7-group, 1M particles/batch, 150 batches',
}


def get_apple_tdp(gpu_name: str) -> float:
    """Get TDP for an Apple GPU name. Defaults to 40W (M4 Max)."""
    name_lower = gpu_name.lower()
    if 'm1' in name_lower:
        return 30.0 if 'pro' in name_lower or 'max' in name_lower else 20.0
    elif 'm2' in name_lower:
        return 35.0 if 'pro' in name_lower or 'max' in name_lower else 22.0
    elif 'm3' in name_lower:
        return 37.0 if 'pro' in name_lower or 'max' in name_lower else 25.0
    elif 'm4' in name_lower:
        return 40.0 if 'max' in name_lower else 28.0
    return 40.0


def print_comparison_table(metal_result: BenchmarkResult):
    """Print formatted comparison table to stdout."""
    def fmt_pps(pps):
        if pps is None: return "n/r"
        if pps >= 1e9: return f"{pps/1e9:.1f}B"
        if pps >= 1e6: return f"{pps/1e6:.0f}M"
        return f"{pps/1e3:.0f}K"

    def fmt_speedup(s):
        if s is None: return "n/r"
        return f"{s:.1f}x"

    def fmt_ppw(pps, tdp):
        if pps is None: return "n/r"
        ppw = pps / tdp
        if ppw >= 1e6: return f"{ppw/1e6:.2f}M/W"
        if ppw >= 1e3: return f"{ppw/1e3:.1f}K/W"
        return f"{ppw:.0f}/W"

    tdp = get_apple_tdp(metal_result.gpu)

    print("\n" + "=" * 80)
    print("          Monte Carlo GPU Benchmark Comparison")
    print("=" * 80)
    print(f"{'Code':<12}{'GPU':<26}{'Particles/sec':<16}{'Speedup':<11}{'Perf/Watt':<16}")
    print("-" * 80)

    # This work first
    print(f"{'THIS WORK':<12}{metal_result.gpu[:25]:<26}"
          f"{fmt_pps(metal_result.particles_per_second):<16}"
          f"{fmt_speedup(metal_result.speedup_vs_cpu):<11}"
          f"{fmt_ppw(metal_result.particles_per_second, tdp):<16}")

    # Published references
    for ref in REFERENCES:
        if ref.code == "Reference":
            continue
        pps_str = f"~{fmt_pps(ref.particles_per_second)}" if ref.particles_per_second else "n/r"
        spd_str = f"~{fmt_speedup(ref.speedup_vs_cpu)}" if ref.speedup_vs_cpu else "n/r"
        ppw_str = f"~{fmt_ppw(ref.particles_per_second, ref.tdp_watts)}" if ref.particles_per_second else "n/r"
        print(f"{ref.code[:11]:<12}{ref.gpu[:25]:<26}{pps_str:<16}{spd_str:<11}{ppw_str:<16}")

    print("=" * 80)
