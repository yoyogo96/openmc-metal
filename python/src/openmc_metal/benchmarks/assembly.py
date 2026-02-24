"""C5G7 17x17 UO2 assembly k-eigenvalue benchmark."""

from ..engine import MetalEngine
from ..geometry import (
    build_c5g7_assembly, assembly_source_sampler, find_cell_assembly,
    ASSEMBLY_REFERENCE_KEFF,
)
from ..cross_sections import c5g7_materials_buffer
from ..simulation import Simulation
from .reference_data import BenchmarkResult, get_apple_tdp, print_comparison_table


class AssemblyBenchmark:
    """C5G7 17x17 UO2 assembly benchmark runner."""

    def __init__(self, engine: MetalEngine):
        self.engine = engine

    def run(self, num_particles: int = 100_000,
            num_batches: int = 50, num_inactive: int = 10,
            verbose: bool = False) -> dict:
        """Run the C5G7 17x17 assembly benchmark.

        Returns: dict with simulation results and comparison data.
        """
        print("\n" + "=" * 60)
        print("   C5G7 17x17 UO2 Assembly K-Eigenvalue Benchmark")
        print("=" * 60)

        # Build geometry
        geometry = build_c5g7_assembly(self.engine)

        # Load materials
        mat_data = c5g7_materials_buffer()
        materials_buffer = self.engine.make_buffer_from_data(mat_data)

        # Create simulation
        sim = Simulation(
            engine=self.engine,
            geometry=geometry,
            materials_buffer=materials_buffer,
            num_particles=num_particles,
            num_batches=num_batches,
            num_inactive=num_inactive,
            num_groups=7,
            cell_finder=find_cell_assembly,
            fallback_source=assembly_source_sampler,
        )

        # Generate initial source across all fuel pins
        source = assembly_source_sampler(num_particles)

        # Run simulation
        result = sim.run(source=source, verbose=verbose)
        result.print_summary(reference=ASSEMBLY_REFERENCE_KEFF)

        # Build benchmark result for comparison
        mean_k, std_k, ci95 = result.k_eff
        delta_pcm = abs(mean_k - ASSEMBLY_REFERENCE_KEFF) * 1e5

        bench_result = BenchmarkResult(
            gpu=result.gpu_name,
            benchmark="C5G7 17x17 Assembly (7-group)",
            particles_per_second=result.particles_per_second,
            speedup_vs_cpu=1.0,  # CPU baseline not implemented
            wall_time_seconds=result.wall_time,
            gpu_time_seconds=result.wall_time,  # approximation
            num_particles=num_particles * (num_batches - num_inactive),
            num_batches=num_batches - num_inactive,
            k_eff=mean_k,
            k_eff_std_dev=std_k,
        )

        # Print comparison table
        print_comparison_table(bench_result)

        return {
            'num_particles': num_particles,
            'num_batches': num_batches,
            'num_inactive': num_inactive,
            'k_eff_mean': mean_k,
            'k_eff_std_dev': std_k,
            'k_eff_ci95': list(ci95),
            'reference_k_eff': ASSEMBLY_REFERENCE_KEFF,
            'delta_pcm': delta_pcm,
            'wall_time_sec': result.wall_time,
            'particles_per_sec': result.particles_per_second,
            'total_particles': result.total_particles,
            'batch_keff_history': result.batch_keff_history,
            'entropy_history': result.entropy_history,
            'gpu_name': result.gpu_name,
        }
