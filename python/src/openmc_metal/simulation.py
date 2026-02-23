"""Main k-eigenvalue simulation driver."""

import time
import struct as struct_mod

from .types import PARTICLE_SIZE, pack_simulation_params
from .engine import MetalEngine
from .geometry import GeometryData, source_sampler
from .particles import ParticleBuffer
from .transport import TransportKernels
from .collision import FissionBank, CollisionKernel
from .tally import TallyManager
from .statistics import keff_statistics, compute_stats
from .keigenvalue import shannon_entropy, is_converged


class SimulationResult:
    """Aggregated results from a completed k-eigenvalue simulation."""

    def __init__(self, k_eff, flux_stats, fission_stats, wall_time,
                 particles_per_second, total_particles, num_active_batches,
                 gpu_name, batch_keff_history, entropy_history):
        self.k_eff = k_eff  # (mean, std_dev, (ci95_lo, ci95_hi))
        self.flux_stats = flux_stats
        self.fission_stats = fission_stats
        self.wall_time = wall_time
        self.particles_per_second = particles_per_second
        self.total_particles = total_particles
        self.num_active_batches = num_active_batches
        self.gpu_name = gpu_name
        self.batch_keff_history = batch_keff_history
        self.entropy_history = entropy_history

    def print_summary(self, reference=None):
        """Print formatted simulation results."""
        print("\n" + "=" * 60)
        print("SIMULATION RESULTS")
        print("=" * 60)
        mean, std_dev, ci95 = self.k_eff
        print(f"  k-eff = {mean:.5f} +/- {std_dev:.5f}")
        print(f"  95% CI: [{ci95[0]:.5f}, {ci95[1]:.5f}]")
        if reference:
            delta_pcm = (mean - reference) * 1e5
            print(f"  Reference: {reference:.5f}  (delta = {delta_pcm:+.1f} pcm)")
        print(f"\n  Performance: {self.particles_per_second:.0f} particles/sec")
        print(f"  Total particles: {self.total_particles}")
        print(f"  Wall time: {self.wall_time:.2f} sec")
        print(f"  Active batches: {self.num_active_batches}")
        print(f"  GPU: {self.gpu_name}")
        print("=" * 60)


class Simulation:
    """k-eigenvalue Monte Carlo simulation using Apple Metal GPU kernels."""

    def __init__(self, engine: MetalEngine, geometry: GeometryData,
                 materials_buffer, num_particles: int = 100_000,
                 num_batches: int = 100, num_inactive: int = 50,
                 num_groups: int = 7):
        self.engine = engine
        self.geometry = geometry
        self.materials_buffer = materials_buffer
        self.num_particles = num_particles
        self.num_batches = num_batches
        self.num_inactive = num_inactive
        self.num_groups = num_groups

        self.kernels = TransportKernels(engine)
        self.collision = CollisionKernel(engine)
        self.tally = TallyManager(engine, geometry.num_cells, num_groups)
        self.fission_bank = FissionBank(engine, num_particles * 3)
        self.particle_buf = ParticleBuffer(engine, num_particles)
        self.params_buffer = engine.make_buffer(32)
        self.lost_count_buffer = engine.make_buffer(4)
        self.current_keff = 1.0
        self._write_params()

    def _write_params(self):
        """Write current simulation params to GPU buffer."""
        data = pack_simulation_params(
            num_particles=self.num_particles,
            num_batches=self.num_batches,
            num_inactive=self.num_inactive,
            num_groups=self.num_groups,
            num_cells=self.geometry.num_cells,
            num_surfaces=self.geometry.num_surfaces,
            num_materials=7,
            k_eff=self.current_keff
        )
        self.engine.write_buffer_bytes(self.params_buffer, data)

    def run(self, source: list, verbose: bool = False) -> SimulationResult:
        """Execute the full k-eigenvalue simulation."""
        wall_start = time.time()
        total_particles = 0
        current_source = source
        active_batches = self.num_batches - self.num_inactive

        print("Starting k-eigenvalue simulation")
        print(f"  Particles per batch: {self.num_particles}")
        print(f"  Inactive batches:    {self.num_inactive}")
        print(f"  Active batches:      {active_batches}")
        print(f"  Energy groups:       {self.num_groups}")
        print(f"  Cells:               {self.geometry.num_cells}")
        print()

        batch_keff = []
        entropy_history = []

        for batch_idx in range(self.num_batches):
            is_active = batch_idx >= self.num_inactive

            # Reset per-batch state
            self.fission_bank.reset()
            self.tally.reset_for_batch()
            self.engine.zero_buffer(self.lost_count_buffer, 4)

            # Seed particles from current source
            self.particle_buf.seed_from_source(current_source, batch_idx)

            # Update params with current k-eff
            self._write_params()

            # Run transport loop
            iterations = self._transport_batch()

            # Read fission bank
            fission_sites = self.fission_bank.sites()
            fission_count = len(fission_sites)

            # Compute batch k-eff: fission bank already includes nu (the
            # collision kernel banks nu*weight sites per fission), so
            # k_eff = fission_sites / source_particles directly.
            batch_k = fission_count / self.num_particles if fission_count > 0 else 0.0

            # Collect tallies for active batches
            if is_active:
                self.tally.collect_batch()
                batch_keff.append(batch_k)

            # Shannon entropy
            if fission_sites:
                positions = [pos for pos, _ in fission_sites]
                entropy = shannon_entropy(positions)
                entropy_history.append(entropy)

            # Update k-eff
            self.current_keff = float(batch_k)

            # Resample fission bank for next batch
            current_source = self._resample_fission_bank(fission_sites)

            total_particles += self.num_particles

            # Clear tally history after inactive phase
            if batch_idx == self.num_inactive - 1:
                self.tally.clear_batch_history()

            # Print progress
            lost_mv = self.engine.buffer_view(self.lost_count_buffer)
            lost_count = struct_mod.unpack_from('<I', lost_mv, 0)[0]
            if verbose or batch_idx % 10 == 0 or batch_idx == self.num_batches - 1:
                status = "Active  " if is_active else "Inactive"
                converged = " [converged]" if is_converged(entropy_history) else ""
                print(f"Batch {batch_idx+1:3d}/{self.num_batches} [{status}] "
                      f"k-eff = {batch_k:.5f}  fission = {fission_count}  "
                      f"lost = {lost_count}  iter = {iterations}{converged}")

        wall_time = time.time() - wall_start
        particles_per_sec = total_particles / wall_time

        # Compute statistics
        k_eff_stats = keff_statistics(batch_keff)
        flux_stats = compute_stats(self.tally.batch_flux)
        fission_stats = compute_stats(self.tally.batch_fission)

        return SimulationResult(
            k_eff=k_eff_stats,
            flux_stats=flux_stats,
            fission_stats=fission_stats,
            wall_time=wall_time,
            particles_per_second=particles_per_sec,
            total_particles=total_particles,
            num_active_batches=active_batches,
            gpu_name=self.engine.gpu_name,
            batch_keff_history=batch_keff,
            entropy_history=entropy_history,
        )

    def _transport_batch(self) -> int:
        """Run event-based transport loop until all particles dead.

        Uses 3 fused GPU kernels per step (down from 5):
          1. xs_lookup_and_distance (fused XS lookup + distance sampling)
          2. move_particle (standalone - complex geometry logic)
          3. collision_and_tally (fused collision + tally scoring)

        Metal serial queues guarantee in-order execution, so we only
        sync (waitUntilCompleted) when we need to read results on CPU.
        Between syncs, command buffers are committed without waiting,
        allowing the GPU to pipeline work continuously.
        """
        max_steps = 5000
        check_interval = 25  # check all_dead every N steps

        last_cmd = None

        for step in range(max_steps):
            cmd = self.engine.command_queue.commandBuffer()

            # 1. Fused XS Lookup + Distance to Collision
            self.kernels.dispatch_xs_lookup_and_distance(
                self.particle_buf.buffer, self.materials_buffer,
                self.geometry.cell_buffer, self.params_buffer,
                self.num_particles, cmd
            )

            # 2. Move Particle (standalone - complex geometry)
            self.kernels.dispatch_move(
                self.particle_buf.buffer,
                self.geometry.surface_buffer,
                self.geometry.cell_buffer,
                self.geometry.cell_surface_buffer,
                self.params_buffer, self.lost_count_buffer,
                self.num_particles, cmd
            )

            # 3. Fused Collision + Tally
            self.collision.dispatch_fused(
                self.particle_buf.buffer, self.materials_buffer,
                self.fission_bank, self.params_buffer,
                self.tally.flux_buffer, self.tally.fission_buffer,
                self.num_particles, cmd
            )

            cmd.commit()
            last_cmd = cmd

            # Only sync and check periodically to reduce CPU-GPU overhead
            if (step + 1) % check_interval == 0:
                cmd.waitUntilCompleted()
                if self.particle_buf.all_dead():
                    return step + 1

        # Final sync before returning to ensure all GPU work is complete
        if last_cmd is not None:
            last_cmd.waitUntilCompleted()

        return max_steps

    def _resample_fission_bank(self, sites):
        """Resample fission sites for next batch source via random sampling with replacement."""
        import random
        if not sites:
            return source_sampler(self.num_particles, self.num_groups)

        return random.choices(sites, k=self.num_particles)
