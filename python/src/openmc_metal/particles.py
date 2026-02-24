"""Particle buffer management for GPU transport."""

import struct as struct_mod
from .types import PARTICLE_FORMAT, PARTICLE_SIZE, ParticleEvent, pack_particle
from .rng import PhiloxRNG


class ParticleBuffer:
    """Manages a GPU buffer of Particle structs."""

    def __init__(self, engine, count: int):
        self.engine = engine
        self.count = count
        self.buffer = engine.make_buffer(count * PARTICLE_SIZE)

    def seed_from_source(self, source: list, batch_index: int,
                         num_groups: int = 7, cell_finder=None):
        """Initialize particles from source distribution.

        Args:
            source: List of ((x, y, z), energy_group) tuples.
            batch_index: Current batch number (used for RNG seeding).
            num_groups: Number of energy groups.
            cell_finder: Callable (x, y, z) -> cell_index. If None, uses
                default pincell logic (2 cells).
        """
        mv = self.engine.buffer_view(self.buffer)

        for i in range(self.count):
            src_idx = i % len(source)
            pos, group = source[src_idx]

            rng = PhiloxRNG(key=i, counter_hi=batch_index)
            dx, dy, dz = rng.sample_isotropic_direction()

            # Determine initial cell from position
            if cell_finder is not None:
                cell_index = cell_finder(pos[0], pos[1], pos[2])
            else:
                # Default pincell logic (backward compatible)
                cx, cy, r = 0.63, 0.63, 0.54
                ddx = pos[0] - cx
                ddy = pos[1] - cy
                cell_index = 0 if (ddx * ddx + ddy * ddy < r * r) else 1

            rng_key = (i ^ ((batch_index * 0x9E3779B9) & 0xFFFFFFFF)) & 0xFFFFFFFF

            particle_data = pack_particle(
                pos_x=pos[0], pos_y=pos[1], pos_z=pos[2],
                dir_x=dx, dir_y=dy, dir_z=dz,
                energy_group=group, weight=1.0, cell_index=cell_index, alive=1,
                event=ParticleEvent.XS_LOOKUP, rng_counter=rng.counter_lo,
                rng_key=rng_key, distance_to_collision=0.0,
                distance_to_boundary=0.0, boundary_surface=-1,
                distance_traveled=0.0, xs_total=0.0,
                xs_scatter=0.0, xs_fission=0.0, xs_nu_fission=0.0, xs_absorption=0.0,
                fission_flag=0, material_index=0,
            )

            offset = i * PARTICLE_SIZE
            mv[offset:offset + PARTICLE_SIZE] = particle_data

    def alive_count(self) -> int:
        """Count particles with alive != 0 using numpy for speed."""
        import numpy as np
        mv = self.engine.buffer_view(self.buffer)
        # alive field: byte offset 44 in each 112-byte particle
        # = uint32 word offset 11 in each 28-word (112/4) particle
        arr = np.frombuffer(mv, dtype=np.uint32)
        words_per_particle = PARTICLE_SIZE // 4  # 28
        alive_word_offset = 44 // 4  # 11
        alive_values = arr[alive_word_offset::words_per_particle]
        return int(np.count_nonzero(alive_values[:self.count]))

    def all_dead(self) -> bool:
        """Check if all particles have alive == 0."""
        return self.alive_count() == 0
