"""Tally manager for GPU tally accumulation."""

import struct as struct_mod
import numpy as np


class TallyManager:
    """Manages GPU tally buffers and batch-level accumulation.

    GPU stores tally values as atomic_uint (float bits via CAS loop).
    After each batch, reads uint32 bits back, reinterprets as float32,
    and widens to float64 for statistics.
    """

    def __init__(self, engine, num_cells: int, num_groups: int):
        self.engine = engine
        self.num_cells = num_cells
        self.num_groups = num_groups
        self.tally_size = num_cells * num_groups

        # GPU buffers (uint32 for atomic CAS float addition)
        self.flux_buffer = engine.make_buffer(self.tally_size * 4)
        self.fission_buffer = engine.make_buffer(self.tally_size * 4)

        # Compile tally pipeline
        self.tally_pipeline = engine.make_pipeline("tally_score")

        # CPU-side batch accumulation (float64)
        self.batch_flux: list[list[float]] = []
        self.batch_fission: list[list[float]] = []

    def reset_for_batch(self):
        """Zero GPU tally buffers before a new batch."""
        self.engine.zero_buffer(self.flux_buffer, self.tally_size * 4)
        self.engine.zero_buffer(self.fission_buffer, self.tally_size * 4)

    def collect_batch(self):
        """Read GPU tallies into CPU float64 arrays and append to batch history."""
        flux_mv = self.engine.buffer_view(self.flux_buffer)
        fission_mv = self.engine.buffer_view(self.fission_buffer)

        flux = [0.0] * self.tally_size
        fission = [0.0] * self.tally_size

        for i in range(self.tally_size):
            offset = i * 4
            # Read uint32 bits, reinterpret as float32, widen to float64
            flux_bits = struct_mod.unpack_from('<I', flux_mv, offset)[0]
            fission_bits = struct_mod.unpack_from('<I', fission_mv, offset)[0]
            flux[i] = float(struct_mod.unpack('<f', struct_mod.pack('<I', flux_bits))[0])
            fission[i] = float(struct_mod.unpack('<f', struct_mod.pack('<I', fission_bits))[0])

        self.batch_flux.append(flux)
        self.batch_fission.append(fission)

    def clear_batch_history(self):
        """Discard accumulated batch data (end of inactive cycles)."""
        self.batch_flux = []
        self.batch_fission = []

    def dispatch(self, particles, params, count: int, command_buffer):
        """Encode tally_score kernel."""
        self.engine.dispatch(
            self.tally_pipeline,
            [particles, self.flux_buffer, self.fission_buffer, params],
            count, command_buffer
        )
