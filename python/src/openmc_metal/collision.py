"""Fission bank and collision kernel dispatch."""

import struct as struct_mod
from .types import FISSIONSITE_FORMAT, FISSIONSITE_SIZE


class FissionBank:
    """GPU fission bank for storing fission sites."""

    def __init__(self, engine, max_sites: int):
        self.engine = engine
        self.max_sites = max_sites
        self.buffer = engine.make_buffer(max_sites * FISSIONSITE_SIZE)
        self.count_buffer = engine.make_buffer(4)  # atomic_uint

    def reset(self):
        """Zero the fission count."""
        self.engine.zero_buffer(self.count_buffer, 4)

    def count(self) -> int:
        """Read the fission site count."""
        mv = self.engine.buffer_view(self.count_buffer)
        return struct_mod.unpack_from('<I', mv, 0)[0]

    def sites(self) -> list:
        """Read fission sites from GPU buffer.

        Returns: List of ((x, y, z), energy_group) tuples.
        """
        n = self.count()
        if n == 0:
            return []

        n = min(n, self.max_sites)
        mv = self.engine.buffer_view(self.buffer)
        result = []

        for i in range(n):
            offset = i * FISSIONSITE_SIZE
            fields = struct_mod.unpack_from(FISSIONSITE_FORMAT, mv, offset)
            # fields: posX, posY, posZ, _pad0, energyGroup, _pad1, _pad2, _pad3
            pos = (fields[0], fields[1], fields[2])
            group = fields[4]
            result.append((pos, group))

        return result


class CollisionKernel:
    """Collision kernel dispatch."""

    def __init__(self, engine):
        self.engine = engine
        self.pipeline = engine.make_pipeline("collision")

    def dispatch(self, particles, materials, fission_bank: FissionBank,
                 params, count: int, command_buffer):
        """Encode collision kernel."""
        self.engine.dispatch(
            self.pipeline,
            [particles, materials, fission_bank.buffer,
             fission_bank.count_buffer, params],
            count, command_buffer
        )
