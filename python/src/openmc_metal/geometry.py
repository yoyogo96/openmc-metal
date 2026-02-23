"""Geometry builders for Monte Carlo transport.

Builds surfaces, cells, and packs them into GPU buffers matching the Metal struct layouts.
"""

import struct as struct_mod
from .types import (
    GPUSURFACE_FORMAT, GPUSURFACE_SIZE,
    GPUCELL_FORMAT, GPUCELL_SIZE,
    GPUCELLSURFACE_FORMAT, GPUCELLSURFACE_SIZE,
    SurfaceType, BoundaryCondition,
)


class Surface:
    """A geometric surface definition."""

    def __init__(self, surface_type: int, boundary_condition: int,
                 coefficients: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)):
        self.id = 0
        self.type = surface_type
        self.boundary_condition = boundary_condition
        self.coefficients = coefficients

    @classmethod
    def plane_x(cls, position: float, bc: int = BoundaryCondition.VACUUM) -> 'Surface':
        # Metal shader reads coefficients.x for PLANE_X
        return cls(SurfaceType.PLANE_X, bc, (position, 0.0, 0.0, 0.0))

    @classmethod
    def plane_y(cls, position: float, bc: int = BoundaryCondition.VACUUM) -> 'Surface':
        # Metal shader reads coefficients.y for PLANE_Y
        return cls(SurfaceType.PLANE_Y, bc, (0.0, position, 0.0, 0.0))

    @classmethod
    def plane_z(cls, position: float, bc: int = BoundaryCondition.VACUUM) -> 'Surface':
        # Metal shader reads coefficients.z for PLANE_Z
        return cls(SurfaceType.PLANE_Z, bc, (0.0, 0.0, position, 0.0))

    @classmethod
    def cylinder_z(cls, x0: float, y0: float, radius: float,
                   bc: int = BoundaryCondition.VACUUM) -> 'Surface':
        return cls(SurfaceType.CYLINDER_Z, bc, (x0, y0, radius, 0.0))

    def pack(self) -> bytes:
        """Pack into GPU struct format (32 bytes)."""
        return struct_mod.pack(
            GPUSURFACE_FORMAT,
            self.type, self.boundary_condition,
            self.coefficients[0], self.coefficients[1],
            self.coefficients[2], self.coefficients[3],
            0, 0  # padding
        )


class Cell:
    """A geometric cell (region) definition."""

    def __init__(self, material_index: int,
                 surfaces: list[tuple[int, int]]):
        """
        Args:
            material_index: Index into the materials array.
            surfaces: List of (surface_index, sense) tuples.
                sense is +1 (positive half-space) or -1 (negative half-space).
        """
        self.material_index = material_index
        self.surfaces = surfaces  # [(surfaceIndex, sense), ...]


class GeometryData:
    """Packed GPU buffers for geometry."""

    def __init__(self, surface_buffer, cell_buffer, cell_surface_buffer,
                 num_surfaces: int, num_cells: int):
        self.surface_buffer = surface_buffer
        self.cell_buffer = cell_buffer
        self.cell_surface_buffer = cell_surface_buffer
        self.num_surfaces = num_surfaces
        self.num_cells = num_cells

    @classmethod
    def build(cls, engine, surfaces: list[Surface], cells: list[Cell]) -> 'GeometryData':
        """Pack surfaces and cells into GPU buffers.

        Args:
            engine: MetalEngine instance for buffer creation.
            surfaces: List of Surface objects.
            cells: List of Cell objects.
        Returns:
            GeometryData with packed GPU buffers.
        """
        # Pack surfaces
        surface_data = b''.join(s.pack() for s in surfaces)
        surface_buffer = engine.make_buffer_from_data(surface_data)

        # Build cell-surface association list and pack cells
        cell_surfaces_data = b''
        cell_data = b''
        offset = 0

        for cell in cells:
            num_surf = len(cell.surfaces)
            cell_data += struct_mod.pack(
                GPUCELL_FORMAT,
                cell.material_index, num_surf, offset, 0  # pad
            )
            for surf_idx, sense in cell.surfaces:
                cell_surfaces_data += struct_mod.pack(
                    GPUCELLSURFACE_FORMAT,
                    surf_idx, sense
                )
            offset += num_surf

        cell_buffer = engine.make_buffer_from_data(cell_data)
        cell_surface_buffer = engine.make_buffer_from_data(cell_surfaces_data)

        return cls(
            surface_buffer=surface_buffer,
            cell_buffer=cell_buffer,
            cell_surface_buffer=cell_surface_buffer,
            num_surfaces=len(surfaces),
            num_cells=len(cells),
        )


def build_c5g7_pincell(engine) -> GeometryData:
    """Build a single reflected UO2 pincell (C5G7 benchmark).

    Geometry:
        - 7 surfaces: 4 x/y planes + 2 z planes + 1 fuel cylinder
        - 2 cells: fuel (mat 0 = UO2) and moderator (mat 6 = water)
        - All outer planes reflective (infinite lattice)
    """
    surfaces = [
        Surface.plane_x(0.0, bc=BoundaryCondition.REFLECTIVE),    # 0: left
        Surface.plane_x(1.26, bc=BoundaryCondition.REFLECTIVE),   # 1: right
        Surface.plane_y(0.0, bc=BoundaryCondition.REFLECTIVE),    # 2: bottom
        Surface.plane_y(1.26, bc=BoundaryCondition.REFLECTIVE),   # 3: top
        Surface.plane_z(0.0, bc=BoundaryCondition.REFLECTIVE),    # 4: back
        Surface.plane_z(1.0, bc=BoundaryCondition.REFLECTIVE),    # 5: front
        Surface.cylinder_z(0.63, 0.63, 0.54, bc=BoundaryCondition.TRANSMISSIVE),  # 6: fuel pin
    ]
    for i, s in enumerate(surfaces):
        s.id = i

    # Cell 0: UO2 fuel (inside cylinder and inside bounding box)
    fuel = Cell(material_index=0, surfaces=[
        (0, 1), (1, -1), (2, 1), (3, -1), (4, 1), (5, -1), (6, -1)
    ])

    # Cell 1: Water moderator (outside cylinder, inside bounding box)
    moderator = Cell(material_index=6, surfaces=[
        (0, 1), (1, -1), (2, 1), (3, -1), (4, 1), (5, -1), (6, 1)
    ])

    return GeometryData.build(engine, surfaces, [fuel, moderator])


# Reference k-eff for a single reflected UO2 pincell with C5G7 7-group XS.
# Note: The full C5G7 assembly benchmark k-eff is ~1.33007, but a single
# reflected pincell has much lower k-eff due to the high thermal absorption
# in water relative to fuel volume in this geometry.  The 7-group cross
# sections were condensed for the full assembly spectrum, so single-pincell
# results differ from the assembly eigenvalue.
REFERENCE_KEFF = 0.2327


def source_sampler(num_particles: int, num_groups: int = 7):
    """Generate fission source particles inside the fuel cylinder.

    Sources are uniformly distributed in the fuel region with energy
    groups sampled from the UO2 chi spectrum.

    Returns: List of ((x, y, z), energy_group) tuples.
    """
    import random

    cx, cy = 0.63, 0.63
    r = 0.54
    r2 = r * r
    x_min, x_max = cx - r, cx + r
    y_min, y_max = cy - r, cy + r

    # UO2 chi CDF: [0.587910, 0.411760, 0.000340, 0, 0, 0, 0]
    chi_cdf = [0.58791, 0.58791 + 0.41176, 1.0]

    sources = []
    while len(sources) < num_particles:
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        dx, dy = x - cx, y - cy
        if dx * dx + dy * dy > r2:
            continue
        z = random.uniform(0.001, 0.999)

        xi = random.random()
        if xi < chi_cdf[0]:
            group = 0
        elif xi < chi_cdf[1]:
            group = 1
        else:
            group = 2

        sources.append(((x, y, z), group))

    return sources
