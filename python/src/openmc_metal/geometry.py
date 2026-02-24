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


def find_cell_pincell(x: float, y: float, z: float) -> int:
    """Find cell index for pincell geometry (2 cells).

    Cell 0: fuel (inside cylinder at (0.63, 0.63) with r=0.54)
    Cell 1: moderator (outside cylinder)
    """
    cx, cy, r = 0.63, 0.63, 0.54
    dx, dy = x - cx, y - cy
    if dx * dx + dy * dy < r * r:
        return 0
    return 1


def find_cell_assembly(x: float, y: float, z: float) -> int:
    """Find cell index for 17x17 assembly geometry (578 cells).

    Cell layout: (row * 17 + col) * 2     = pin interior
                 (row * 17 + col) * 2 + 1 = moderator annulus
    """
    N = _ASSEMBLY_N
    pitch = _PIN_PITCH
    half = pitch / 2.0
    r2 = _FUEL_RADIUS * _FUEL_RADIUS

    col = int(x / pitch)
    row = int(y / pitch)

    # Clamp to valid range
    col = max(0, min(col, N - 1))
    row = max(0, min(row, N - 1))

    cx = col * pitch + half
    cy = row * pitch + half
    dx, dy = x - cx, y - cy

    pin_idx = row * N + col
    if dx * dx + dy * dy < r2:
        return pin_idx * 2      # pin interior
    return pin_idx * 2 + 1      # moderator annulus


# Reference k-eff for a single reflected UO2 pincell with C5G7 7-group XS.
# With the official MIT-CRPG C5G7 cross-section data, a reflected UO2 pincell
# gives k-eff ≈ 1.33 (close to the full C5G7 assembly eigenvalue of 1.33007).
REFERENCE_KEFF = 1.33007


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


# ---------------------------------------------------------------------------
# 17x17 UO2 Assembly (C5G7 benchmark)
# ---------------------------------------------------------------------------

# Standard C5G7 UO2 assembly reference k-inf for infinite lattice.
ASSEMBLY_REFERENCE_KEFF = 1.33007

# Pin pitch and fuel radius (cm)
_PIN_PITCH = 1.26
_FUEL_RADIUS = 0.54
_ASSEMBLY_N = 17  # 17x17 lattice

# Guide tube positions (row, col) -- 0-indexed.
# 24 guide tubes filled with moderator-like material (material 5).
_GUIDE_TUBE_POSITIONS = frozenset([
    (2, 5), (2, 8), (2, 11),
    (3, 3), (3, 13),
    (5, 2), (5, 5), (5, 8), (5, 11), (5, 14),
    (8, 2), (8, 5), (8, 11), (8, 14),
    (11, 2), (11, 5), (11, 8), (11, 11), (11, 14),
    (13, 3), (13, 13),
    (14, 5), (14, 8), (14, 11),
])

# Fission chamber position -- center of the assembly.
_FISSION_CHAMBER_POS = (8, 8)


def _assembly_pin_material(row: int, col: int) -> int:
    """Return material index for the pin at (row, col).

    0 = UO2, 4 = fission chamber, 5 = guide tube.
    """
    if (row, col) == _FISSION_CHAMBER_POS:
        return 4
    if (row, col) in _GUIDE_TUBE_POSITIONS:
        return 5
    return 0


def build_c5g7_assembly(engine) -> GeometryData:
    """Build a 17x17 UO2 assembly with reflective BCs (C5G7 benchmark).

    Geometry:
        - 327 surfaces: 18 X-planes + 18 Y-planes + 2 Z-planes + 289 cylinders
        - 578 cells: 289 pin cells (fuel/tube/chamber) + 289 moderator cells
        - Outer planes reflective (infinite lattice approximation)

    Surface index layout:
        0..17   : X-planes at x = i * 1.26  (i = 0..17)
        18..35  : Y-planes at y = j * 1.26  (j = 0..17)
        36      : Z-plane at z = 0.0
        37      : Z-plane at z = 1.0
        38..326 : Cylinder_Z for pin (row, col) = 38 + row * 17 + col

    Cell index layout:
        (row * 17 + col) * 2       : pin interior (fuel / guide tube / fission chamber)
        (row * 17 + col) * 2 + 1   : moderator annulus around pin
    """
    N = _ASSEMBLY_N
    pitch = _PIN_PITCH
    assembly_size = N * pitch  # 21.42 cm

    BC = BoundaryCondition
    surfaces: list[Surface] = []

    # --- X-planes: indices 0..17 ---
    for i in range(N + 1):
        x = i * pitch
        if i == 0 or i == N:
            bc = BC.REFLECTIVE
        else:
            bc = BC.TRANSMISSIVE
        surfaces.append(Surface.plane_x(x, bc=bc))

    # --- Y-planes: indices 18..35 ---
    for j in range(N + 1):
        y = j * pitch
        if j == 0 or j == N:
            bc = BC.REFLECTIVE
        else:
            bc = BC.TRANSMISSIVE
        surfaces.append(Surface.plane_y(y, bc=bc))

    # --- Z-planes: indices 36, 37 ---
    surfaces.append(Surface.plane_z(0.0, bc=BC.REFLECTIVE))   # 36
    surfaces.append(Surface.plane_z(1.0, bc=BC.REFLECTIVE))   # 37

    # --- Cylinders: indices 38..326 ---
    # One per pin position, centered at (col * pitch + pitch/2, row * pitch + pitch/2)
    half = pitch / 2.0
    for row in range(N):
        for col in range(N):
            cx = col * pitch + half
            cy = row * pitch + half
            surfaces.append(Surface.cylinder_z(cx, cy, _FUEL_RADIUS, bc=BC.TRANSMISSIVE))

    # Assign IDs
    for idx, s in enumerate(surfaces):
        s.id = idx

    assert len(surfaces) == 327, f"Expected 327 surfaces, got {len(surfaces)}"

    # Surface index helpers
    x_plane_base = 0
    y_plane_base = N + 1       # 18
    z_lo = 2 * (N + 1)         # 36
    z_hi = z_lo + 1            # 37
    cyl_base = z_hi + 1        # 38

    # --- Build cells ---
    cells: list[Cell] = []

    for row in range(N):
        for col in range(N):
            left_x  = x_plane_base + col
            right_x = x_plane_base + col + 1
            bot_y   = y_plane_base + row
            top_y   = y_plane_base + row + 1
            cyl_idx = cyl_base + row * N + col

            pin_mat = _assembly_pin_material(row, col)

            # Bounding box surfaces shared by both pin interior and moderator
            bbox = [
                (left_x,  +1),  # x > left plane
                (right_x, -1),  # x < right plane
                (bot_y,   +1),  # y > bottom plane
                (top_y,   -1),  # y < top plane
                (z_lo,    +1),  # z > z_lo
                (z_hi,    -1),  # z < z_hi
            ]

            # Pin interior: inside cylinder (sense -1)
            cells.append(Cell(material_index=pin_mat, surfaces=bbox + [(cyl_idx, -1)]))

            # Moderator annulus: outside cylinder (sense +1), inside bounding box
            cells.append(Cell(material_index=6, surfaces=bbox + [(cyl_idx, +1)]))

    assert len(cells) == 578, f"Expected 578 cells, got {len(cells)}"

    return GeometryData.build(engine, surfaces, cells)


def assembly_source_sampler(num_particles: int, num_groups: int = 7):
    """Generate fission source particles across all fuel pins in the 17x17 assembly.

    Sources are uniformly distributed across UO2 fuel pins and the fission
    chamber pin (all positions with non-zero fission cross sections).
    Energy groups sampled from UO2 chi spectrum.

    Returns: List of ((x, y, z), energy_group) tuples.
    """
    import random

    N = _ASSEMBLY_N
    pitch = _PIN_PITCH
    half = pitch / 2.0
    r = _FUEL_RADIUS
    r2 = r * r

    # Build list of fuel pin centers (all pins with fissile material)
    fuel_pins = []
    for row in range(N):
        for col in range(N):
            mat = _assembly_pin_material(row, col)
            # UO2 (0) and fission chamber (4) have fission XS
            if mat in (0, 4):
                cx = col * pitch + half
                cy = row * pitch + half
                fuel_pins.append((cx, cy))

    num_fuel_pins = len(fuel_pins)  # 265 = 264 UO2 + 1 fission chamber

    # UO2 chi CDF: [0.587910, 0.411760, 0.000340, 0, 0, 0, 0]
    chi_cdf = [0.58791, 0.58791 + 0.41176, 1.0]

    sources = []
    while len(sources) < num_particles:
        # Pick a random fuel pin
        pin_idx = random.randint(0, num_fuel_pins - 1)
        cx, cy = fuel_pins[pin_idx]

        # Sample uniformly inside fuel cylinder via rejection
        x = random.uniform(cx - r, cx + r)
        y = random.uniform(cy - r, cy + r)
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
