"""
Metal GPU communication struct layout definitions.

All structs use little-endian byte order and must match the Swift/Metal
struct layouts byte-for-byte for correct CPU-GPU memory sharing.

Format characters:
  I = uint32 (unsigned int32)
  i = int32  (signed int32)
  f = float32
"""

import struct
from enum import IntEnum


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_ENERGY_GROUPS: int = 7
MAX_SURFACES_PER_CELL: int = 16
BUMP_DISTANCE: float = 1e-8


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ParticleEvent(IntEnum):
    """Particle event codes stored in Particle.event (must match Metal ParticleEventType)."""
    INITIALIZE  = 0
    XS_LOOKUP   = 1
    DISTANCE    = 2
    MOVE        = 3
    COLLIDE     = 4
    TALLY       = 5
    DEAD        = 6
    CENSUS      = 7


class SurfaceType(IntEnum):
    """Geometry surface types stored in GPUSurface.type (must match Metal SurfaceTypeEnum)."""
    PLANE_X     = 0
    PLANE_Y     = 1
    PLANE_Z     = 2
    CYLINDER_Z  = 3
    SPHERE      = 4


class BoundaryCondition(IntEnum):
    """Boundary condition types stored in GPUSurface.boundaryCondition."""
    VACUUM          = 0
    REFLECTIVE      = 1
    TRANSMISSIVE    = 2  # Internal surface: particles pass through


# ---------------------------------------------------------------------------
# Struct format strings (little-endian '<')
# ---------------------------------------------------------------------------

# Particle: 112 bytes = 28 x 4-byte fields
#   float3 position   : posX posY posZ _pad0       (4f = 16B)
#   float3 direction  : dirX dirY dirZ _pad1        (4f = 16B)
#   misc              : energyGroup weight cellIndex alive  (IIfI = 16B)
#   transport         : event rngCounter rngKey distanceToCollision (IIIf = 16B)
#   boundary          : distanceToBoundary boundarySurface distanceTraveled xsTotal (fifF -> fiFF)
#                       note boundarySurface is signed int32 (i)          (16B)
#   cross sections 1  : xsScatter xsFission xsNuFission xsAbsorption      (4f = 16B)
#   fission / padding : fissionFlag materialIndex _pad2 _pad3             (4I = 16B)
PARTICLE_FORMAT: str = "<" + (
    "4f"   # posX posY posZ _pad0
    "4f"   # dirX dirY dirZ _pad1
    "I"    # energyGroup
    "f"    # weight
    "I"    # cellIndex
    "I"    # alive
    "I"    # event
    "I"    # rngCounter
    "I"    # rngKey
    "f"    # distanceToCollision
    "f"    # distanceToBoundary
    "i"    # boundarySurface  (signed int32)
    "f"    # distanceTraveled
    "f"    # xsTotal
    "f"    # xsScatter
    "f"    # xsFission
    "f"    # xsNuFission
    "f"    # xsAbsorption
    "I"    # fissionFlag
    "I"    # materialIndex
    "I"    # _pad2
    "I"    # _pad3
)
PARTICLE_SIZE: int = 112

# GPUSurface: 48 bytes (Metal float4 requires 16-byte alignment, creating
# an 8-byte gap after the two uint fields, plus 8 bytes of struct trailing padding)
#   type boundaryCondition [8-byte gap] coeffX coeffY coeffZ coeffW _pad0 _pad1 [8-byte trailing]
GPUSURFACE_FORMAT: str = "<II8xffffII8x"
GPUSURFACE_SIZE: int = 48

# GPUCell: 16 bytes
#   materialIndex numSurfaces surfaceOffset _pad
GPUCELL_FORMAT: str = "<IIII"
GPUCELL_SIZE: int = 16

# GPUCellSurface: 8 bytes
#   surfaceIndex sense   (sense is signed int32)
GPUCELLSURFACE_FORMAT: str = "<Ii"
GPUCELLSURFACE_SIZE: int = 8

# SimulationParams: 32 bytes
#   numParticles numBatches numInactive numGroups numCells numSurfaces numMaterials kEff
SIMULATIONPARAMS_FORMAT: str = "<IIIIIIIf"
SIMULATIONPARAMS_SIZE: int = 32

# XSBenchParams: 16 bytes
#   numLookups numMaterials numGroups _pad
XSBENCHPARAMS_FORMAT: str = "<IIII"
XSBENCHPARAMS_SIZE: int = 16

# FissionSite: 32 bytes
#   posX posY posZ _pad0  energyGroup _pad1 _pad2 _pad3
FISSIONSITE_FORMAT: str = "<ffffIIII"
FISSIONSITE_SIZE: int = 32


# ---------------------------------------------------------------------------
# Size assertions: verify struct formats match expected byte counts
# ---------------------------------------------------------------------------

assert struct.calcsize(PARTICLE_FORMAT) == PARTICLE_SIZE, (
    f"Particle size mismatch: got {struct.calcsize(PARTICLE_FORMAT)}, expected {PARTICLE_SIZE}"
)
assert struct.calcsize(GPUSURFACE_FORMAT) == GPUSURFACE_SIZE, (
    f"GPUSurface size mismatch: got {struct.calcsize(GPUSURFACE_FORMAT)}, expected {GPUSURFACE_SIZE}. "
    f"Metal float4 in GPUSurface requires 16-byte alignment padding."
)
assert struct.calcsize(GPUCELL_FORMAT) == GPUCELL_SIZE, (
    f"GPUCell size mismatch: got {struct.calcsize(GPUCELL_FORMAT)}, expected {GPUCELL_SIZE}"
)
assert struct.calcsize(GPUCELLSURFACE_FORMAT) == GPUCELLSURFACE_SIZE, (
    f"GPUCellSurface size mismatch: got {struct.calcsize(GPUCELLSURFACE_FORMAT)}, expected {GPUCELLSURFACE_SIZE}"
)
assert struct.calcsize(SIMULATIONPARAMS_FORMAT) == SIMULATIONPARAMS_SIZE, (
    f"SimulationParams size mismatch: got {struct.calcsize(SIMULATIONPARAMS_FORMAT)}, expected {SIMULATIONPARAMS_SIZE}"
)
assert struct.calcsize(XSBENCHPARAMS_FORMAT) == XSBENCHPARAMS_SIZE, (
    f"XSBenchParams size mismatch: got {struct.calcsize(XSBENCHPARAMS_FORMAT)}, expected {XSBENCHPARAMS_SIZE}"
)
assert struct.calcsize(FISSIONSITE_FORMAT) == FISSIONSITE_SIZE, (
    f"FissionSite size mismatch: got {struct.calcsize(FISSIONSITE_FORMAT)}, expected {FISSIONSITE_SIZE}"
)


# ---------------------------------------------------------------------------
# Particle helpers
# ---------------------------------------------------------------------------

def pack_particle(
    pos_x: float,
    pos_y: float,
    pos_z: float,
    dir_x: float,
    dir_y: float,
    dir_z: float,
    energy_group: int,
    weight: float,
    cell_index: int,
    alive: int,
    event: int,
    rng_counter: int,
    rng_key: int,
    distance_to_collision: float,
    distance_to_boundary: float,
    boundary_surface: int,
    distance_traveled: float,
    xs_total: float,
    xs_scatter: float,
    xs_fission: float,
    xs_nu_fission: float,
    xs_absorption: float,
    fission_flag: int,
    material_index: int,
    *,
    pad0: float = 0.0,
    pad1: float = 0.0,
    pad2: int = 0,
    pad3: int = 0,
) -> bytes:
    """Pack a Particle into 112-byte little-endian bytes."""
    return struct.pack(
        PARTICLE_FORMAT,
        pos_x, pos_y, pos_z, pad0,
        dir_x, dir_y, dir_z, pad1,
        energy_group,
        weight,
        cell_index,
        alive,
        event,
        rng_counter,
        rng_key,
        distance_to_collision,
        distance_to_boundary,
        boundary_surface,
        distance_traveled,
        xs_total,
        xs_scatter,
        xs_fission,
        xs_nu_fission,
        xs_absorption,
        fission_flag,
        material_index,
        pad2,
        pad3,
    )


def unpack_particle(data: bytes) -> dict:
    """Unpack 112-byte little-endian bytes into a Particle dict."""
    if len(data) != PARTICLE_SIZE:
        raise ValueError(
            f"Expected {PARTICLE_SIZE} bytes for Particle, got {len(data)}"
        )
    fields = struct.unpack(PARTICLE_FORMAT, data)
    return {
        "pos_x":                  fields[0],
        "pos_y":                  fields[1],
        "pos_z":                  fields[2],
        "_pad0":                  fields[3],
        "dir_x":                  fields[4],
        "dir_y":                  fields[5],
        "dir_z":                  fields[6],
        "_pad1":                  fields[7],
        "energy_group":           fields[8],
        "weight":                 fields[9],
        "cell_index":             fields[10],
        "alive":                  fields[11],
        "event":                  fields[12],
        "rng_counter":            fields[13],
        "rng_key":                fields[14],
        "distance_to_collision":  fields[15],
        "distance_to_boundary":   fields[16],
        "boundary_surface":       fields[17],
        "distance_traveled":      fields[18],
        "xs_total":               fields[19],
        "xs_scatter":             fields[20],
        "xs_fission":             fields[21],
        "xs_nu_fission":          fields[22],
        "xs_absorption":          fields[23],
        "fission_flag":           fields[24],
        "material_index":         fields[25],
        "_pad2":                  fields[26],
        "_pad3":                  fields[27],
    }


# ---------------------------------------------------------------------------
# SimulationParams helpers
# ---------------------------------------------------------------------------

def pack_simulation_params(
    num_particles: int,
    num_batches: int,
    num_inactive: int,
    num_groups: int,
    num_cells: int,
    num_surfaces: int,
    num_materials: int,
    k_eff: float,
) -> bytes:
    """Pack SimulationParams into 32-byte little-endian bytes."""
    return struct.pack(
        SIMULATIONPARAMS_FORMAT,
        num_particles,
        num_batches,
        num_inactive,
        num_groups,
        num_cells,
        num_surfaces,
        num_materials,
        k_eff,
    )


def unpack_simulation_params(data: bytes) -> dict:
    """Unpack 32-byte little-endian bytes into a SimulationParams dict."""
    if len(data) != SIMULATIONPARAMS_SIZE:
        raise ValueError(
            f"Expected {SIMULATIONPARAMS_SIZE} bytes for SimulationParams, got {len(data)}"
        )
    fields = struct.unpack(SIMULATIONPARAMS_FORMAT, data)
    return {
        "num_particles":  fields[0],
        "num_batches":    fields[1],
        "num_inactive":   fields[2],
        "num_groups":     fields[3],
        "num_cells":      fields[4],
        "num_surfaces":   fields[5],
        "num_materials":  fields[6],
        "k_eff":          fields[7],
    }
