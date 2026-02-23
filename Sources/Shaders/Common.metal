#include <metal_stdlib>
using namespace metal;

// =============================================================================
// OpenMC-Metal: Common Definitions
// =============================================================================
// ALL struct layouts here MUST match Swift Types.swift byte-for-byte.
// Metal float3 is 16 bytes (aligned to float4). We use explicit padding
// fields to ensure identical layout between Swift SIMD3<Float>+pad and
// Metal float3 (which is implicitly padded to 16 bytes).
// =============================================================================

// MARK: - Constants

constant int MAX_ENERGY_GROUPS = 7;
constant int MAX_SURFACES_PER_CELL = 16;
constant float BUMP_DISTANCE = 1.0e-8f;

// MARK: - Enums

enum ParticleEventType : uint {
    EVENT_INITIALIZE = 0,
    EVENT_XS_LOOKUP  = 1,
    EVENT_DISTANCE   = 2,
    EVENT_MOVE       = 3,
    EVENT_COLLIDE    = 4,
    EVENT_TALLY      = 5,
    EVENT_DEAD       = 6,
    EVENT_CENSUS     = 7
};

enum SurfaceTypeEnum : uint {
    SURFACE_PLANE_X    = 0,
    SURFACE_PLANE_Y    = 1,
    SURFACE_PLANE_Z    = 2,
    SURFACE_CYLINDER_Z = 3,
    SURFACE_SPHERE     = 4
};

enum BoundaryConditionEnum : uint {
    BC_VACUUM     = 0,
    BC_REFLECTIVE = 1
};

// MARK: - GPU Structs

/// Particle state. Must match Swift `Particle` layout exactly.
/// Total size: 112 bytes (28 x 4-byte fields, with float3 occupying 16 bytes each)
struct Particle {
    float3 position;            // 16 bytes (float3 is padded to 16 in Metal)
    float _pad0;                // explicit pad for Swift compatibility

    float3 direction;           // 16 bytes
    float _pad1;                // explicit pad for Swift compatibility

    uint   energyGroup;
    float  weight;
    uint   cellIndex;
    uint   alive;

    uint   event;               // ParticleEventType
    uint   rngCounter;
    uint   rngKey;
    float  distanceToCollision;

    float  distanceToBoundary;
    int    boundarySurface;
    float  distanceTraveled;
    float  xsTotal;

    float  xsScatter;
    float  xsFission;
    float  xsNuFission;
    float  xsAbsorption;

    uint   fissionFlag;
    uint   materialIndex;
    uint   _pad2;
    uint   _pad3;
};

/// Surface definition. Must match Swift `GPUSurface`.
struct GPUSurface {
    uint   type;                // SurfaceTypeEnum
    uint   boundaryCondition;   // BoundaryConditionEnum
    float4 coefficients;
    // planeX/Y/Z: position = coefficients.x
    // cylinderZ:  x0=.x, y0=.y, R=.z
    // sphere:     x0=.x, y0=.y, z0=.z, R=.w
    uint   _pad0;
    uint   _pad1;
};

/// Cell definition. Must match Swift `GPUCell`.
struct GPUCell {
    uint materialIndex;
    uint numSurfaces;
    uint surfaceOffset;         // Index into GPUCellSurface array
    uint _pad;
};

/// Cell-surface association. Must match Swift `GPUCellSurface`.
struct GPUCellSurface {
    uint surfaceIndex;
    int  sense;                 // +1 or -1
};

/// Lattice definition. Must match Swift `GPULattice`.
struct GPULattice {
    float3 origin;
    float  _pad0;
    float3 pitch;
    float  _pad1;
    uint3  dimensions;
    uint   _pad2;
};

/// Multi-group cross sections for one material.
/// Must match Swift `MaterialXS`.
struct MaterialXS {
    float total[MAX_ENERGY_GROUPS];                              // 7 floats
    float scatter[MAX_ENERGY_GROUPS * MAX_ENERGY_GROUPS];        // 49 floats
    float fission[MAX_ENERGY_GROUPS];                            // 7 floats
    float nuFission[MAX_ENERGY_GROUPS];                          // 7 floats
    float chi[MAX_ENERGY_GROUPS];                                // 7 floats
};

/// Simulation parameters. Must match Swift `SimulationParams`.
struct SimParams {
    uint  numParticles;
    uint  numBatches;
    uint  numInactive;
    uint  numGroups;
    uint  numCells;
    uint  numSurfaces;
    uint  numMaterials;
    float kEff;
};

// =============================================================================
// Philox-2x32-10 Counter-Based RNG
// =============================================================================
// Philox is a stateless CBRNG: output = f(counter, key).
// Each particle uses its index as key and increments counter per draw.
// This guarantees independent, reproducible streams per particle.
// =============================================================================

constant uint PHILOX_M = 0xD2511F53;
constant uint PHILOX_W = 0x9E3779B9;

/// Single Philox round.
inline uint2 philox2x32_round(uint2 ctr, uint key) {
    uint hi = mulhi(PHILOX_M, ctr.x);
    uint lo = ctr.x * PHILOX_M;
    return uint2(hi ^ key ^ ctr.y, lo);
}

/// Full Philox-2x32-10: 10 rounds of mixing.
inline uint2 philox2x32_10(uint2 counter, uint key) {
    uint2 ctr = counter;
    uint k = key;
    for (int i = 0; i < 10; i++) {
        ctr = philox2x32_round(ctr, k);
        k += PHILOX_W;
    }
    return ctr;
}

/// Draw a uniform float in [0, 1) and advance the counter.
inline float philox_uniform(thread uint& counter_lo, uint counter_hi, uint key) {
    uint2 result = philox2x32_10(uint2(counter_lo, counter_hi), key);
    counter_lo++;
    return float(result.x) * 2.3283064365e-10f;  // 1.0 / 2^32
}

// =============================================================================
// Atomic Float Addition Helper
// =============================================================================
// Uses a CAS loop to atomically add a float delta to a device atomic_uint
// (float stored as uint bits). Correct even under high contention.
// Shared by tally_score and collision_and_tally kernels.
// =============================================================================

inline void atomic_add_float(device atomic_uint* target, float delta) {
    uint expected = atomic_load_explicit(target, memory_order_relaxed);
    while (true) {
        float  current = as_type<float>(expected);
        float  newVal  = current + delta;
        uint   desired = as_type<uint>(newVal);
        if (atomic_compare_exchange_weak_explicit(
                target, &expected, desired,
                memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
    }
}
