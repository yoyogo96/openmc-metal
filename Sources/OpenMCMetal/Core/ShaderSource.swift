// ShaderSource.swift
// Embedded Metal shader source for runtime compilation.
// This allows the CLI tool to compile shaders without a bundle.
//
// Concatenation order:
//   1. Common.metal   – structs, constants, RNG
//   2. Geometry.metal – surface/cell/lattice geometry
//   3. XSLookup.metal – cross-section lookup kernels
//   4. Transport.metal – init/distance/move kernels
//   5. Collision.metal – collision/scatter kernel
//   6. Tally.metal    – tally scoring kernel
//
// Duplicate constants that appear in both XSLookup.metal and Collision.metal
// are kept only once (in the XSLookup section).

enum ShaderSource {
    static let allShaders: String = #"""
#include <metal_stdlib>
using namespace metal;

// =============================================================================
// OpenMC-Metal: Common Definitions
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
struct Particle {
    float3 position;
    float _pad0;

    float3 direction;
    float _pad1;

    uint   energyGroup;
    float  weight;
    uint   cellIndex;
    uint   alive;

    uint   event;
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
    uint   type;
    uint   boundaryCondition;
    float4 coefficients;
    uint   _pad0;
    uint   _pad1;
};

/// Cell definition. Must match Swift `GPUCell`.
struct GPUCell {
    uint materialIndex;
    uint numSurfaces;
    uint surfaceOffset;
    uint _pad;
};

/// Cell-surface association. Must match Swift `GPUCellSurface`.
struct GPUCellSurface {
    uint surfaceIndex;
    int  sense;
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
struct MaterialXS {
    float total[MAX_ENERGY_GROUPS];
    float scatter[MAX_ENERGY_GROUPS * MAX_ENERGY_GROUPS];
    float fission[MAX_ENERGY_GROUPS];
    float nuFission[MAX_ENERGY_GROUPS];
    float chi[MAX_ENERGY_GROUPS];
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

constant uint PHILOX_M = 0xD2511F53;
constant uint PHILOX_W = 0x9E3779B9;

inline uint2 philox2x32_round(uint2 ctr, uint key) {
    uint hi = mulhi(PHILOX_M, ctr.x);
    uint lo = ctr.x * PHILOX_M;
    return uint2(hi ^ key ^ ctr.y, lo);
}

inline uint2 philox2x32_10(uint2 counter, uint key) {
    uint2 ctr = counter;
    uint k = key;
    for (int i = 0; i < 10; i++) {
        ctr = philox2x32_round(ctr, k);
        k += PHILOX_W;
    }
    return ctr;
}

inline float philox_uniform(thread uint& counter_lo, uint counter_hi, uint key) {
    uint2 result = philox2x32_10(uint2(counter_lo, counter_hi), key);
    counter_lo++;
    return float(result.x) * 2.3283064365e-10f;
}

// =============================================================================
// Geometry
// =============================================================================

constant float BUMP     = 1.0e-8f;
constant float INF_DIST = 1.0e20f;

inline int evaluate_surface(float3 pos, device const GPUSurface& surface) {
    float val = 0.0f;

    switch (surface.type) {
        case 0: {
            val = pos.x - surface.coefficients.x;
            break;
        }
        case 1: {
            val = pos.y - surface.coefficients.y;
            break;
        }
        case 2: {
            val = pos.z - surface.coefficients.z;
            break;
        }
        case 3: {
            float dx = pos.x - surface.coefficients.x;
            float dy = pos.y - surface.coefficients.y;
            float R  = surface.coefficients.z;
            val = dx*dx + dy*dy - R*R;
            break;
        }
        case 4: {
            float dx = pos.x - surface.coefficients.x;
            float dy = pos.y - surface.coefficients.y;
            float dz = pos.z - surface.coefficients.z;
            float R  = surface.coefficients.w;
            val = dx*dx + dy*dy + dz*dz - R*R;
            break;
        }
        default:
            val = 0.0f;
            break;
    }

    return (val >= 0.0f) ? 1 : -1;
}

inline float distance_to_surface(float3 pos, float3 dir, device const GPUSurface& surface) {

    switch (surface.type) {

        case 0: {
            if (abs(dir.x) < 1.0e-10f) return INF_DIST;
            float d = (surface.coefficients.x - pos.x) / dir.x;
            return (d > BUMP) ? d : INF_DIST;
        }

        case 1: {
            if (abs(dir.y) < 1.0e-10f) return INF_DIST;
            float d = (surface.coefficients.y - pos.y) / dir.y;
            return (d > BUMP) ? d : INF_DIST;
        }

        case 2: {
            if (abs(dir.z) < 1.0e-10f) return INF_DIST;
            float d = (surface.coefficients.z - pos.z) / dir.z;
            return (d > BUMP) ? d : INF_DIST;
        }

        case 3: {
            float x0 = surface.coefficients.x;
            float y0 = surface.coefficients.y;
            float R  = surface.coefficients.z;

            float dx = pos.x - x0;
            float dy = pos.y - y0;

            float A = dir.x*dir.x + dir.y*dir.y;
            if (A < 1.0e-10f) return INF_DIST;

            float B    = 2.0f * (dx*dir.x + dy*dir.y);
            float C    = dx*dx + dy*dy - R*R;
            float disc = B*B - 4.0f*A*C;

            if (disc < 0.0f) return INF_DIST;

            float sq = sqrt(disc);
            float inv2A = 1.0f / (2.0f * A);
            float d1 = (-B - sq) * inv2A;
            float d2 = (-B + sq) * inv2A;

            float result = INF_DIST;
            if (d1 > BUMP) result = d1;
            if (d2 > BUMP && d2 < result) result = d2;
            return result;
        }

        case 4: {
            float3 center = float3(
                surface.coefficients.x,
                surface.coefficients.y,
                surface.coefficients.z);
            float R = surface.coefficients.w;

            float3 oc  = pos - center;
            float B    = 2.0f * dot(oc, dir);
            float C    = dot(oc, oc) - R*R;
            float disc = B*B - 4.0f*C;

            if (disc < 0.0f) return INF_DIST;

            float sq = sqrt(disc);
            float d1 = (-B - sq) * 0.5f;
            float d2 = (-B + sq) * 0.5f;

            float result = INF_DIST;
            if (d1 > BUMP) result = d1;
            if (d2 > BUMP && d2 < result) result = d2;
            return result;
        }
    }

    return INF_DIST;
}

inline bool point_in_cell(float3 pos,
                           device const GPUCell&        cell,
                           device const GPUCellSurface* cellSurfaces,
                           device const GPUSurface*     surfaces) {
    for (uint i = 0; i < cell.numSurfaces; i++) {
        uint csIdx  = cell.surfaceOffset + i;
        int  actual = evaluate_surface(pos, surfaces[cellSurfaces[csIdx].surfaceIndex]);
        if (actual != cellSurfaces[csIdx].sense) return false;
    }
    return true;
}

inline int find_cell(float3 pos,
                     device const GPUCell*        cells,
                     uint                          numCells,
                     device const GPUCellSurface* cellSurfaces,
                     device const GPUSurface*     surfaces) {
    for (uint i = 0; i < numCells; i++) {
        if (point_in_cell(pos, cells[i], cellSurfaces, surfaces)) {
            return int(i);
        }
    }
    return -1;
}

inline float2 distance_to_boundary(float3 pos,
                                    float3 dir,
                                    device const GPUCell&        cell,
                                    device const GPUCellSurface* cellSurfaces,
                                    device const GPUSurface*     surfaces) {
    float minDist      = INF_DIST;
    int   crossedSurf  = -1;

    for (uint i = 0; i < cell.numSurfaces; i++) {
        uint csIdx   = cell.surfaceOffset + i;
        uint surfIdx = cellSurfaces[csIdx].surfaceIndex;

        float d = distance_to_surface(pos, dir, surfaces[surfIdx]);
        if (d < minDist) {
            minDist     = d;
            crossedSurf = int(surfIdx);
        }
    }

    return float2(minDist, as_type<float>(crossedSurf));
}

inline uint3 lattice_element(float3 pos, device const GPULattice& lattice) {
    float3 local = (pos - lattice.origin) / lattice.pitch;
    int3   idx   = int3(floor(local));
    int3   dmax  = int3(lattice.dimensions) - 1;
    idx = clamp(idx, int3(0), dmax);
    return uint3(idx);
}

inline float3 lattice_local_pos(float3 pos, uint3 element, device const GPULattice& lattice) {
    float3 element_origin = lattice.origin + float3(element) * lattice.pitch;
    return pos - element_origin;
}

inline float distance_to_lattice_boundary(float3               pos,
                                           float3               dir,
                                           uint3                element,
                                           device const GPULattice& lattice) {
    float3 elem_origin = lattice.origin + float3(element) * lattice.pitch;
    float3 elem_max    = elem_origin + lattice.pitch;

    float dmin = INF_DIST;

    if (abs(dir.x) > 1.0e-10f) {
        float d = (dir.x > 0.0f)
            ? (elem_max.x   - pos.x) / dir.x
            : (elem_origin.x - pos.x) / dir.x;
        if (d > BUMP && d < dmin) dmin = d;
    }

    if (abs(dir.y) > 1.0e-10f) {
        float d = (dir.y > 0.0f)
            ? (elem_max.y   - pos.y) / dir.y
            : (elem_origin.y - pos.y) / dir.y;
        if (d > BUMP && d < dmin) dmin = d;
    }

    if (abs(dir.z) > 1.0e-10f) {
        float d = (dir.z > 0.0f)
            ? (elem_max.z   - pos.z) / dir.z
            : (elem_origin.z - pos.z) / dir.z;
        if (d > BUMP && d < dmin) dmin = d;
    }

    return dmin;
}

inline uint lattice_cell_offset(uint3                    element,
                                 device const GPULattice& lattice,
                                 device const uint*       universeMap) {
    uint nx    = lattice.dimensions.x;
    uint ny    = lattice.dimensions.y;
    uint flatI = element.z * (nx * ny) + element.y * nx + element.x;
    return universeMap[flatI];
}

inline float3 surface_normal(float3 pos, device const GPUSurface& surface) {
    switch (surface.type) {
        case 0: return float3(1.0f, 0.0f, 0.0f);
        case 1: return float3(0.0f, 1.0f, 0.0f);
        case 2: return float3(0.0f, 0.0f, 1.0f);
        case 3: {
            float2 n = float2(pos.x - surface.coefficients.x,
                              pos.y - surface.coefficients.y);
            float len = length(n);
            return (len > 1.0e-12f)
                ? float3(n / len, 0.0f)
                : float3(1.0f, 0.0f, 0.0f);
        }
        case 4: {
            float3 n = pos - float3(surface.coefficients.x,
                                    surface.coefficients.y,
                                    surface.coefficients.z);
            float len = length(n);
            return (len > 1.0e-12f) ? n / len : float3(0.0f, 0.0f, 1.0f);
        }
        default:
            return float3(0.0f, 0.0f, 1.0f);
    }
}

// =============================================================================
// XS Lookup
// =============================================================================

constant uint FLOATS_PER_MATERIAL = 77;
constant uint OFFSET_TOTAL        = 0;
constant uint OFFSET_SCATTER      = 7;
constant uint OFFSET_FISSION      = 56;
constant uint OFFSET_NU_FISSION   = 63;
constant uint OFFSET_CHI          = 70;

kernel void xs_lookup(device Particle* particles       [[buffer(0)]],
                      device const float* materials    [[buffer(1)]],
                      device const GPUCell* cells      [[buffer(2)]],
                      device const SimParams& params   [[buffer(3)]],
                      uint tid [[thread_position_in_grid]]) {
    if (tid >= params.numParticles) return;

    device Particle& p = particles[tid];
    if (p.alive == 0 || p.event != EVENT_XS_LOOKUP) return;

    uint cellIdx = p.cellIndex;
    uint matIdx  = cells[cellIdx].materialIndex;
    uint group   = p.energyGroup;

    uint matOffset = matIdx * FLOATS_PER_MATERIAL;

    p.xsTotal     = materials[matOffset + OFFSET_TOTAL + group];
    p.xsFission   = materials[matOffset + OFFSET_FISSION + group];
    p.xsNuFission = materials[matOffset + OFFSET_NU_FISSION + group];
    p.materialIndex = matIdx;

    float scatterTotal = 0.0f;
    uint scatterRowStart = matOffset + OFFSET_SCATTER + group * params.numGroups;
    for (uint g = 0; g < params.numGroups; g++) {
        scatterTotal += materials[scatterRowStart + g];
    }
    p.xsScatter    = scatterTotal;
    p.xsAbsorption = p.xsTotal - scatterTotal;

    p.event = EVENT_DISTANCE;
}

struct XSBenchParams {
    uint numLookups;
    uint numMaterials;
    uint numGroups;
    uint _pad;
};

kernel void xsbench_lookup(device const uint*            lookups   [[buffer(0)]],
                           device const float*           materials [[buffer(1)]],
                           device float*                 output    [[buffer(2)]],
                           device const XSBenchParams&   params    [[buffer(3)]],
                           uint tid [[thread_position_in_grid]]) {
    if (tid >= params.numLookups) return;

    uint matIdx = lookups[tid * 2];
    uint group  = lookups[tid * 2 + 1];

    uint matBase = matIdx * FLOATS_PER_MATERIAL;

    float total = materials[matBase + OFFSET_TOTAL + group];

    float scatterSum = 0.0f;
    uint scatterRowBase = matBase + OFFSET_SCATTER + group * params.numGroups;
    for (uint g = 0; g < params.numGroups; g++) {
        scatterSum += materials[scatterRowBase + g];
    }

    output[tid] = total + scatterSum;
}

// =============================================================================
// Transport (init_particles, distance_to_collision, move_particle)
// =============================================================================

kernel void init_particles(
    device Particle* particles [[buffer(0)]],
    device const SimParams& params [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.numParticles) return;

    uint rng_counter = 0;
    uint rng_hi = tid;
    uint rng_key = tid;

    float cosTheta = 2.0f * philox_uniform(rng_counter, rng_hi, rng_key) - 1.0f;
    float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
    float phi = 2.0f * M_PI_F * philox_uniform(rng_counter, rng_hi, rng_key);

    float3 dir = float3(
        sinTheta * cos(phi),
        sinTheta * sin(phi),
        cosTheta
    );

    Particle p;
    p.position = float3(0.0f, 0.0f, 0.0f);
    p._pad0 = 0.0f;
    p.direction = dir;
    p._pad1 = 0.0f;

    p.energyGroup = 0;
    p.weight = 1.0f;
    p.cellIndex = 0;
    p.alive = 1;

    p.event = EVENT_XS_LOOKUP;
    p.rngCounter = rng_counter;
    p.rngKey = rng_key;
    p.distanceToCollision = 0.0f;

    p.distanceToBoundary = 0.0f;
    p.boundarySurface = -1;
    p.distanceTraveled = 0.0f;
    p.xsTotal = 0.0f;

    p.xsScatter = 0.0f;
    p.xsFission = 0.0f;
    p.xsNuFission = 0.0f;
    p.xsAbsorption = 0.0f;

    p.fissionFlag = 0;
    p.materialIndex = 0;
    p._pad2 = 0;
    p._pad3 = 0;

    particles[tid] = p;
}

kernel void distance_to_collision(
    device Particle* particles       [[buffer(0)]],
    device const SimParams& params   [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.numParticles) return;

    device Particle& p = particles[tid];
    if (p.alive == 0 || p.event != EVENT_DISTANCE) return;

    if (p.xsTotal <= 0.0f) {
        p.alive = 0;
        p.event = EVENT_DEAD;
        return;
    }

    uint rng_counter = p.rngCounter;
    float xi = philox_uniform(rng_counter, tid, p.rngKey);
    p.rngCounter = rng_counter;

    p.distanceToCollision = -log(xi) / p.xsTotal;
    p.event = EVENT_MOVE;
}

kernel void move_particle(
    device Particle* particles                [[buffer(0)]],
    device const GPUSurface* surfaces         [[buffer(1)]],
    device const GPUCell* cells               [[buffer(2)]],
    device const GPUCellSurface* cellSurfaces [[buffer(3)]],
    device const SimParams& params            [[buffer(4)]],
    device atomic_uint* lostParticleCount     [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.numParticles) return;

    device Particle& p = particles[tid];
    if (p.alive == 0 || p.event != EVENT_MOVE) return;

    float2 boundaryResult = distance_to_boundary(p.position, p.direction,
                                                  cells[p.cellIndex],
                                                  cellSurfaces, surfaces);
    float dBoundary = boundaryResult.x;
    int crossedSurfIdx = as_type<int>(boundaryResult.y);

    p.distanceToBoundary = dBoundary;
    p.boundarySurface = crossedSurfIdx;

    if (p.distanceToCollision < dBoundary) {
        float d = p.distanceToCollision;
        p.position += d * p.direction;
        p.distanceTraveled = d;
        p.event = EVENT_COLLIDE;
    } else {
        float d = dBoundary;
        p.position += (d + BUMP) * p.direction;
        p.distanceTraveled = d;

        if (crossedSurfIdx >= 0 && uint(crossedSurfIdx) < params.numSurfaces) {
            uint bc = surfaces[crossedSurfIdx].boundaryCondition;

            if (bc == BC_VACUUM) {
                p.alive = 0;
                p.event = EVENT_DEAD;
                return;
            } else if (bc == BC_REFLECTIVE) {
                float3 normal = surface_normal(p.position, surfaces[crossedSurfIdx]);
                p.direction = p.direction - 2.0f * dot(p.direction, normal) * normal;
                p.direction = normalize(p.direction);
            }
        }

        int newCell = find_cell(p.position, cells, params.numCells,
                                cellSurfaces, surfaces);
        if (newCell < 0) {
            p.alive = 0;
            p.event = EVENT_DEAD;
            atomic_fetch_add_explicit(lostParticleCount, 1, memory_order_relaxed);
            return;
        }
        p.cellIndex = uint(newCell);

        p.event = EVENT_XS_LOOKUP;
    }
}

// =============================================================================
// Collision
// =============================================================================

struct FissionSite {
    float3 position;
    float  _pad;
    uint   energyGroup;
    uint   _pad2;
    uint   _pad3;
    uint   _pad4;
};

kernel void collision(
    device Particle*        particles    [[buffer(0)]],
    device const float*     materials    [[buffer(1)]],
    device FissionSite*     fissionBank  [[buffer(2)]],
    device atomic_uint*     fissionCount [[buffer(3)]],
    device const SimParams& params       [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.numParticles) return;

    device Particle& p = particles[tid];
    if (p.alive == 0 || p.event != EVENT_COLLIDE) return;

    uint matOffset = p.materialIndex * FLOATS_PER_MATERIAL;
    uint g = p.energyGroup;

    // Copy RNG counter to thread-local variable (philox_uniform needs thread uint&)
    uint rng_counter = p.rngCounter;

    // -------------------------------------------------------------------------
    // Sample collision type: absorption or scatter
    // -------------------------------------------------------------------------
    float xi = philox_uniform(rng_counter, tid, p.rngKey);

    float absorptionProb = (p.xsTotal > 0.0f)
                           ? (p.xsAbsorption / p.xsTotal)
                           : 0.0f;

    if (xi < absorptionProb) {
        // ---------------------------------------------------------------------
        // ABSORPTION
        // ---------------------------------------------------------------------
        if (p.xsFission > 0.0f) {
            float fissionProb = (p.xsAbsorption > 0.0f)
                                ? (p.xsFission / p.xsAbsorption)
                                : 0.0f;
            float xi2 = philox_uniform(rng_counter, tid, p.rngKey);
            if (xi2 < fissionProb) {
                float nu  = (p.xsFission > 0.0f)
                            ? (p.xsNuFission / p.xsFission)
                            : 0.0f;
                float xi3 = philox_uniform(rng_counter, tid, p.rngKey);
                uint numFission = (uint)(nu * p.weight + xi3);

                for (uint f = 0; f < min(numFission, 3u); f++) {
                    uint idx = atomic_fetch_add_explicit(
                        fissionCount, 1u, memory_order_relaxed);

                    if (idx < params.numParticles * 2u) {
                        float xiChi   = philox_uniform(rng_counter, tid, p.rngKey);
                        float cumChi  = 0.0f;
                        uint  fGroup  = params.numGroups - 1u;

                        for (uint gg = 0; gg < params.numGroups; gg++) {
                            cumChi += materials[matOffset + OFFSET_CHI + gg];
                            if (xiChi < cumChi) {
                                fGroup = gg;
                                break;
                            }
                        }

                        fissionBank[idx].position    = p.position;
                        fissionBank[idx]._pad        = 0.0f;
                        fissionBank[idx].energyGroup = fGroup;
                        fissionBank[idx]._pad2       = 0u;
                        fissionBank[idx]._pad3       = 0u;
                        fissionBank[idx]._pad4       = 0u;
                    }
                }
            }
        }

        p.alive = 0u;
        p.event = EVENT_TALLY;

    } else {
        // ---------------------------------------------------------------------
        // SCATTER
        // Sample outgoing energy group from scatter matrix row g.
        // ---------------------------------------------------------------------
        float xiScatter  = philox_uniform(rng_counter, tid, p.rngKey);
        float cumScatter = 0.0f;
        uint  newGroup   = g;

        float scatterRowSum = 0.0f;
        for (uint gg = 0; gg < params.numGroups; gg++) {
            scatterRowSum += materials[matOffset + OFFSET_SCATTER + g * params.numGroups + gg];
        }

        float threshold = xiScatter * ((scatterRowSum > 0.0f) ? scatterRowSum : 1.0f);
        float cumS = 0.0f;
        for (uint gg = 0; gg < params.numGroups; gg++) {
            cumS += materials[matOffset + OFFSET_SCATTER + g * params.numGroups + gg];
            if (cumS >= threshold) {
                newGroup = gg;
                break;
            }
        }
        p.energyGroup = newGroup;

        // Sample new isotropic direction (lab frame).
        float mu      = 2.0f * philox_uniform(rng_counter, tid, p.rngKey) - 1.0f;
        float phi     = 2.0f * M_PI_F * philox_uniform(rng_counter, tid, p.rngKey);
        float sinTheta = sqrt(max(0.0f, 1.0f - mu * mu));

        p.direction = normalize(float3(
            sinTheta * cos(phi),
            sinTheta * sin(phi),
            mu
        ));
        p._pad1 = 0.0f;

        p.event = EVENT_TALLY;
    }

    // Write back RNG counter to device memory
    p.rngCounter = rng_counter;
}

// =============================================================================
// Tally
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

kernel void tally_score(
    device Particle*        particles     [[buffer(0)]],
    device atomic_uint*     tallyFlux     [[buffer(1)]],
    device atomic_uint*     tallyFission  [[buffer(2)]],
    device const SimParams& params        [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.numParticles) return;

    device Particle& p = particles[tid];
    if (p.event != EVENT_TALLY) return;

    if (p.distanceTraveled <= 0.0f || p.cellIndex >= params.numCells) {
        p.event = (p.alive == 1u) ? EVENT_XS_LOOKUP : EVENT_DEAD;
        return;
    }

    uint idx = p.cellIndex * params.numGroups + p.energyGroup;

    float fluxScore    = p.weight * p.distanceTraveled;
    float fissionScore = p.weight * p.distanceTraveled * p.xsFission;

    atomic_add_float(&tallyFlux[idx],    fluxScore);
    atomic_add_float(&tallyFission[idx], fissionScore);

    p.event = (p.alive == 1u) ? EVENT_XS_LOOKUP : EVENT_DEAD;
}
"""#
}
