#include <metal_stdlib>
using namespace metal;

// Include shared definitions (Common.metal is compiled in the same target)
// The structs and RNG functions are available from Common.metal.

// Forward declarations of types and functions from Common.metal
// (Metal compiles all .metal files together, so these are already visible)

// =============================================================================
// Particle Initialization Kernel
// =============================================================================
// Initializes particle state for a new batch. Each particle gets:
// - Default alive state and XS_LOOKUP event
// - Unit weight
// - Unique RNG key based on thread ID
// - Isotropic random direction
// - Position from source (uniform in a box for now)
//
// Future: source distribution buffer for fission bank sampling.
// =============================================================================

kernel void init_particles(
    device Particle* particles [[buffer(0)]],
    device const SimParams& params [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.numParticles) return;

    // Initialize RNG for this particle
    uint rng_counter = 0;
    uint rng_hi = tid;  // Use tid as high counter word for uniqueness
    uint rng_key = tid;

    // Sample isotropic direction
    float cosTheta = 2.0f * philox_uniform(rng_counter, rng_hi, rng_key) - 1.0f;
    float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
    float phi = 2.0f * M_PI_F * philox_uniform(rng_counter, rng_hi, rng_key);

    float3 dir = float3(
        sinTheta * cos(phi),
        sinTheta * sin(phi),
        cosTheta
    );

    // Initialize particle state
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

// =============================================================================
// Distance-to-Collision Kernel
// =============================================================================
// Samples the distance to the next collision using the inverse-CDF method:
//   d = -ln(xi) / sigma_total
// where xi is a uniform random number in (0,1).
//
// Requires XS lookup to have been performed (event == DISTANCE).
// Advances event state to MOVE.
// =============================================================================

kernel void distance_to_collision(
    device Particle* particles       [[buffer(0)]],
    device const SimParams& params   [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.numParticles) return;

    device Particle& p = particles[tid];
    if (p.alive == 0 || p.event != EVENT_DISTANCE) return;

    // Guard against zero or negative total XS
    if (p.xsTotal <= 0.0f) {
        p.alive = 0;
        p.event = EVENT_DEAD;
        return;
    }

    // Sample distance: d = -ln(xi) / sigma_t
    // philox_uniform requires a thread-local reference for counter_lo
    uint rng_counter = p.rngCounter;
    float xi = philox_uniform(rng_counter, tid, p.rngKey);
    p.rngCounter = rng_counter;

    p.distanceToCollision = -log(xi) / p.xsTotal;
    p.event = EVENT_MOVE;
}

// =============================================================================
// Move Particle + Boundary Crossing Kernel
// =============================================================================
// The most complex transport kernel. For each active particle in MOVE state:
//
// 1. Compute distance to the nearest cell boundary
// 2. Compare with the sampled distance to collision
// 3a. If collision is closer: move particle to collision site, set COLLIDE
// 3b. If boundary is closer:
//     - Move particle to boundary (plus BUMP)
//     - Check boundary condition:
//       * VACUUM:     kill particle
//       * REFLECTIVE: reflect direction off surface normal
//     - Find new cell for the particle's updated position
//     - If lost (outside all cells): kill particle, increment lost counter
//     - Otherwise: set XS_LOOKUP (need new XS in new material)
//
// Geometry functions (distance_to_boundary, find_cell, surface_normal) are
// provided by Geometry.metal in the same Metal library compilation unit.
// =============================================================================

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

    // Get distance to cell boundary
    float2 boundaryResult = distance_to_boundary(p.position, p.direction,
                                                  cells[p.cellIndex],
                                                  cellSurfaces, surfaces);
    float dBoundary = boundaryResult.x;
    int crossedSurfIdx = as_type<int>(boundaryResult.y);

    p.distanceToBoundary = dBoundary;
    p.boundarySurface = crossedSurfIdx;

    if (p.distanceToCollision < dBoundary) {
        // -----------------------------------------------------------
        // Collision in this cell
        // -----------------------------------------------------------
        float d = p.distanceToCollision;
        p.position += d * p.direction;
        p.distanceTraveled = d;
        p.event = EVENT_COLLIDE;
    } else {
        // -----------------------------------------------------------
        // Hit boundary -- move to surface + BUMP past it
        // -----------------------------------------------------------
        float d = dBoundary;
        p.position += (d + BUMP) * p.direction;
        p.distanceTraveled = d;

        // Check boundary condition of the crossed surface
        if (crossedSurfIdx >= 0 && uint(crossedSurfIdx) < params.numSurfaces) {
            uint bc = surfaces[crossedSurfIdx].boundaryCondition;

            if (bc == BC_VACUUM) {
                // Particle escapes the problem domain
                p.alive = 0;
                p.event = EVENT_DEAD;
                return;
            } else if (bc == BC_REFLECTIVE) {
                // Reflect direction off the surface normal
                float3 normal = surface_normal(p.position, surfaces[crossedSurfIdx]);
                p.direction = p.direction - 2.0f * dot(p.direction, normal) * normal;
                // Re-normalize to prevent floating-point drift
                p.direction = normalize(p.direction);
            }
        }

        // Find which cell the particle is now in
        int newCell = find_cell(p.position, cells, params.numCells,
                                cellSurfaces, surfaces);
        if (newCell < 0) {
            // Lost particle -- outside all defined cells
            p.alive = 0;
            p.event = EVENT_DEAD;
            atomic_fetch_add_explicit(lostParticleCount, 1, memory_order_relaxed);
            return;
        }
        p.cellIndex = uint(newCell);

        // Need new cross-section lookup in the (possibly different) material
        p.event = EVENT_XS_LOOKUP;
    }
}
