#include <metal_stdlib>
using namespace metal;

// Common.metal is compiled in the same Metal target; all shared types,
// constants, and RNG functions are already visible.

// =============================================================================
// OpenMC-Metal: Tally Kernel
// =============================================================================
// Scores EVENT_TALLY particles using track-length (path-length) estimators:
//
//   flux    += weight * distanceTraveled
//   fission += weight * distanceTraveled * xsFission
//
// Tally arrays are indexed as [cellIndex * numGroups + energyGroup].
// Atomic float addition is implemented via a compare-and-swap loop using
// uint bit-reinterpretation (compatible with all Metal feature sets).
//
// After scoring:
//   - Alive particles advance to EVENT_XS_LOOKUP (continue transport).
//   - Dead particles advance to EVENT_DEAD.
// =============================================================================

// atomic_add_float is now defined in Common.metal (shared with fused kernels).

// =============================================================================
// tally_score kernel
// =============================================================================
kernel void tally_score(
    device Particle*        particles     [[buffer(0)]],
    device atomic_uint*     tallyFlux     [[buffer(1)]],  // float bits via atomic_uint
    device atomic_uint*     tallyFission  [[buffer(2)]],  // float bits via atomic_uint
    device const SimParams& params        [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.numParticles) return;

    device Particle& p = particles[tid];
    if (p.event != EVENT_TALLY) return;

    // Guard against degenerate state.
    if (p.distanceTraveled <= 0.0f || p.cellIndex >= params.numCells) {
        // Advance state without scoring.
        p.event = (p.alive == 1u) ? EVENT_XS_LOOKUP : EVENT_DEAD;
        return;
    }

    // Tally bin: flat index into [numCells * numGroups] arrays.
    uint idx = p.cellIndex * params.numGroups + p.energyGroup;

    // Track-length flux estimator.
    float fluxScore    = p.weight * p.distanceTraveled;
    // Fission rate estimator.
    float fissionScore = p.weight * p.distanceTraveled * p.xsFission;

    atomic_add_float(&tallyFlux[idx],    fluxScore);
    atomic_add_float(&tallyFission[idx], fissionScore);

    // Advance particle event state.
    p.event = (p.alive == 1u) ? EVENT_XS_LOOKUP : EVENT_DEAD;
}
