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

// Helper: atomic add on a float stored as uint bits.
// Uses a CAS loop; correct even under high contention.
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
        // expected was updated by the intrinsic on failure; retry.
    }
}

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
