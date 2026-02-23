#include <metal_stdlib>
using namespace metal;

// Common.metal is compiled in the same Metal target; all shared types,
// constants, and RNG functions (philox_uniform, etc.) are already visible.

// =============================================================================
// OpenMC-Metal: Collision / Scattering Kernel
// =============================================================================
// Processes EVENT_COLLIDE particles:
//   - Samples absorption vs. scatter using pre-stored cross sections.
//   - On absorption: kills particle (event -> TALLY).
//     If fission cross section is non-zero, banks fission sites.
//   - On scatter: samples outgoing group from the scatter matrix,
//     samples a new isotropic direction, advances event -> TALLY.
//
// Material buffer layout (flat float array, FLOATS_PER_MATERIAL per material):
//   [0..6]   total[7]
//   [7..55]  scatter[7*7]   row-major: scatter[from*7 + to]
//   [56..62] fission[7]
//   [63..69] nuFission[7]
//   [70..76] chi[7]
//
// This matches MaterialXS in Common.metal / Types.swift exactly.
// =============================================================================

// Material buffer layout constants — must match MaterialXS struct offsets.
constant uint FLOATS_PER_MATERIAL = 77;  // 7 + 49 + 7 + 7 + 7
constant uint OFFSET_TOTAL        = 0;
constant uint OFFSET_SCATTER      = 7;   // after total[7]
constant uint OFFSET_FISSION      = 56;  // after total[7] + scatter[49]
constant uint OFFSET_NU_FISSION   = 63;  // after fission[7]
constant uint OFFSET_CHI          = 70;  // after nuFission[7]

// Fission site stored in the fission bank.
// 32 bytes — 8 x 4-byte fields (padded for alignment).
struct FissionSite {
    float3 position;    // 16 bytes (Metal float3 is padded to 16)
    float  _pad;
    uint   energyGroup; // sampled from chi spectrum
    uint   _pad2;
    uint   _pad3;
    uint   _pad4;
};

// =============================================================================
// collision kernel
// =============================================================================
kernel void collision(
    device Particle*        particles    [[buffer(0)]],
    device const float*     materials    [[buffer(1)]],   // flat MaterialXS array
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

    // xsAbsorption and xsTotal were stored by the XS lookup kernel.
    float absorptionProb = (p.xsTotal > 0.0f)
                           ? (p.xsAbsorption / p.xsTotal)
                           : 0.0f;

    if (xi < absorptionProb) {
        // ---------------------------------------------------------------------
        // ABSORPTION
        // ---------------------------------------------------------------------
        // Check for fission contribution.
        if (p.xsFission > 0.0f) {
            float fissionProb = (p.xsAbsorption > 0.0f)
                                ? (p.xsFission / p.xsAbsorption)
                                : 0.0f;
            float xi2 = philox_uniform(rng_counter, tid, p.rngKey);
            if (xi2 < fissionProb) {
                // Bank fission sites.
                // nu = nuFission / fission  (expected neutrons per fission)
                float nu  = (p.xsFission > 0.0f)
                            ? (p.xsNuFission / p.xsFission)
                            : 0.0f;
                float xi3 = philox_uniform(rng_counter, tid, p.rngKey);
                uint numFission = (uint)(nu * p.weight + xi3);

                for (uint f = 0; f < min(numFission, 3u); f++) {
                    uint idx = atomic_fetch_add_explicit(
                        fissionCount, 1u, memory_order_relaxed);

                    if (idx < params.numParticles * 2u) {
                        // Sample fission energy group from chi spectrum.
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

        // Kill particle; tally kernel will score before marking DEAD.
        p.alive = 0u;
        p.event = EVENT_TALLY;

    } else {
        // ---------------------------------------------------------------------
        // SCATTER
        // Sample outgoing energy group from scatter matrix row g.
        // Row g: materials[matOffset + OFFSET_SCATTER + g*numGroups + gg]
        // ---------------------------------------------------------------------
        float xiScatter  = philox_uniform(rng_counter, tid, p.rngKey);
        float cumScatter = 0.0f;
        uint  newGroup   = g;  // fallback: stay in same group

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

        // Continue transport: tally, then XS lookup next step.
        p.event = EVENT_TALLY;
    }

    // Write back RNG counter to device memory
    p.rngCounter = rng_counter;
}
