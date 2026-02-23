// XSLookup.metal
// OpenMC-Metal: Cross-section lookup kernel for multi-group neutron transport.
//
// Looks up macroscopic cross sections from a flat material data buffer
// for each active particle based on its cell's material and energy group.
//
// Struct definitions (Particle, GPUCell, SimParams, MaterialXS) and constants
// are provided by Common.metal which is compiled in the same Metal library.
// Do NOT redefine those here.

#include <metal_stdlib>
using namespace metal;

// Material buffer layout constants are defined in Common.metal.

// =============================================================================
// Cross-Section Lookup Kernel
// =============================================================================
// For each alive particle in the XS_LOOKUP event state:
// 1. Read the particle's cell -> material index
// 2. Read total, scatter, fission, nuFission from flat buffer
// 3. Compute scattering XS as the row sum of the scatter matrix
// 4. Derive absorption XS = total - scatter
// 5. Advance event state to DISTANCE
// =============================================================================

kernel void xs_lookup(device Particle* particles       [[buffer(0)]],
                      device const float* materials    [[buffer(1)]],
                      device const GPUCell* cells      [[buffer(2)]],
                      device const SimParams& params   [[buffer(3)]],
                      uint tid [[thread_position_in_grid]]) {
    if (tid >= params.numParticles) return;

    device Particle& p = particles[tid];
    if (p.alive == 0 || p.event != EVENT_XS_LOOKUP) return;

    // Get material from cell
    uint cellIdx = p.cellIndex;
    uint matIdx  = cells[cellIdx].materialIndex;
    uint group   = p.energyGroup;

    // Compute offset into flat material buffer
    uint matOffset = matIdx * FLOATS_PER_MATERIAL;

    // Read cross sections for this group
    p.xsTotal     = materials[matOffset + OFFSET_TOTAL + group];
    p.xsFission   = materials[matOffset + OFFSET_FISSION + group];
    p.xsNuFission = materials[matOffset + OFFSET_NU_FISSION + group];
    p.materialIndex = matIdx;

    // Compute scattering XS as sum of scatter matrix row for this group
    float scatterTotal = 0.0f;
    uint scatterRowStart = matOffset + OFFSET_SCATTER + group * params.numGroups;
    for (uint g = 0; g < params.numGroups; g++) {
        scatterTotal += materials[scatterRowStart + g];
    }
    p.xsScatter    = scatterTotal;
    p.xsAbsorption = p.xsTotal - scatterTotal;

    // Advance to distance sampling
    p.event = EVENT_DISTANCE;
}

// =============================================================================
// Fused XS Lookup + Distance-to-Collision Kernel
// =============================================================================
// Combines xs_lookup and distance_to_collision into a single kernel dispatch.
// Same thread does XS lookup then immediately samples collision distance.
// No barrier needed (same-thread data dependency).
// Eliminates one encoder + dispatch round-trip per transport step.
// =============================================================================

kernel void xs_lookup_and_distance(
    device Particle* particles       [[buffer(0)]],
    device const float* materials    [[buffer(1)]],
    device const GPUCell* cells      [[buffer(2)]],
    device const SimParams& params   [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.numParticles) return;

    device Particle& p = particles[tid];
    if (p.alive == 0 || p.event != EVENT_XS_LOOKUP) return;

    // --- Phase 1: XS Lookup (identical to xs_lookup) ---
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

    // --- Phase 2: Distance to Collision (identical to distance_to_collision) ---
    if (p.xsTotal <= 0.0f) {
        p.alive = 0;
        p.event = EVENT_DEAD;
        return;
    }

    uint rng_counter = p.rngCounter;
    float xi = max(philox_uniform(rng_counter, tid, p.rngKey), 1.0e-30f);
    p.rngCounter = rng_counter;

    p.distanceToCollision = -log(xi) / p.xsTotal;
    p.event = EVENT_MOVE;
}

// =============================================================================
// XSBench Microbenchmark Kernel
// =============================================================================
// Standalone cross-section lookup benchmark (not part of transport loop).
// Inspired by ANL XSBench proxy app.
//
// For each thread:
//   1. Read (matIdx, group) from the lookup index buffer.
//   2. Fetch total XS for that group from the flat material buffer.
//   3. Fetch and sum the scatter matrix row (simulates real lookup work).
//   4. Write total + scatterSum to output to prevent dead-code elimination.
//
// Buffer bindings:
//   0  lookups   — uint pairs [matIdx, groupIdx] per lookup
//   1  materials — flat C5G7 XS data (77 floats/material, same as xs_lookup)
//   2  output    — float result per lookup
//   3  params    — XSBenchParams
// =============================================================================

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

    // Base offset for this material (77 floats per material)
    uint matBase = matIdx * FLOATS_PER_MATERIAL;

    // Total cross section for this group
    float total = materials[matBase + OFFSET_TOTAL + group];

    // Sum scatter matrix row for this group (simulates real XS lookup work)
    float scatterSum = 0.0f;
    uint scatterRowBase = matBase + OFFSET_SCATTER + group * params.numGroups;
    for (uint g = 0; g < params.numGroups; g++) {
        scatterSum += materials[scatterRowBase + g];
    }

    // Write combined result — prevents compiler from optimising away the reads
    output[tid] = total + scatterSum;
}
