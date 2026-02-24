"""Persistent history-based Metal shader for OpenMC-Metal.

Contains PERSISTENT_SHADER_SOURCE -- a single Metal Shading Language kernel
that runs one complete particle history per GPU thread, eliminating the
~600 kernel dispatches per batch of the event-based architecture.

Physics ported verbatim from:
  - Common.metal    (Philox RNG, atomic_add_float)
  - Geometry.metal  (surface eval, distance, cell finding, lattice fast-path)
  - XSLookup.metal  (cross-section lookup from flat material buffer)
  - Transport.metal (distance sampling, move, boundary handling)
  - Collision.metal (scatter/absorb/fission banking)
  - Tally.metal     (track-length flux and fission tallies)
"""

PERSISTENT_SHADER_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Constants
// =============================================================================

constant int   NUM_GROUPS          = 7;
constant float BUMP                = 1.0e-8f;
constant float INF_DIST            = 1.0e20f;
constant int   MAX_STEPS           = 5000;

// Material buffer layout: 77 floats per material
//   [0..6]   total[7]
//   [7..55]  scatter[7*7]  row-major: scatter[from*7 + to]
//   [56..62] fission[7]
//   [63..69] nuFission[7]
//   [70..76] chi[7]
constant uint FLOATS_PER_MATERIAL  = 77;
constant uint OFFSET_TOTAL         = 0;
constant uint OFFSET_SCATTER       = 7;
constant uint OFFSET_FISSION       = 56;
constant uint OFFSET_NU_FISSION    = 63;
constant uint OFFSET_CHI           = 70;

// Surface packed layout: 8 floats per surface
//   [0] type  (0=PLANE_X, 1=PLANE_Y, 2=PLANE_Z, 3=CYLINDER_Z, 4=SPHERE)
//   [1] boundary condition  (0=VACUUM, 1=REFLECTIVE, 2=TRANSMISSIVE)
//   [2] coeff_x
//   [3] coeff_y
//   [4] coeff_z
//   [5] coeff_w  (radius for cylinder/sphere)
//   [6] pad
//   [7] pad
constant uint SURF_STRIDE = 8;

// Cell packed layout: 4 uints per cell
//   [0] material_index
//   [1] num_surfaces
//   [2] surface_offset (into cell_surfs)
//   [3] pad
constant uint CELL_STRIDE = 4;

// Cell-surface packed layout: 2 ints per entry
//   [0] surface_index
//   [1] sense (+1 or -1)
constant uint CS_STRIDE = 2;

// Boundary condition codes
constant uint BC_VACUUM      = 0;
constant uint BC_REFLECTIVE  = 1;

// =============================================================================
// Philox-2x32-10 Counter-Based RNG
// =============================================================================
// Stateless CBRNG: output = f(counter, key).
// Each thread uses tid + batch_seed to form a unique key and increments
// counter per draw, guaranteeing independent reproducible streams.
// =============================================================================

constant uint PHILOX_M = 0xD2511F53u;
constant uint PHILOX_W = 0x9E3779B9u;

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
    return float(result.x) * 2.3283064365e-10f;  // 1.0 / 2^32
}

// =============================================================================
// Atomic Float Addition (CAS loop)
// =============================================================================

inline void atomic_add_float(device atomic_uint* target, float delta) {
    uint expected = atomic_load_explicit(target, memory_order_relaxed);
    while (true) {
        float current = as_type<float>(expected);
        float newVal  = current + delta;
        uint  desired = as_type<uint>(newVal);
        if (atomic_compare_exchange_weak_explicit(
                target, &expected, desired,
                memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
    }
}

// =============================================================================
// Surface Evaluation: sense at a point
// =============================================================================

inline int evaluate_surface_packed(float3 pos,
                                   device const float* surfaces,
                                   uint surfIdx) {
    uint base = surfIdx * SURF_STRIDE;
    uint stype = as_type<uint>(as_type<int>(surfaces[base + 0]));
    float cx = surfaces[base + 2];
    float cy = surfaces[base + 3];
    float cz = surfaces[base + 4];
    float cw = surfaces[base + 5];

    float val = 0.0f;
    switch (stype) {
        case 0: // PLANE_X
            val = pos.x - cx;
            break;
        case 1: // PLANE_Y
            val = pos.y - cy;
            break;
        case 2: // PLANE_Z
            val = pos.z - cz;
            break;
        case 3: { // CYLINDER_Z: (x-x0)^2 + (y-y0)^2 - R^2
            float dx = pos.x - cx;
            float dy = pos.y - cy;
            val = dx*dx + dy*dy - cw*cw;
            break;
        }
        case 4: { // SPHERE: (x-x0)^2 + (y-y0)^2 + (z-z0)^2 - R^2
            float dx = pos.x - cx;
            float dy = pos.y - cy;
            float dz = pos.z - cz;
            val = dx*dx + dy*dy + dz*dz - cw*cw;
            break;
        }
    }
    return (val >= 0.0f) ? 1 : -1;
}

// =============================================================================
// Distance to Surface along a ray
// =============================================================================

inline float distance_to_surface_packed(float3 pos, float3 dir,
                                        device const float* surfaces,
                                        uint surfIdx) {
    uint base = surfIdx * SURF_STRIDE;
    uint stype = as_type<uint>(as_type<int>(surfaces[base + 0]));
    float cx = surfaces[base + 2];
    float cy = surfaces[base + 3];
    float cz = surfaces[base + 4];
    float cw = surfaces[base + 5];

    switch (stype) {
        case 0: { // PLANE_X
            if (abs(dir.x) < 1.0e-10f) return INF_DIST;
            float d = (cx - pos.x) / dir.x;
            return (d > BUMP) ? d : INF_DIST;
        }
        case 1: { // PLANE_Y
            if (abs(dir.y) < 1.0e-10f) return INF_DIST;
            float d = (cy - pos.y) / dir.y;
            return (d > BUMP) ? d : INF_DIST;
        }
        case 2: { // PLANE_Z
            if (abs(dir.z) < 1.0e-10f) return INF_DIST;
            float d = (cz - pos.z) / dir.z;
            return (d > BUMP) ? d : INF_DIST;
        }
        case 3: { // CYLINDER_Z
            float dx = pos.x - cx;
            float dy = pos.y - cy;
            float R  = cw;

            float A = dir.x*dir.x + dir.y*dir.y;
            if (A < 1.0e-10f) return INF_DIST;

            float B    = 2.0f * (dx*dir.x + dy*dir.y);
            float C    = dx*dx + dy*dy - R*R;
            float disc = B*B - 4.0f*A*C;
            if (disc < 0.0f) return INF_DIST;

            float sq    = sqrt(disc);
            float inv2A = 1.0f / (2.0f * A);
            float d1    = (-B - sq) * inv2A;
            float d2    = (-B + sq) * inv2A;

            float result = INF_DIST;
            if (d1 > BUMP) result = d1;
            if (d2 > BUMP && d2 < result) result = d2;
            return result;
        }
        case 4: { // SPHERE
            float3 center = float3(cx, cy, cz);
            float R = cw;
            float3 oc = pos - center;

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

// =============================================================================
// Surface Normal (for reflective BC)
// =============================================================================

inline float3 surface_normal_packed(float3 pos,
                                    device const float* surfaces,
                                    uint surfIdx) {
    uint base = surfIdx * SURF_STRIDE;
    uint stype = as_type<uint>(as_type<int>(surfaces[base + 0]));
    float cx = surfaces[base + 2];
    float cy = surfaces[base + 3];
    float cz = surfaces[base + 4];

    switch (stype) {
        case 0: return float3(1.0f, 0.0f, 0.0f);
        case 1: return float3(0.0f, 1.0f, 0.0f);
        case 2: return float3(0.0f, 0.0f, 1.0f);
        case 3: { // CYLINDER_Z
            float2 n = float2(pos.x - cx, pos.y - cy);
            float len = length(n);
            return (len > 1.0e-12f) ? float3(n / len, 0.0f)
                                    : float3(1.0f, 0.0f, 0.0f);
        }
        case 4: { // SPHERE
            float3 n = pos - float3(cx, cy, cz);
            float len = length(n);
            return (len > 1.0e-12f) ? n / len : float3(0.0f, 0.0f, 1.0f);
        }
    }
    return float3(0.0f, 0.0f, 1.0f);
}

// =============================================================================
// Point-in-cell test (CSG half-space intersection)
// =============================================================================

inline bool point_in_cell_packed(float3 pos,
                                 device const uint*  cells,
                                 uint cellIdx,
                                 device const int*   cell_surfs,
                                 device const float* surfaces) {
    uint cbase    = cellIdx * CELL_STRIDE;
    uint nSurfs   = cells[cbase + 1];
    uint csOffset = cells[cbase + 2];

    for (uint i = 0; i < nSurfs; i++) {
        uint  csBase  = (csOffset + i) * CS_STRIDE;
        uint  surfIdx = as_type<uint>(cell_surfs[csBase + 0]);
        int   sense   = cell_surfs[csBase + 1];
        int   actual  = evaluate_surface_packed(pos, surfaces, surfIdx);
        if (actual != sense) return false;
    }
    return true;
}

// =============================================================================
// Cell Finder (with lattice fast-path for 17x17 assembly)
// =============================================================================

inline int find_cell_packed(float3 pos,
                            device const uint*  cells,
                            uint numCells,
                            device const int*   cell_surfs,
                            device const float* surfaces) {

    // --- Lattice fast-path for 17x17 assembly (578 = 17*17*2 cells) ---
    if (numCells == 578) {
        float lattice_pitch = 1.26f;
        uint  lattice_N     = 17;
        float halfPitch     = lattice_pitch * 0.5f;
        float fuel_R2       = 0.54f * 0.54f;

        int col = clamp(int(pos.x / lattice_pitch), 0, int(lattice_N) - 1);
        int row = clamp(int(pos.y / lattice_pitch), 0, int(lattice_N) - 1);

        float ccx = float(col) * lattice_pitch + halfPitch;
        float ccy = float(row) * lattice_pitch + halfPitch;
        float dx  = pos.x - ccx;
        float dy  = pos.y - ccy;

        uint pin_idx = uint(row) * lattice_N + uint(col);
        int candidate;
        if (dx * dx + dy * dy < fuel_R2) {
            candidate = int(pin_idx * 2);       // pin interior (fuel)
        } else {
            candidate = int(pin_idx * 2 + 1);   // moderator annulus
        }

        // Verify candidate
        if (point_in_cell_packed(pos, cells, uint(candidate), cell_surfs, surfaces)) {
            return candidate;
        }

        // Try the other half of the same pin
        int alt = (candidate & 1) ? candidate - 1 : candidate + 1;
        if (uint(alt) < numCells &&
            point_in_cell_packed(pos, cells, uint(alt), cell_surfs, surfaces)) {
            return alt;
        }

        // Try adjacent pins
        for (int dr = -1; dr <= 1; dr++) {
            for (int dc = -1; dc <= 1; dc++) {
                if (dr == 0 && dc == 0) continue;
                int nr = row + dr;
                int nc = col + dc;
                if (nr < 0 || nr >= int(lattice_N) ||
                    nc < 0 || nc >= int(lattice_N)) continue;
                uint adj_pin = uint(nr) * lattice_N + uint(nc);
                int c0 = int(adj_pin * 2);
                int c1 = c0 + 1;
                if (point_in_cell_packed(pos, cells, uint(c0), cell_surfs, surfaces))
                    return c0;
                if (point_in_cell_packed(pos, cells, uint(c1), cell_surfs, surfaces))
                    return c1;
            }
        }
        // Fall through to linear scan
    }

    // --- Linear scan ---
    for (uint i = 0; i < numCells; i++) {
        if (point_in_cell_packed(pos, cells, i, cell_surfs, surfaces))
            return int(i);
    }
    return -1;  // lost particle
}

// =============================================================================
// Distance to cell boundary
// =============================================================================
// Returns float2:
//   .x = minimum distance
//   .y = crossed surface index (bit-cast as float)

inline float2 dist_to_boundary_packed(float3 pos, float3 dir,
                                      device const uint*  cells,
                                      uint cellIdx,
                                      device const int*   cell_surfs,
                                      device const float* surfaces) {
    uint cbase    = cellIdx * CELL_STRIDE;
    uint nSurfs   = cells[cbase + 1];
    uint csOffset = cells[cbase + 2];

    float minDist     = INF_DIST;
    int   crossedSurf = -1;

    for (uint i = 0; i < nSurfs; i++) {
        uint csBase  = (csOffset + i) * CS_STRIDE;
        uint surfIdx = as_type<uint>(cell_surfs[csBase + 0]);

        float d = distance_to_surface_packed(pos, dir, surfaces, surfIdx);
        if (d < minDist) {
            minDist     = d;
            crossedSurf = int(surfIdx);
        }
    }

    return float2(minDist, as_type<float>(crossedSurf));
}

// =============================================================================
// Persistent History-Based Transport Kernel
// =============================================================================
// ONE thread = ONE complete particle history.
// No inter-kernel synchronization, no event state machine.
// Each thread runs: init -> [xs_lookup -> sample_d2c -> move/boundary ->
//   collision -> tally -> repeat] until absorbed/escaped/lost.
// =============================================================================

kernel void transport_kernel(
    // Particle SoA inputs (buffers 0-7) - read only
    device const float *px_in     [[ buffer(0) ]],
    device const float *py_in     [[ buffer(1) ]],
    device const float *pz_in     [[ buffer(2) ]],
    device const float *ux_in     [[ buffer(3) ]],
    device const float *uy_in     [[ buffer(4) ]],
    device const float *uz_in     [[ buffer(5) ]],
    device const int   *group_in  [[ buffer(6) ]],
    device const float *weight_in [[ buffer(7) ]],

    // Fission bank outputs (buffers 8-12)
    device float       *fiss_x    [[ buffer(8) ]],
    device float       *fiss_y    [[ buffer(9) ]],
    device float       *fiss_z    [[ buffer(10) ]],
    device int         *fiss_g    [[ buffer(11) ]],
    device atomic_uint *fiss_cnt  [[ buffer(12) ]],

    // Material data (buffers 13-19) - read only, flat layout
    device const float *sigma_t   [[ buffer(13) ]],   // total XS [mat*7+g]
    device const float *sigma_s   [[ buffer(14) ]],   // scatter matrix [mat*49+from*7+to]
    device const float *sigma_a   [[ buffer(15) ]],   // absorption XS [mat*7+g]
    device const float *sigma_f   [[ buffer(16) ]],   // fission XS [mat*7+g]
    device const float *nu_sf     [[ buffer(17) ]],   // nu*sigma_f [mat*7+g]
    device const float *chi       [[ buffer(18) ]],   // fission spectrum [mat*7+g]
    device const float *is_fiss   [[ buffer(19) ]],   // is fissile flag [mat]

    // Geometry data (buffers 20-22) - read only, packed
    device const float *surfaces  [[ buffer(20) ]],   // packed surface data
    device const uint  *cells     [[ buffer(21) ]],   // packed cell data
    device const int   *cell_surfs[[ buffer(22) ]],   // cell-surface list

    // Tally outputs (buffers 23-24)
    device atomic_uint *tally_flux     [[ buffer(23) ]],
    device atomic_uint *tally_fission  [[ buffer(24) ]],

    // Simulation parameters (buffer 25)
    device const float *params    [[ buffer(25) ]],

    uint tid [[ thread_position_in_grid ]]
) {
    // -------------------------------------------------------------------------
    // Unpack simulation parameters
    // -------------------------------------------------------------------------
    uint  num_particles   = as_type<uint>(as_type<int>(params[0]));
    uint  num_groups      = as_type<uint>(as_type<int>(params[1]));
    uint  num_cells       = as_type<uint>(as_type<int>(params[2]));
    uint  num_surfaces    = as_type<uint>(as_type<int>(params[3]));
    float k_eff           = params[4];
    uint  batch_seed      = as_type<uint>(as_type<int>(params[5]));
    uint  max_fission     = as_type<uint>(as_type<int>(params[6]));

    if (tid >= num_particles) return;

    // -------------------------------------------------------------------------
    // Load particle state from SoA arrays
    // -------------------------------------------------------------------------
    float3 pos = float3(px_in[tid], py_in[tid], pz_in[tid]);
    float3 dir = float3(ux_in[tid], uy_in[tid], uz_in[tid]);
    int    group  = group_in[tid];
    float  weight = weight_in[tid];

    // Ensure direction is normalized
    dir = normalize(dir);

    // -------------------------------------------------------------------------
    // Initialize RNG: unique stream per particle per batch
    // -------------------------------------------------------------------------
    uint rng_counter = 0;
    uint rng_hi      = tid;
    uint rng_key     = tid ^ batch_seed;

    // -------------------------------------------------------------------------
    // Find initial cell
    // -------------------------------------------------------------------------
    int cell_idx = find_cell_packed(pos, cells, num_cells, cell_surfs, surfaces);
    if (cell_idx < 0) return;  // particle outside geometry

    // -------------------------------------------------------------------------
    // Main transport loop
    // -------------------------------------------------------------------------
    for (int step = 0; step < MAX_STEPS; step++) {
        if (cell_idx < 0) break;

        // =====================================================================
        // 1. Cross-section lookup
        // =====================================================================
        uint mat = cells[uint(cell_idx) * CELL_STRIDE + 0];
        uint g   = uint(group);

        float sig_t = sigma_t[mat * NUM_GROUPS + g];
        float sig_f = sigma_f[mat * NUM_GROUPS + g];
        float nusf  = nu_sf[mat * NUM_GROUPS + g];
        float sig_a = sigma_a[mat * NUM_GROUPS + g];

        // Compute scattering XS as row sum of scatter matrix
        float sig_s_total = 0.0f;
        uint scatter_row_start = mat * (NUM_GROUPS * NUM_GROUPS) + g * NUM_GROUPS;
        for (uint gg = 0; gg < uint(NUM_GROUPS); gg++) {
            sig_s_total += sigma_s[scatter_row_start + gg];
        }

        // Guard against zero total XS
        if (sig_t <= 0.0f) break;

        // =====================================================================
        // 2. Sample distance to collision
        // =====================================================================
        float xi_d2c = max(philox_uniform(rng_counter, rng_hi, rng_key), 1.0e-30f);
        float d_collision = -log(xi_d2c) / sig_t;

        // =====================================================================
        // 3. Distance to cell boundary
        // =====================================================================
        float2 bnd_result = dist_to_boundary_packed(pos, dir,
                                                     cells, uint(cell_idx),
                                                     cell_surfs, surfaces);
        float d_boundary = bnd_result.x;
        int   crossed_surface = as_type<int>(bnd_result.y);

        // =====================================================================
        // 4. Determine outcome: collision or boundary crossing
        // =====================================================================
        if (d_collision < d_boundary) {
            // =================================================================
            // COLLISION
            // =================================================================
            float d = d_collision;
            pos += dir * d;

            // Track-length tally (score before collision changes group)
            if (uint(cell_idx) < num_cells) {
                uint tally_idx = uint(cell_idx) * num_groups + g;
                float flux_score    = weight * d;
                float fission_score = weight * d * sig_f;
                atomic_add_float(&tally_flux[tally_idx],    flux_score);
                atomic_add_float(&tally_fission[tally_idx], fission_score);
            }

            // Sample reaction type
            float xi_react = philox_uniform(rng_counter, rng_hi, rng_key);
            float absorption_prob = (sig_t > 0.0f) ? (sig_a / sig_t) : 0.0f;

            if (xi_react < absorption_prob) {
                // =============================================================
                // ABSORPTION
                // =============================================================
                // Check for fission
                if (sig_f > 0.0f && is_fiss[mat] > 0.5f) {
                    float fission_prob = (sig_a > 0.0f) ? (sig_f / sig_a) : 0.0f;
                    float xi_fiss = philox_uniform(rng_counter, rng_hi, rng_key);
                    if (xi_fiss < fission_prob) {
                        // Bank fission sites
                        float nu = (sig_f > 0.0f) ? (nusf / sig_f) : 0.0f;
                        float xi_nu = philox_uniform(rng_counter, rng_hi, rng_key);
                        uint num_sites = uint(nu * weight + xi_nu);

                        for (uint f = 0; f < min(num_sites, 3u); f++) {
                            uint fiss_idx = atomic_fetch_add_explicit(
                                fiss_cnt, 1u, memory_order_relaxed);

                            if (fiss_idx < max_fission) {
                                // Sample fission energy group from chi spectrum
                                float xi_chi  = philox_uniform(rng_counter, rng_hi, rng_key);
                                float cum_chi = 0.0f;
                                int   f_group = int(num_groups) - 1;

                                for (uint gg = 0; gg < num_groups; gg++) {
                                    cum_chi += chi[mat * NUM_GROUPS + gg];
                                    if (xi_chi < cum_chi) {
                                        f_group = int(gg);
                                        break;
                                    }
                                }

                                fiss_x[fiss_idx] = pos.x;
                                fiss_y[fiss_idx] = pos.y;
                                fiss_z[fiss_idx] = pos.z;
                                fiss_g[fiss_idx] = f_group;
                            }
                        }
                    }
                }
                break;  // particle absorbed -- history ends
            } else {
                // =============================================================
                // SCATTER
                // =============================================================
                // Sample outgoing energy group from scatter matrix row
                float xi_scatter = philox_uniform(rng_counter, rng_hi, rng_key);
                float threshold  = xi_scatter * ((sig_s_total > 0.0f) ? sig_s_total : 1.0f);
                float cum_s      = 0.0f;
                uint  new_group  = g;

                for (uint gg = 0; gg < uint(NUM_GROUPS); gg++) {
                    cum_s += sigma_s[scatter_row_start + gg];
                    if (cum_s >= threshold) {
                        new_group = gg;
                        break;
                    }
                }
                group = int(new_group);

                // Sample new isotropic direction
                float mu       = 2.0f * philox_uniform(rng_counter, rng_hi, rng_key) - 1.0f;
                float phi      = 2.0f * M_PI_F * philox_uniform(rng_counter, rng_hi, rng_key);
                float sinTheta = sqrt(max(0.0f, 1.0f - mu * mu));

                dir = normalize(float3(
                    sinTheta * cos(phi),
                    sinTheta * sin(phi),
                    mu
                ));
                // Continue to next step (new XS lookup with new group)
            }
        } else {
            // =================================================================
            // BOUNDARY CROSSING
            // =================================================================
            float d = d_boundary;

            // Track-length tally for distance traveled to boundary
            if (d > 0.0f && uint(cell_idx) < num_cells) {
                uint tally_idx = uint(cell_idx) * num_groups + g;
                float flux_score    = weight * d;
                float fission_score = weight * d * sig_f;
                atomic_add_float(&tally_flux[tally_idx],    flux_score);
                atomic_add_float(&tally_fission[tally_idx], fission_score);
            }

            // Handle near-zero distance boundary (stuck on surface)
            if (d < 1.0e-6f) {
                pos += 1.0e-5f * dir;
                cell_idx = find_cell_packed(pos, cells, num_cells,
                                            cell_surfs, surfaces);
                if (cell_idx < 0) break;  // lost
                continue;
            }

            // Check boundary condition
            if (crossed_surface >= 0 && uint(crossed_surface) < num_surfaces) {
                uint bc_base = uint(crossed_surface) * SURF_STRIDE;
                uint bc = as_type<uint>(as_type<int>(surfaces[bc_base + 1]));

                if (bc == BC_VACUUM) {
                    // Particle escapes
                    pos += (d + BUMP) * dir;
                    break;
                } else if (bc == BC_REFLECTIVE) {
                    // Move exactly to surface
                    pos += d * dir;
                    // Reflect direction off surface normal
                    float3 normal = surface_normal_packed(pos, surfaces,
                                                         uint(crossed_surface));
                    dir = dir - 2.0f * dot(dir, normal) * normal;
                    dir = normalize(dir);
                    // Bump in reflected direction (back into the cell)
                    pos += BUMP * dir;
                    // Stay in same cell, continue transport
                    continue;
                }
            }

            // Transmissive boundary: bump past surface, find new cell
            // Use larger bump (100x) to avoid getting stuck on curved surfaces
            pos += (d + 100.0f * BUMP) * dir;

            int new_cell = find_cell_packed(pos, cells, num_cells,
                                            cell_surfs, surfaces);
            if (new_cell < 0) break;  // lost particle

            cell_idx = new_cell;
            // Continue to next step (new XS lookup in possibly new material)
        }

        // =====================================================================
        // 5. Russian roulette for low-weight particles
        // =====================================================================
        if (weight < 0.25f) {
            if (philox_uniform(rng_counter, rng_hi, rng_key) < 0.5f) {
                weight *= 2.0f;
            } else {
                break;  // killed by Russian roulette
            }
        }
    }  // end transport loop
}
"""
