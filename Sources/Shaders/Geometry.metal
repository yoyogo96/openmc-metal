// Geometry.metal
// OpenMC-Metal: GPU geometry engine for Monte Carlo neutron transport.
//
// This file provides all surface evaluation, ray-surface intersection,
// CSG cell containment, and rectangular lattice traversal functions.
//
// Struct definitions (GPUSurface, GPUCell, GPUCellSurface, GPULattice, etc.)
// are provided by Common.metal which is compiled in the same Metal library.
// Do NOT redefine those structs here.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Small positive offset to avoid re-intersecting the surface a particle just
/// crossed (avoids zero-distance "self-hit" numerical artefacts).
constant float BUMP     = 1.0e-8f;

/// Sentinel distance meaning "no intersection found".
constant float INF_DIST = 1.0e20f;

// ---------------------------------------------------------------------------
// Surface sense evaluation
// ---------------------------------------------------------------------------

/// Evaluate the implicit surface equation at `pos` and return the algebraic sign.
/// Returns +1 if the point is on the positive (outside) side of the surface,
/// -1 if on the negative (inside) side.
/// A value of exactly zero is treated as +1 (positive side) for consistency.
///
/// Surface types:
///   0 – PLANE_X:    f = x - x0
///   1 – PLANE_Y:    f = y - y0
///   2 – PLANE_Z:    f = z - z0
///   3 – CYLINDER_Z: f = (x-x0)^2 + (y-y0)^2 - R^2
///   4 – SPHERE:     f = (x-x0)^2 + (y-y0)^2 + (z-z0)^2 - R^2
inline int evaluate_surface(float3 pos, device const GPUSurface& surface) {
    float val = 0.0f;

    switch (surface.type) {
        case 0: { // PLANE_X
            val = pos.x - surface.coefficients.x;
            break;
        }
        case 1: { // PLANE_Y
            val = pos.y - surface.coefficients.y;
            break;
        }
        case 2: { // PLANE_Z
            val = pos.z - surface.coefficients.z;
            break;
        }
        case 3: { // CYLINDER_Z: (x-x0)^2 + (y-y0)^2 - R^2
            float dx = pos.x - surface.coefficients.x;
            float dy = pos.y - surface.coefficients.y;
            float R  = surface.coefficients.z;
            val = dx*dx + dy*dy - R*R;
            break;
        }
        case 4: { // SPHERE: (x-x0)^2 + (y-y0)^2 + (z-z0)^2 - R^2
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

// ---------------------------------------------------------------------------
// Distance to surface along a ray
// ---------------------------------------------------------------------------

/// Compute the smallest positive distance from `pos` along direction `dir`
/// to the given surface.  Returns INF_DIST when:
///   - the ray is parallel (or nearly so) to the surface,
///   - all intersections are behind the ray origin (distance <= BUMP),
///   - the discriminant is negative (no real intersection).
///
/// BUMP guards against the particle re-hitting the surface it just crossed.
inline float distance_to_surface(float3 pos, float3 dir, device const GPUSurface& surface) {

    switch (surface.type) {

        case 0: { // PLANE_X: x = x0
            if (abs(dir.x) < 1.0e-10f) return INF_DIST;
            float d = (surface.coefficients.x - pos.x) / dir.x;
            return (d > BUMP) ? d : INF_DIST;
        }

        case 1: { // PLANE_Y: y = y0
            if (abs(dir.y) < 1.0e-10f) return INF_DIST;
            float d = (surface.coefficients.y - pos.y) / dir.y;
            return (d > BUMP) ? d : INF_DIST;
        }

        case 2: { // PLANE_Z: z = z0
            if (abs(dir.z) < 1.0e-10f) return INF_DIST;
            float d = (surface.coefficients.z - pos.z) / dir.z;
            return (d > BUMP) ? d : INF_DIST;
        }

        case 3: { // CYLINDER_Z: (x-x0)^2 + (y-y0)^2 = R^2
            float x0 = surface.coefficients.x;
            float y0 = surface.coefficients.y;
            float R  = surface.coefficients.z;

            float dx = pos.x - x0;
            float dy = pos.y - y0;

            // Quadratic coefficients for t: A*t^2 + B*t + C = 0
            float A = dir.x*dir.x + dir.y*dir.y;
            if (A < 1.0e-10f) return INF_DIST; // ray parallel to axis

            float B    = 2.0f * (dx*dir.x + dy*dir.y);
            float C    = dx*dx + dy*dy - R*R;
            float disc = B*B - 4.0f*A*C;

            if (disc < 0.0f) return INF_DIST;

            float sq = sqrt(disc);
            float inv2A = 1.0f / (2.0f * A);
            float d1 = (-B - sq) * inv2A;
            float d2 = (-B + sq) * inv2A;

            // Choose smallest positive root > BUMP
            float result = INF_DIST;
            if (d1 > BUMP) result = d1;
            if (d2 > BUMP && d2 < result) result = d2;
            return result;
        }

        case 4: { // SPHERE: |pos - center|^2 = R^2
            float3 center = float3(
                surface.coefficients.x,
                surface.coefficients.y,
                surface.coefficients.z);
            float R = surface.coefficients.w;

            float3 oc  = pos - center;
            // Quadratic: t^2 + B*t + C = 0  (A == 1 since dir is unit)
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

// ---------------------------------------------------------------------------
// CSG cell containment
// ---------------------------------------------------------------------------

/// Return true if `pos` satisfies all surface-sense conditions for `cell`.
/// A particle is inside the cell when its sense matches the required sense
/// for every bounding surface (CSG half-space intersection).
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

// ---------------------------------------------------------------------------
// Cell finder
// ---------------------------------------------------------------------------

/// Search all cells and return the index of the first cell that contains `pos`.
/// Returns -1 when the particle is lost (outside every defined cell).
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
    return -1; // lost particle
}

// ---------------------------------------------------------------------------
// Distance to cell boundary
// ---------------------------------------------------------------------------

/// Find the nearest surface crossing along the ray (pos, dir) for `cell`.
///
/// Returns a float2 where:
///   .x = minimum distance to the boundary (INF_DIST if none)
///   .y = bit-cast of the integer surface index crossed (-1 if none)
///
/// Using as_type<float>(int) to pack the integer into the float2 channel
/// avoids needing a custom return struct in MSL inline functions.
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

// ---------------------------------------------------------------------------
// Lattice helper functions
// ---------------------------------------------------------------------------

/// Map a world-space position to the integer (i, j, k) index of its lattice element.
/// Clamps to valid range so particles just outside the lattice are snapped to the edge.
inline uint3 lattice_element(float3 pos, device const GPULattice& lattice) {
    float3 local = (pos - lattice.origin) / lattice.pitch;
    int3   idx   = int3(floor(local));
    int3   dmax  = int3(lattice.dimensions) - 1;
    idx = clamp(idx, int3(0), dmax);
    return uint3(idx);
}

/// Transform a world-space position into the local coordinate frame of lattice
/// element `element` (i.e., relative to that element's lower-left-front corner).
inline float3 lattice_local_pos(float3 pos, uint3 element, device const GPULattice& lattice) {
    float3 element_origin = lattice.origin + float3(element) * lattice.pitch;
    return pos - element_origin;
}

/// Compute the distance from `pos` (inside element `element`) along `dir` to the
/// nearest face of that lattice element's bounding box.
/// Returns INF_DIST only if all direction components are effectively zero (degenerate ray).
inline float distance_to_lattice_boundary(float3               pos,
                                           float3               dir,
                                           uint3                element,
                                           device const GPULattice& lattice) {
    float3 elem_origin = lattice.origin + float3(element) * lattice.pitch;
    float3 elem_max    = elem_origin + lattice.pitch;

    float dmin = INF_DIST;

    // X slab
    if (abs(dir.x) > 1.0e-10f) {
        float d = (dir.x > 0.0f)
            ? (elem_max.x   - pos.x) / dir.x
            : (elem_origin.x - pos.x) / dir.x;
        if (d > BUMP && d < dmin) dmin = d;
    }

    // Y slab
    if (abs(dir.y) > 1.0e-10f) {
        float d = (dir.y > 0.0f)
            ? (elem_max.y   - pos.y) / dir.y
            : (elem_origin.y - pos.y) / dir.y;
        if (d > BUMP && d < dmin) dmin = d;
    }

    // Z slab
    if (abs(dir.z) > 1.0e-10f) {
        float d = (dir.z > 0.0f)
            ? (elem_max.z   - pos.z) / dir.z
            : (elem_origin.z - pos.z) / dir.z;
        if (d > BUMP && d < dmin) dmin = d;
    }

    return dmin;
}

// ---------------------------------------------------------------------------
// Lattice universe map lookup
// ---------------------------------------------------------------------------

/// Return the cell offset for the lattice element at integer index `element`.
/// universeMap is indexed as: k*(nx*ny) + j*nx + i
inline uint lattice_cell_offset(uint3                    element,
                                 device const GPULattice& lattice,
                                 device const uint*       universeMap) {
    uint nx    = lattice.dimensions.x;
    uint ny    = lattice.dimensions.y;
    uint flatI = element.z * (nx * ny) + element.y * nx + element.x;
    return universeMap[flatI];
}

// ---------------------------------------------------------------------------
// Surface normal (outward-pointing unit normal at a point on the surface)
// ---------------------------------------------------------------------------
// Useful for reflective boundary conditions or tallying.

/// Compute the outward-pointing unit normal to `surface` at `pos`.
/// `pos` is assumed to lie (approximately) on the surface.
inline float3 surface_normal(float3 pos, device const GPUSurface& surface) {
    switch (surface.type) {
        case 0: return float3(1.0f, 0.0f, 0.0f);   // PLANE_X
        case 1: return float3(0.0f, 1.0f, 0.0f);   // PLANE_Y
        case 2: return float3(0.0f, 0.0f, 1.0f);   // PLANE_Z
        case 3: { // CYLINDER_Z: gradient of (x-x0)^2+(y-y0)^2-R^2
            float2 n = float2(pos.x - surface.coefficients.x,
                              pos.y - surface.coefficients.y);
            float len = length(n);
            return (len > 1.0e-12f)
                ? float3(n / len, 0.0f)
                : float3(1.0f, 0.0f, 0.0f);
        }
        case 4: { // SPHERE: gradient of |pos-center|^2-R^2
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
