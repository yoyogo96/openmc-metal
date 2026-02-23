"""CPU-only Monte Carlo transport baseline for GPU speedup comparison.

Implements the same C5G7 7-group pincell physics as the Metal GPU shaders,
in pure Python, so we can measure GPU-vs-CPU speedup directly.

Geometry:
  - Cylinder_Z at (0.63, 0.63) radius 0.54 (UO2 fuel pin)
  - Box [0, 1.26] x [0, 1.26] x [0, 1.0] (square moderator pitch)
  - All 6 outer planes reflective (infinite lattice approximation)
  - Cylinder surface transmissive (fuel/moderator interface)

Physics (same algorithm as GPU shaders):
  - Sample distance to collision: d = -log(xi) / sigma_t
  - Track to boundary or collision site
  - Reflective BC: reflect direction, stay in same cell, subtract boundary dist
  - Transmissive surface: move past it, find new cell, resample if material changed
  - Collision: absorption vs scatter based on sigma_a / sigma_t
  - Absorbed + fissile: bank nu*weight fission sites (for next generation source)
  - Scatter: sample outgoing group from scatter matrix row, isotropic direction
  - K-eigenvalue power iteration with configurable batches
"""

import math
import random
import time

from ..cross_sections import material_uo2, material_moderator
from ..geometry import source_sampler

# ---------------------------------------------------------------------------
# Geometry constants (C5G7 pincell)
# ---------------------------------------------------------------------------
_BOX_X_MIN = 0.0
_BOX_X_MAX = 1.26
_BOX_Y_MIN = 0.0
_BOX_Y_MAX = 1.26
_BOX_Z_MIN = 0.0
_BOX_Z_MAX = 1.0

_CYL_CX = 0.63
_CYL_CY = 0.63
_CYL_R  = 0.54
_CYL_R2 = _CYL_R * _CYL_R

# Small bump to avoid re-hitting the same surface
_BUMP = 1.0e-8

# NU for U-235 (average neutrons per fission)
# We use the nu_fission / fission ratio from the cross sections directly,
# but for fission site counting we use floor + stochastic rounding of nu.
_NU_DEFAULT = 2.43  # fallback if computed nu is zero


# ---------------------------------------------------------------------------
# Material data (pre-flattened for fast lookup)
# ---------------------------------------------------------------------------

def _build_material(mat_dict, num_groups=7):
    """Convert material dict to flat arrays for fast access."""
    g = num_groups
    return {
        "total":      mat_dict["total"][:g],
        "scatter":    mat_dict["scatter"][:g*g],   # row-major [from_g * g + to_g]
        "fission":    mat_dict["fission"][:g],
        "nu_fission": mat_dict["nu_fission"][:g],
        "chi":        mat_dict["chi"][:g],
        # Pre-compute CDF for scatter and chi
        "_chi_cdf":   _make_cdf(mat_dict["chi"][:g]),
    }


def _make_cdf(prob_list):
    """Convert probability list to CDF list."""
    total = sum(prob_list)
    if total == 0.0:
        return [1.0] * len(prob_list)
    cdf = []
    running = 0.0
    for p in prob_list:
        running += p / total
        cdf.append(running)
    cdf[-1] = 1.0  # ensure last bin catches rounding
    return cdf


def _sample_cdf(cdf):
    """Sample an index from a CDF list."""
    xi = random.random()
    for i, c in enumerate(cdf):
        if xi <= c:
            return i
    return len(cdf) - 1


def _scatter_row_cdf(mat, from_group, num_groups):
    """Return CDF for scatter from from_group to all outgoing groups."""
    g = num_groups
    row = mat["scatter"][from_group * g : from_group * g + g]
    return _make_cdf(row)


# ---------------------------------------------------------------------------
# Geometry: surface evaluations and distances
# ---------------------------------------------------------------------------

def _in_fuel(x, y):
    """True if (x,y) is inside the fuel cylinder."""
    dx = x - _CYL_CX
    dy = y - _CYL_CY
    return dx * dx + dy * dy < _CYL_R2


def _find_cell(x, y, z):
    """Return material index: 0=UO2 fuel, 6=moderator. -1=lost."""
    # Check bounding box
    if x < _BOX_X_MIN or x > _BOX_X_MAX:
        return -1
    if y < _BOX_Y_MIN or y > _BOX_Y_MAX:
        return -1
    if z < _BOX_Z_MIN or z > _BOX_Z_MAX:
        return -1
    if _in_fuel(x, y):
        return 0   # UO2
    return 6       # Moderator


def _dist_to_plane(pos, direction, plane_pos):
    """Distance from pos along direction to an axis-aligned plane.

    Returns positive distance if the plane is ahead, else infinity.
    """
    if abs(direction) < 1.0e-14:
        return math.inf
    d = (plane_pos - pos) / direction
    if d > 0.0:
        return d
    return math.inf


def _dist_to_cylinder_z(x, y, ux, uy):
    """Distance from (x,y) in direction (ux,uy) to the cylinder surface.

    Solves: ||(x + t*ux - cx, y + t*uy - cy)||^2 = r^2
    => a*t^2 + b*t + c = 0  where
       a = ux^2 + uy^2
       b = 2*((x-cx)*ux + (y-cy)*uy)
       c = (x-cx)^2 + (y-cy)^2 - r^2

    Returns the smallest positive root, or infinity if no positive real root.
    """
    dx = x - _CYL_CX
    dy = y - _CYL_CY
    a = ux * ux + uy * uy
    if a < 1.0e-14:
        return math.inf   # moving parallel to z axis, never hits cylinder
    b = 2.0 * (dx * ux + dy * uy)
    c = dx * dx + dy * dy - _CYL_R2
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return math.inf
    sqrt_disc = math.sqrt(discriminant)
    inv2a = 1.0 / (2.0 * a)
    t1 = (-b - sqrt_disc) * inv2a
    t2 = (-b + sqrt_disc) * inv2a
    # Return smallest positive root (slightly past zero to avoid self-intersect)
    min_positive = math.inf
    eps = 1.0e-9
    if t1 > eps:
        min_positive = t1
    if t2 > eps and t2 < min_positive:
        min_positive = t2
    return min_positive


def _cylinder_normal(x, y):
    """Outward unit normal to the cylinder at point (x, y)."""
    dx = x - _CYL_CX
    dy = y - _CYL_CY
    mag = math.sqrt(dx * dx + dy * dy)
    if mag < 1.0e-14:
        return (1.0, 0.0)
    return (dx / mag, dy / mag)


def _reflect_direction(ux, uy, uz, nx, ny, nz):
    """Reflect direction (ux,uy,uz) off surface with normal (nx,ny,nz)."""
    dot = ux * nx + uy * ny + uz * nz
    return (ux - 2.0 * dot * nx,
            uy - 2.0 * dot * ny,
            uz - 2.0 * dot * nz)


def _normalize(ux, uy, uz):
    mag = math.sqrt(ux * ux + uy * uy + uz * uz)
    if mag < 1.0e-14:
        return (0.0, 0.0, 1.0)
    return (ux / mag, uy / mag, uz / mag)


def _sample_isotropic():
    """Sample a uniformly distributed direction on the unit sphere."""
    cos_theta = 2.0 * random.random() - 1.0
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    phi = 2.0 * math.pi * random.random()
    return (sin_theta * math.cos(phi),
            sin_theta * math.sin(phi),
            cos_theta)


# ---------------------------------------------------------------------------
# Surface crossing: find boundary distance and which surface was hit
# ---------------------------------------------------------------------------

# Surface IDs (for reflective normal lookup)
_SURF_X_MIN = 0
_SURF_X_MAX = 1
_SURF_Y_MIN = 2
_SURF_Y_MAX = 3
_SURF_Z_MIN = 4
_SURF_Z_MAX = 5
_SURF_CYL   = 6

# Boundary condition tags
_BC_REFLECTIVE   = 1
_BC_TRANSMISSIVE = 2


def _distances_to_all_surfaces(x, y, z, ux, uy, uz):
    """Return list of (distance, surface_id, is_reflective) for all surfaces."""
    surfaces = []
    # Outer planes (all reflective)
    d = _dist_to_plane(x, ux, _BOX_X_MIN); surfaces.append((d, _SURF_X_MIN, True))
    d = _dist_to_plane(x, ux, _BOX_X_MAX); surfaces.append((d, _SURF_X_MAX, True))
    d = _dist_to_plane(y, uy, _BOX_Y_MIN); surfaces.append((d, _SURF_Y_MIN, True))
    d = _dist_to_plane(y, uy, _BOX_Y_MAX); surfaces.append((d, _SURF_Y_MAX, True))
    d = _dist_to_plane(z, uz, _BOX_Z_MIN); surfaces.append((d, _SURF_Z_MIN, True))
    d = _dist_to_plane(z, uz, _BOX_Z_MAX); surfaces.append((d, _SURF_Z_MAX, True))
    # Fuel cylinder (transmissive)
    d = _dist_to_cylinder_z(x, y, ux, uy); surfaces.append((d, _SURF_CYL, False))
    return surfaces


def _nearest_surface(surfaces):
    """Return (min_dist, surface_id, is_reflective) from surface list."""
    best = (math.inf, -1, False)
    for entry in surfaces:
        if entry[0] < best[0]:
            best = entry
    return best


def _surface_normal(surf_id, x, y, z, in_fuel_flag):
    """Return outward normal (nx, ny, nz) for the given surface."""
    if surf_id == _SURF_X_MIN:
        return (-1.0, 0.0, 0.0)
    elif surf_id == _SURF_X_MAX:
        return (1.0, 0.0, 0.0)
    elif surf_id == _SURF_Y_MIN:
        return (0.0, -1.0, 0.0)
    elif surf_id == _SURF_Y_MAX:
        return (0.0, 1.0, 0.0)
    elif surf_id == _SURF_Z_MIN:
        return (0.0, 0.0, -1.0)
    elif surf_id == _SURF_Z_MAX:
        return (0.0, 0.0, 1.0)
    elif surf_id == _SURF_CYL:
        nx, ny = _cylinder_normal(x, y)
        # Outward from cylinder center; flip if particle is inside fuel
        if in_fuel_flag:
            return (nx, ny, 0.0)
        else:
            return (-nx, -ny, 0.0)
    return (0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Single-particle transport
# ---------------------------------------------------------------------------

def _transport_particle(x, y, z, ux, uy, uz, group,
                        weight, mat_uo2, mat_mod, num_groups,
                        fission_bank):
    """Transport one particle to death and collect fission sites.

    Args:
        x, y, z: Initial position.
        ux, uy, uz: Initial direction (unit vector).
        group: Initial energy group index.
        weight: Statistical weight.
        mat_uo2, mat_mod: Pre-built material dicts.
        num_groups: Number of energy groups.
        fission_bank: List to append (x, y, z, weight) fission sites to.

    Returns:
        Number of collision events (for tally purposes).
    """
    g = num_groups
    alive = True
    num_collisions = 0

    # Current material
    mat_idx = _find_cell(x, y, z)
    if mat_idx < 0:
        return 0  # born outside geometry (shouldn't happen)

    mat = mat_uo2 if mat_idx == 0 else mat_mod

    # Sample distance to collision
    sigma_t = mat["total"][group]
    if sigma_t <= 0.0:
        sigma_t = 1.0e-10
    d_collision = -math.log(random.random()) / sigma_t

    while alive:
        # Current cell membership
        in_fuel = _in_fuel(x, y)

        # Find nearest surface
        surfs = _distances_to_all_surfaces(x, y, z, ux, uy, uz)
        d_boundary, surf_id, is_reflective = _nearest_surface(surfs)

        if d_boundary < 1.0e-14:
            # Degenerate: particle is right on a surface, nudge and retry
            x += _BUMP * ux
            y += _BUMP * uy
            z += _BUMP * uz
            mat_idx = _find_cell(x, y, z)
            if mat_idx < 0:
                break
            mat = mat_uo2 if mat_idx == 0 else mat_mod
            continue

        if d_collision <= d_boundary:
            # -------------------------------------------------------
            # Collision
            # -------------------------------------------------------
            x += d_collision * ux
            y += d_collision * uy
            z += d_collision * uz

            # Confirm still in geometry
            mat_idx = _find_cell(x, y, z)
            if mat_idx < 0:
                break

            mat = mat_uo2 if mat_idx == 0 else mat_mod
            num_collisions += 1

            sigma_t_g = mat["total"][group]
            if sigma_t_g <= 0.0:
                sigma_t_g = 1.0e-10
            sigma_f_g = mat["fission"][group]
            nu_sig_f  = mat["nu_fission"][group]
            # Absorption XS = total - scatter sum for this group
            scatter_row_sum = sum(mat["scatter"][group * g : group * g + g])
            sigma_a_g = sigma_t_g - scatter_row_sum
            if sigma_a_g < 0.0:
                sigma_a_g = 0.0

            p_absorb = sigma_a_g / sigma_t_g if sigma_t_g > 0 else 1.0

            if random.random() < p_absorb:
                # Absorbed — decide fission vs capture (matches GPU algorithm)
                # Note: C5G7 data has sigma_f > sigma_a for some groups because
                # sigma_a = sigma_t - scatter_row_sum can be smaller than sigma_f.
                # Cap fission probability at 1.0 (same as GPU shader).
                if sigma_a_g > 0.0 and sigma_f_g > 0.0:
                    fission_prob = min(1.0, sigma_f_g / sigma_a_g)
                    if random.random() < fission_prob:
                        # Fission: bank nu sites (stochastic rounding)
                        nu = nu_sig_f / sigma_f_g if sigma_f_g > 0.0 else _NU_DEFAULT
                        expected_sites = nu * weight
                        n_int = int(expected_sites)
                        if random.random() < (expected_sites - n_int):
                            n_int += 1
                        n_int = min(n_int, 3)  # cap per event (matches GPU)
                        for _ in range(n_int):
                            fission_bank.append((x, y, z, weight))
                alive = False
            else:
                # Scatter: sample outgoing group from scatter matrix row
                row_cdf = _scatter_row_cdf(mat, group, g)
                group = _sample_cdf(row_cdf)
                # Sample new isotropic direction
                ux, uy, uz = _sample_isotropic()
                # Sample new distance to collision
                sigma_t_new = mat["total"][group]
                if sigma_t_new <= 0.0:
                    sigma_t_new = 1.0e-10
                d_collision = -math.log(random.random()) / sigma_t_new

        else:
            # -------------------------------------------------------
            # Hit boundary
            # -------------------------------------------------------
            # Subtract boundary distance from remaining collision path
            d_collision -= d_boundary

            if is_reflective:
                # Move exactly to surface
                x += d_boundary * ux
                y += d_boundary * uy
                z += d_boundary * uz
                # Reflect direction off surface normal
                nx, ny, nz = _surface_normal(surf_id, x, y, z, in_fuel)
                ux, uy, uz = _reflect_direction(ux, uy, uz, nx, ny, nz)
                ux, uy, uz = _normalize(ux, uy, uz)
                # Bump in reflected direction (back into geometry)
                x += _BUMP * ux
                y += _BUMP * uy
                z += _BUMP * uz
                # Clamp to geometry bounds (for floating-point safety)
                x = max(_BOX_X_MIN, min(_BOX_X_MAX, x))
                y = max(_BOX_Y_MIN, min(_BOX_Y_MAX, y))
                z = max(_BOX_Z_MIN, min(_BOX_Z_MAX, z))
                # Material unchanged (stay in same cell, continue with d_collision)
                # No XS resample needed since we're in the same material
            else:
                # Transmissive (cylinder surface)
                old_mat = mat_idx
                # Move past surface with a small bump
                x += (d_boundary + 100.0 * _BUMP) * ux
                y += (d_boundary + 100.0 * _BUMP) * uy
                z += (d_boundary + 100.0 * _BUMP) * uz
                new_mat = _find_cell(x, y, z)
                if new_mat < 0:
                    alive = False
                    break
                mat_idx = new_mat
                mat = mat_uo2 if mat_idx == 0 else mat_mod
                if old_mat != mat_idx:
                    # Material changed: resample d_collision in new material
                    sigma_t_new = mat["total"][group]
                    if sigma_t_new <= 0.0:
                        sigma_t_new = 1.0e-10
                    d_collision = -math.log(random.random()) / sigma_t_new

    return num_collisions


# ---------------------------------------------------------------------------
# Power iteration (k-eigenvalue)
# ---------------------------------------------------------------------------

def run_cpu_baseline(num_particles: int = 1000,
                     num_batches: int = 20,
                     num_inactive: int = 5,
                     num_groups: int = 7) -> dict:
    """Run CPU-only MC transport and return performance metrics.

    Implements the same C5G7 7-group pincell k-eigenvalue calculation as
    the Metal GPU code, but in pure Python for direct speedup comparison.

    Args:
        num_particles: Particles per batch.
        num_batches:   Total number of power-iteration batches.
        num_inactive:  Inactive (source-convergence) batches (not tallied).
        num_groups:    Number of energy groups (must be 7 for C5G7).

    Returns:
        dict with keys:
            cpu_particles_per_sec  - particles tracked per wall-clock second
            cpu_keff               - estimated k-effective (mean over active batches)
            cpu_keff_std           - std deviation over active batches
            num_particles          - particles per batch
            num_batches            - total batches
            num_inactive           - inactive batches
            wall_time_sec          - total wall time in seconds
            batch_keff_history     - list of per-batch k-eff values
    """
    g = num_groups

    # Pre-build material data structures
    mat_uo2 = _build_material(material_uo2(), g)
    mat_mod = _build_material(material_moderator(), g)

    # Initial source: uniformly in fuel, chi-weighted groups
    source = source_sampler(num_particles, g)
    # source is list of ((x, y, z), group) tuples

    batch_keff_history = []
    active_keff = []
    total_tracked = 0

    t_start = time.perf_counter()

    for batch_idx in range(num_batches):
        fission_bank = []

        # Transport all source particles
        for (px, py, pz), p_group in source:
            ux, uy, uz = _sample_isotropic()
            _transport_particle(
                px, py, pz, ux, uy, uz, p_group,
                1.0, mat_uo2, mat_mod, g, fission_bank
            )
            total_tracked += 1

        # Estimate k-eff for this batch
        k_batch = len(fission_bank) / max(num_particles, 1)
        batch_keff_history.append(k_batch)

        if batch_idx >= num_inactive:
            active_keff.append(k_batch)

        # Build next generation source from fission bank
        if len(fission_bank) == 0:
            # No fissions: restart from sampler (shouldn't happen with good XS)
            source = source_sampler(num_particles, g)
        else:
            # Sample num_particles sites from fission bank (with replacement)
            # and assign chi-sampled energy groups
            chi_cdf = mat_uo2["_chi_cdf"]
            sampled = []
            for _ in range(num_particles):
                idx = random.randrange(len(fission_bank))
                fx, fy, fz, _fw = fission_bank[idx]
                fg = _sample_cdf(chi_cdf)
                sampled.append(((fx, fy, fz), fg))
            source = sampled

    t_end = time.perf_counter()
    wall_time = t_end - t_start

    # Compute keff statistics
    if active_keff:
        mean_keff = sum(active_keff) / len(active_keff)
        if len(active_keff) > 1:
            variance = sum((k - mean_keff) ** 2 for k in active_keff) / (len(active_keff) - 1)
            std_keff = math.sqrt(variance / len(active_keff))
        else:
            std_keff = 0.0
    else:
        # No active batches: use all
        mean_keff = sum(batch_keff_history) / max(len(batch_keff_history), 1)
        std_keff = 0.0

    particles_per_sec = total_tracked / wall_time if wall_time > 0 else 0.0

    return {
        "cpu_particles_per_sec": particles_per_sec,
        "cpu_keff":              mean_keff,
        "cpu_keff_std":          std_keff,
        "num_particles":         num_particles,
        "num_batches":           num_batches,
        "num_inactive":          num_inactive,
        "wall_time_sec":         wall_time,
        "batch_keff_history":    batch_keff_history,
    }
