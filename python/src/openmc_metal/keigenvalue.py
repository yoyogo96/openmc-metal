"""K-eigenvalue convergence diagnostics."""

import math


def shannon_entropy(sites: list[tuple[float, float, float]],
                    mesh_bins: tuple[int, int, int] = (10, 10, 1),
                    bounds_min: tuple[float, float, float] = (0.0, 0.0, 0.0),
                    bounds_max: tuple[float, float, float] = (1.26, 1.26, 1.0)) -> float:
    """Compute Shannon entropy of fission source distribution.

    Args:
        sites: List of (x, y, z) fission site positions.
        mesh_bins: Number of bins in each direction.
        bounds_min: Minimum bounds of the mesh.
        bounds_max: Maximum bounds of the mesh.
    Returns:
        Shannon entropy value.
    """
    if not sites:
        return 0.0

    nx, ny, nz = mesh_bins
    total_bins = nx * ny * nz
    counts = [0] * total_bins

    dx = (bounds_max[0] - bounds_min[0]) / nx
    dy = (bounds_max[1] - bounds_min[1]) / ny
    dz = (bounds_max[2] - bounds_min[2]) / nz

    for x, y, z in sites:
        ix = min(int((x - bounds_min[0]) / dx), nx - 1)
        iy = min(int((y - bounds_min[1]) / dy), ny - 1)
        iz = min(int((z - bounds_min[2]) / dz), nz - 1)
        ix = max(0, ix)
        iy = max(0, iy)
        iz = max(0, iz)
        idx = iz * nx * ny + iy * nx + ix
        counts[idx] += 1

    total = sum(counts)
    if total == 0:
        return 0.0

    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)

    return entropy


def is_converged(entropy_history: list[float], window: int = 10, tolerance: float = 0.1) -> bool:
    """Check if the fission source has converged based on Shannon entropy stability."""
    if len(entropy_history) < window:
        return False
    recent = entropy_history[-window:]
    mean_val = sum(recent) / len(recent)
    if mean_val == 0.0:
        return False
    max_dev = max(abs(v - mean_val) / mean_val for v in recent)
    return max_dev < tolerance
