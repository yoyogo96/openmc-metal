"""Statistical utilities for Monte Carlo tally post-processing."""

import math


def compute_stats(batch_values: list[list[float]]) -> list[tuple[float, float, float]]:
    """Compute (mean, std_dev_of_mean, relative_error) for each bin across batches.

    Uses Welford's online algorithm for numerical stability.

    Args:
        batch_values: List of batches, each batch is a list of bin values.
    Returns:
        List of (mean, stddev, rel_error) tuples, one per bin.
    """
    if not batch_values:
        return []
    n_bins = len(batch_values[0])
    n_batches = len(batch_values)
    results = []

    for i in range(n_bins):
        mean = 0.0
        m2 = 0.0
        for b in range(n_batches):
            x = batch_values[b][i]
            delta = x - mean
            mean += delta / (b + 1)
            delta2 = x - mean
            m2 += delta * delta2

        variance = m2 / (n_batches - 1) if n_batches > 1 else 0.0
        std_dev = math.sqrt(variance / n_batches)  # std dev of mean
        rel_error = std_dev / abs(mean) if mean != 0.0 else 0.0
        results.append((mean, std_dev, rel_error))

    return results


def keff_statistics(batch_keff: list[float]) -> tuple[float, float, tuple[float, float]]:
    """Compute mean, std-dev-of-mean, and 95% CI for batch k-eff values.

    Returns: (mean, std_dev, (ci95_lower, ci95_upper))
    """
    n = len(batch_keff)
    if n == 0:
        return (0.0, 0.0, (0.0, 0.0))

    mean = 0.0
    m2 = 0.0
    for i in range(n):
        delta = batch_keff[i] - mean
        mean += delta / (i + 1)
        delta2 = batch_keff[i] - mean
        m2 += delta * delta2

    variance = m2 / (n - 1) if n > 1 else 0.0
    std_dev = math.sqrt(variance / n)
    ci95 = (mean - 1.96 * std_dev, mean + 1.96 * std_dev)
    return (mean, std_dev, ci95)


def figure_of_merit(rel_error: float, wall_time: float) -> float:
    """FOM = 1 / (R^2 * T). Higher is better."""
    if rel_error <= 0.0 or wall_time <= 0.0:
        return 0.0
    return 1.0 / (rel_error * rel_error * wall_time)


def batch_means(batch_values: list[list[float]], size: int) -> list[float]:
    """Compute mean across batches for each bin."""
    if not batch_values or size <= 0:
        return [0.0] * size
    sums = [0.0] * size
    for batch in batch_values:
        for i in range(min(size, len(batch))):
            sums[i] += batch[i]
    n = len(batch_values)
    return [s / n for s in sums]
