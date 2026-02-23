import Foundation

// MARK: - TallyStatistics (namespace alias)
typealias TallyStatistics = Statistics

// MARK: - Statistics

/// Statistical utilities for Monte Carlo tally post-processing.
///
/// Uses Welford's online algorithm for numerically stable mean and variance
/// accumulation, avoiding catastrophic cancellation for large batch counts.
enum Statistics {

    // MARK: - Core Welford Accumulation

    /// Compute (mean, population std-dev-of-mean, relative error) for each bin
    /// across all batches using Welford's online algorithm.
    ///
    /// - Parameters:
    ///   - batchValues: Array of batches; each batch is an array of `size` bin values.
    ///   - size: Number of tally bins (must equal `batchValues[i].count` for all i).
    /// - Returns: Array of `(mean, stdDev, relError)` tuples, one per bin.
    static func computeStats(
        batchValues: [[Double]]
    ) -> [(mean: Double, stdDev: Double, relError: Double)] {
        guard let first = batchValues.first else { return [] }
        let n         = first.count
        let numBatches = batchValues.count

        var results: [(mean: Double, stdDev: Double, relError: Double)] = []
        results.reserveCapacity(n)

        for i in 0..<n {
            var mean = 0.0
            var m2   = 0.0

            for b in 0..<numBatches {
                let x      = batchValues[b][i]
                let delta  = x - mean
                mean      += delta / Double(b + 1)
                let delta2 = x - mean
                m2        += delta * delta2
            }

            // Sample variance of the batch mean = s² / N
            let variance = numBatches > 1 ? m2 / Double(numBatches - 1) : 0.0
            let stdDev   = sqrt(variance / Double(numBatches))   // std dev of mean
            let relError = mean != 0.0 ? stdDev / abs(mean) : 0.0

            results.append((mean: mean, stdDev: stdDev, relError: relError))
        }

        return results
    }

    // MARK: - k-effective Statistics

    /// Compute mean, std-dev-of-mean, and 95 % confidence interval for
    /// a series of per-batch k-effective estimates using Welford's algorithm.
    ///
    /// - Parameter batchKEff: Per-batch k-eff values (active batches only).
    /// - Returns: `(mean, stdDev, ci95)` where `ci95 = (lower, upper)`.
    static func kEffStatistics(
        batchKEff: [Double]
    ) -> (mean: Double, stdDev: Double, ci95: (Double, Double)) {
        let n = batchKEff.count
        guard n > 0 else { return (0, 0, (0, 0)) }

        var mean = 0.0
        var m2   = 0.0

        for i in 0..<n {
            let delta  = batchKEff[i] - mean
            mean      += delta / Double(i + 1)
            let delta2 = batchKEff[i] - mean
            m2        += delta * delta2
        }

        let variance = n > 1 ? m2 / Double(n - 1) : 0.0
        let stdDev   = sqrt(variance / Double(n))   // std dev of mean
        let ci95     = (mean - 1.96 * stdDev, mean + 1.96 * stdDev)

        return (mean: mean, stdDev: stdDev, ci95: ci95)
    }

    // MARK: - Figure of Merit

    /// FOM = 1 / (R² * T), where R is relative error and T is wall time in seconds.
    /// Higher FOM indicates better computational efficiency.
    ///
    /// - Parameters:
    ///   - relError: Relative error of the tally of interest.
    ///   - wallTime: Total simulation wall time in seconds.
    /// - Returns: Figure of merit, or 0 if inputs are degenerate.
    static func figureOfMerit(relError: Double, wallTime: Double) -> Double {
        guard relError > 0.0, wallTime > 0.0 else { return 0.0 }
        return 1.0 / (relError * relError * wallTime)
    }

    // MARK: - Batch Mean Helper

    /// Compute the mean across batches for each bin.
    /// Lighter-weight than `computeStats` when only means are needed.
    ///
    /// - Parameters:
    ///   - batchValues: Array of batches; each batch is an array of bin values.
    ///   - size: Expected number of bins per batch.
    /// - Returns: Array of means, length `size`.
    static func batchMeans(batchValues: [[Double]], size: Int) -> [Double] {
        guard !batchValues.isEmpty, size > 0 else {
            return [Double](repeating: 0, count: size)
        }
        var sums = [Double](repeating: 0, count: size)
        for batch in batchValues {
            for i in 0..<min(size, batch.count) {
                sums[i] += batch[i]
            }
        }
        let n = Double(batchValues.count)
        return sums.map { $0 / n }
    }
}
