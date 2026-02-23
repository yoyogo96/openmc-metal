// KEigenvalue.swift
// OpenMC-Metal
//
// Utilities for k-eigenvalue convergence diagnostics.
// Provides Shannon entropy of the fission source and a convergence check
// based on entropy stabilisation over a sliding window.

import simd
import Foundation

// MARK: - KEigenvalue

/// Static helpers for k-eigenvalue convergence diagnostics.
enum KEigenvalue {

    // MARK: Shannon Entropy

    /// Compute the spatial Shannon entropy of the fission source.
    ///
    /// The geometry is overlaid with a regular `meshBins` mesh spanning `bounds`.
    /// Each fission site is assigned to a bin; the entropy is:
    ///
    ///   H = -sum_i ( p_i * log2(p_i) )
    ///
    /// where p_i = (sites in bin i) / (total sites).
    ///
    /// - Parameters:
    ///   - sites:     Fission sites from the current batch.
    ///   - meshBins:  Number of bins in x, y, z directions.
    ///   - bounds:    Spatial extent of the mesh (min and max corners).
    /// - Returns: Shannon entropy in bits. Returns 0 when `sites` is empty.
    static func shannonEntropy(
        sites: [FissionSite],
        meshBins: (Int, Int, Int),
        bounds: (min: SIMD3<Float>, max: SIMD3<Float>)
    ) -> Double {
        guard !sites.isEmpty else { return 0.0 }

        let (nx, ny, nz) = meshBins
        let totalBins = nx * ny * nz
        guard totalBins > 0 else { return 0.0 }

        let rangeX = bounds.max.x - bounds.min.x
        let rangeY = bounds.max.y - bounds.min.y
        let rangeZ = bounds.max.z - bounds.min.z

        // Bin counts
        var counts = [Int](repeating: 0, count: totalBins)

        for site in sites {
            let pos = site.position

            // Normalised position [0, 1) clamped to valid range
            let fx = rangeX > 0 ? Double((pos.x - bounds.min.x) / rangeX) : 0.0
            let fy = rangeY > 0 ? Double((pos.y - bounds.min.y) / rangeY) : 0.0
            let fz = rangeZ > 0 ? Double((pos.z - bounds.min.z) / rangeZ) : 0.0

            let ix = min(Int(fx * Double(nx)), nx - 1)
            let iy = min(Int(fy * Double(ny)), ny - 1)
            let iz = min(Int(fz * Double(nz)), nz - 1)

            // Clamp negative indices that could arise from sites outside bounds
            guard ix >= 0, iy >= 0, iz >= 0 else { continue }

            let binIdx = iz * ny * nx + iy * nx + ix
            counts[binIdx] += 1
        }

        let total = Double(sites.count)
        var entropy = 0.0

        for count in counts {
            guard count > 0 else { continue }
            let p = Double(count) / total
            entropy -= p * log2(p)
        }

        return entropy
    }

    // MARK: Convergence Check

    /// Determine whether the fission source has converged by checking if the
    /// Shannon entropy has stabilised over the last `window` batches.
    ///
    /// Convergence is declared when the relative standard deviation of the
    /// last `window` entropy values is below `threshold`.
    ///
    /// - Parameters:
    ///   - entropyHistory: Chronological history of per-batch Shannon entropy values.
    ///   - window:         Number of recent batches to examine (default 10).
    ///   - threshold:      Maximum allowed relative std-dev for convergence (default 0.1).
    /// - Returns: `true` when the source is considered converged.
    static func isConverged(
        entropyHistory: [Double],
        window: Int = 10,
        threshold: Double = 0.1
    ) -> Bool {
        guard entropyHistory.count >= window else { return false }

        let recent = Array(entropyHistory.suffix(window))

        let mean = recent.reduce(0.0, +) / Double(window)
        guard mean > 0.0 else { return false }

        let variance = recent.reduce(0.0) { acc, v in
            let d = v - mean
            return acc + d * d
        } / Double(window)

        let relStdDev = sqrt(variance) / mean
        return relStdDev < threshold
    }
}
