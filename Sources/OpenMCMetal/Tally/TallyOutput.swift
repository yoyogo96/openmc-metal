import Foundation

// MARK: - TallyOutput

/// Formatted output helpers for Monte Carlo tally results.
///
/// Produces human-readable reports for k-eigenvalue results,
/// flux summaries, and fission rate summaries.
enum TallyOutput {

    // MARK: - k-effective

    /// Print a formatted k-eigenvalue result block.
    ///
    /// - Parameters:
    ///   - stats:      Mean, std-dev of mean, and 95 % CI from `Statistics.kEffStatistics`.
    ///   - reference:  Optional reference k-eff for pass/fail comparison (100 pcm threshold).
    ///   - numBatches: Number of active batches used in the estimate.
    ///   - wallTime:   Total simulation wall time in seconds.
    static func printKEffResult(
        stats: (mean: Double, stdDev: Double, ci95: (Double, Double)),
        reference: Double?,
        numBatches: Int,
        wallTime: Double
    ) {
        print("\n=== K-Eigenvalue Results ===")
        print(String(format: "k-eff         = %.5f +/- %.5f", stats.mean, stats.stdDev))
        print(String(format: "95%% CI        = [%.5f, %.5f]", stats.ci95.0, stats.ci95.1))

        if let ref = reference {
            let deltaPcm = (stats.mean - ref) / ref * 1.0e5
            print(String(format: "Reference     = %.5f", ref))
            print(String(format: "Difference    = %+.1f pcm", deltaPcm))
            print(abs(deltaPcm) < 100.0 ? "Result        = PASS (|Δk| < 100 pcm)"
                                        : "Result        = FAIL (|Δk| >= 100 pcm)")
        }

        print(String(format: "Active batches= %d",   numBatches))
        print(String(format: "Wall time     = %.2f s", wallTime))
    }

    // MARK: - Flux Summary

    /// Print a tabular flux summary for the first `maxCells` cells and all groups.
    ///
    /// - Parameters:
    ///   - stats:     Per-bin statistics from `Statistics.computeStats`.
    ///   - numCells:  Total number of cells in the geometry.
    ///   - numGroups: Number of energy groups.
    ///   - maxCells:  Maximum number of cells to print (default 5).
    static func printFluxSummary(
        stats: [(mean: Double, stdDev: Double, relError: Double)],
        numCells: Int,
        numGroups: Int,
        maxCells: Int = 5
    ) {
        print("\n=== Flux Summary (first \(min(maxCells, numCells)) cells) ===")
        print("Cell  | Group | Flux          | Rel. Error")
        print("------+-------+---------------+-----------")

        for cell in 0..<min(maxCells, numCells) {
            for g in 0..<numGroups {
                let idx = cell * numGroups + g
                guard idx < stats.count else { continue }
                let s = stats[idx]
                print(String(format: "%5d | %5d | %13.6e | %9.4f%%",
                             cell, g, s.mean, s.relError * 100.0))
            }
        }
    }

    // MARK: - Fission Rate Summary

    /// Print a tabular fission-rate summary for the first `maxCells` cells.
    ///
    /// - Parameters:
    ///   - stats:     Per-bin statistics from `Statistics.computeStats`.
    ///   - numCells:  Total number of cells in the geometry.
    ///   - numGroups: Number of energy groups.
    ///   - maxCells:  Maximum number of cells to print (default 5).
    static func printFissionSummary(
        stats: [(mean: Double, stdDev: Double, relError: Double)],
        numCells: Int,
        numGroups: Int,
        maxCells: Int = 5
    ) {
        print("\n=== Fission Rate Summary (first \(min(maxCells, numCells)) cells) ===")
        print("Cell  | Group | Fission Rate  | Rel. Error")
        print("------+-------+---------------+-----------")

        for cell in 0..<min(maxCells, numCells) {
            for g in 0..<numGroups {
                let idx = cell * numGroups + g
                guard idx < stats.count else { continue }
                let s = stats[idx]
                print(String(format: "%5d | %5d | %13.6e | %9.4f%%",
                             cell, g, s.mean, s.relError * 100.0))
            }
        }
    }

    // MARK: - Figure of Merit

    /// Print a figure-of-merit summary.
    ///
    /// - Parameters:
    ///   - fom:      Figure of merit value from `Statistics.figureOfMerit`.
    ///   - binLabel: Human-readable label for the tally bin used.
    static func printFOM(fom: Double, binLabel: String = "max-rel-error bin") {
        print(String(format: "\nFigure of Merit (FOM) for %@: %.3e  [1/(R²·s)]",
                     binLabel, fom))
    }
}
