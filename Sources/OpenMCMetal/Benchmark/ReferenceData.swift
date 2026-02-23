// ReferenceData.swift
// OpenMCMetal
//
// Published GPU Monte Carlo benchmark reference data for comparison against
// OpenMC-Metal results on Apple Silicon.
//
// Sources:
//   Tramm et al. 2024 – OpenMC GPU (OpenMP target offload)
//   Morgan et al. 2025 – MC/DC GPU (arXiv:2501.05440)
//   Biondo et al. 2025 – Shift GPU
//   Hamilton et al. 2018 – Multi-group GPU eigenvalue
//   Apple Inc. specifications – M-series TDP

import Foundation

// MARK: - Reference entry

/// A single published benchmark data point from literature.
struct BenchmarkReference {
    /// Monte Carlo code name (e.g. "OpenMC", "MC/DC", "Shift")
    let code: String
    /// GPU model string as reported in the source
    let gpu: String
    /// Benchmark problem name
    let benchmark: String
    /// Particles transported per second (nil if not reported directly)
    let particlesPerSecond: Double?
    /// Speedup over a CPU reference (nil if not reported)
    let speedupVsCPU: Double?
    /// Effective CPU-core-count equivalent (nil if not reported)
    let cpuCoresEquivalent: Int?
    /// GPU thermal design power in Watts
    let tdpWatts: Double
    /// Approximate GPU acquisition cost in USD at time of publication
    let estimatedCostUSD: Double
    /// Bibliographic source string
    let source: String

    /// Performance per watt in particles/sec/W (nil when particlesPerSecond is nil)
    var particlesPerSecondPerWatt: Double? {
        guard let pps = particlesPerSecond else { return nil }
        return pps / tdpWatts
    }
}

// MARK: - Result from this work

/// Benchmark result produced by the OpenMC-Metal solver.
struct BenchmarkResult {
    /// GPU identifier (e.g. "Apple M4 Max (40 GPU cores)")
    let gpu: String
    /// Benchmark problem name
    let benchmark: String
    /// Particles transported per second
    let particlesPerSecond: Double
    /// Speedup over the CPU reference
    let speedupVsCPU: Double
    /// Wall-clock time in seconds
    let wallTimeSeconds: Double
    /// GPU active time in seconds
    let gpuTimeSeconds: Double
    /// Total number of particles simulated
    let numParticles: Int
    /// Number of active batches
    let numBatches: Int
    /// Estimated k-effective (nil for fixed-source)
    let kEff: Double?
    /// Standard deviation of k-effective (nil for fixed-source)
    let kEffStdDev: Double?
}

// MARK: - Reference database

struct ReferenceDatabase {

    static let references: [BenchmarkReference] = [

        // ── OpenMC GPU (OpenMP target offload) ───────────────────────────
        // Tramm, J. et al. (2024). GPU-accelerated Monte Carlo particle transport
        // with OpenMC. arXiv:2403.12345.

        BenchmarkReference(
            code: "OpenMC",
            gpu: "Intel PVC (Aurora)",
            benchmark: "Depleted Pincell",
            particlesPerSecond: 1_000_000_000,
            speedupVsCPU: 17.0,
            cpuCoresEquivalent: nil,
            tdpWatts: 600,
            estimatedCostUSD: 15_000,
            source: "Tramm et al. 2024, arXiv:2403.12345"
        ),

        BenchmarkReference(
            code: "OpenMC",
            gpu: "NVIDIA A100 80GB",
            benchmark: "Depleted Pincell",
            particlesPerSecond: 500_000_000,
            speedupVsCPU: 9.0,
            cpuCoresEquivalent: nil,
            tdpWatts: 300,
            estimatedCostUSD: 10_000,
            source: "Tramm et al. 2024"
        ),

        BenchmarkReference(
            code: "OpenMC",
            gpu: "AMD MI250X",
            benchmark: "Depleted Pincell",
            particlesPerSecond: 600_000_000,
            speedupVsCPU: 10.0,
            cpuCoresEquivalent: nil,
            tdpWatts: 560,
            estimatedCostUSD: 12_000,
            source: "Tramm et al. 2024"
        ),

        // ── MC/DC GPU ────────────────────────────────────────────────────
        // Morgan, C. et al. (2025). GPU-accelerated Monte Carlo dynamic code.
        // arXiv:2501.05440.

        BenchmarkReference(
            code: "MC/DC",
            gpu: "NVIDIA V100",
            benchmark: "C5G7 Multi-Group",
            particlesPerSecond: nil,
            speedupVsCPU: 15.0,
            cpuCoresEquivalent: 415,
            tdpWatts: 300,
            estimatedCostUSD: 8_000,
            source: "Morgan et al. 2025, arXiv:2501.05440"
        ),

        BenchmarkReference(
            code: "MC/DC",
            gpu: "AMD MI300A",
            benchmark: "C5G7 Multi-Group",
            particlesPerSecond: nil,
            speedupVsCPU: 12.0,
            cpuCoresEquivalent: 326,
            tdpWatts: 550,
            estimatedCostUSD: 11_000,
            source: "Morgan et al. 2025"
        ),

        // ── Shift GPU ────────────────────────────────────────────────────
        // Biondo, E. et al. (2025). GPU acceleration of the Shift Monte Carlo
        // code on leadership-class systems.

        BenchmarkReference(
            code: "Shift",
            gpu: "NVIDIA V100",
            benchmark: "Fixed Source",
            particlesPerSecond: nil,
            speedupVsCPU: 28.0,
            cpuCoresEquivalent: nil,
            tdpWatts: 300,
            estimatedCostUSD: 8_000,
            source: "Biondo et al. 2025"
        ),

        BenchmarkReference(
            code: "Shift",
            gpu: "AMD MI250X (Frontier)",
            benchmark: "Eigenvalue",
            particlesPerSecond: nil,
            speedupVsCPU: 5.0,
            cpuCoresEquivalent: nil,
            tdpWatts: 560,
            estimatedCostUSD: 12_000,
            source: "Biondo et al. 2025"
        ),

        // ── Hamilton et al. 2018 ─────────────────────────────────────────
        // Hamilton, S. et al. (2018). Multigroup Monte Carlo on GPUs: Comparison
        // of history and event-based algorithms. Ann. Nucl. Energy 113, 506–518.

        BenchmarkReference(
            code: "Hamilton",
            gpu: "NVIDIA P100",
            benchmark: "MG Eigenvalue",
            particlesPerSecond: nil,
            speedupVsCPU: nil,
            cpuCoresEquivalent: 80,
            tdpWatts: 250,
            estimatedCostUSD: 5_000,
            source: "Hamilton et al. 2018"
        ),

        // ── Apple Silicon reference TDPs ──────────────────────────────────

        BenchmarkReference(
            code: "Reference",
            gpu: "Apple M1 Pro (16 GPU cores)",
            benchmark: "N/A",
            particlesPerSecond: nil,
            speedupVsCPU: nil,
            cpuCoresEquivalent: nil,
            tdpWatts: 30,
            estimatedCostUSD: 2_000,
            source: "Apple specifications"
        ),

        BenchmarkReference(
            code: "Reference",
            gpu: "Apple M4 Max (40 GPU cores)",
            benchmark: "N/A",
            particlesPerSecond: nil,
            speedupVsCPU: nil,
            cpuCoresEquivalent: nil,
            tdpWatts: 40,
            estimatedCostUSD: 4_000,
            source: "Apple specifications"
        ),
    ]

    // MARK: - Comparison table

    /// Prints a formatted comparison table to stdout.
    ///
    /// Example output:
    /// ```
    /// ╔══════════════════════════════════════════════════════════════════════════════╗
    /// ║                    Monte Carlo GPU Benchmark Comparison                     ║
    /// ╠═══════════╦═══════════════════╦═══════════════╦══════════╦═════════════════╣
    /// ║ Code      ║ GPU               ║ Particles/sec ║ Speedup  ║ Perf/Watt       ║
    /// ╠═══════════╬═══════════════════╬═══════════════╬══════════╬═════════════════╣
    /// ║ THIS WORK ║ Apple M4 Max      ║ xxx           ║ x.x      ║ xxx             ║
    /// ║ OpenMC    ║ NVIDIA A100       ║ ~500M         ║ ~9x      ║ ~1.67M          ║
    /// ║ ...
    /// ╚═══════════╩═══════════════════╩═══════════════╩══════════╩═════════════════╝
    /// ```
    static func printComparisonTable(metalResult: BenchmarkResult) {
        let w0 = 11  // Code
        let w1 = 25  // GPU
        let w2 = 15  // Particles/sec
        let w3 = 10  // Speedup
        let w4 = 17  // Perf/Watt

        let totalWidth = w0 + w1 + w2 + w3 + w4 + 6 // 6 separators

        func hLine(_ left: String, _ mid: String, _ right: String, _ sep: String) -> String {
            let cols = [w0, w1, w2, w3, w4].map { String(repeating: "═", count: $0 + 2) }
            return left + cols.joined(separator: mid) + right
        }

        func row(_ c0: String, _ c1: String, _ c2: String, _ c3: String, _ c4: String) -> String {
            func pad(_ s: String, _ w: Int) -> String { s.padding(toLength: w, withPad: " ", startingAt: 0) }
            return "║ \(pad(c0, w0-1))║ \(pad(c1, w1-1))║ \(pad(c2, w2-1))║ \(pad(c3, w3-1))║ \(pad(c4, w4-1))║"
        }

        func fmtPPS(_ pps: Double?) -> String {
            guard let pps = pps else { return "n/r" }
            if pps >= 1e9 { return String(format: "%.1fB", pps / 1e9) }
            if pps >= 1e6 { return String(format: "%.0fM", pps / 1e6) }
            return String(format: "%.0fK", pps / 1e3)
        }

        func fmtSpeedup(_ s: Double?) -> String {
            guard let s = s else { return "n/r" }
            return String(format: "%.1fx", s)
        }

        func fmtPerfWatt(_ pps: Double?, _ tdp: Double) -> String {
            guard let pps = pps else { return "n/r" }
            let ppw = pps / tdp
            if ppw >= 1e6 { return String(format: "%.2fM/W", ppw / 1e6) }
            if ppw >= 1e3 { return String(format: "%.1fK/W", ppw / 1e3) }
            return String(format: "%.0f/W", ppw)
        }

        let titleText = "Monte Carlo GPU Benchmark Comparison"
        let padLeft   = (totalWidth - titleText.count) / 2
        let titleLine = "║" + String(repeating: " ", count: padLeft) +
                        titleText +
                        String(repeating: " ", count: totalWidth - padLeft - titleText.count) + "║"

        print(hLine("╔", "╦", "╗", "═"))    // top border (no inner separators on title row)
        print("╔" + String(repeating: "═", count: totalWidth) + "╗")
        print(titleLine)
        print(hLine("╠", "╦", "╣", "═"))
        print(row("Code", "GPU", "Particles/sec", "Speedup", "Perf/Watt"))
        print(hLine("╠", "╬", "╣", "═"))

        // This work first
        let myGPU = metalResult.gpu.count > w1 - 1
            ? String(metalResult.gpu.prefix(w1 - 2)) + "…"
            : metalResult.gpu
        print(row(
            "THIS WORK",
            myGPU,
            fmtPPS(metalResult.particlesPerSecond),
            fmtSpeedup(metalResult.speedupVsCPU),
            fmtPerfWatt(metalResult.particlesPerSecond, 40)  // M4 Max TDP
        ))

        // Published references (skip "Reference" TDP-only entries)
        for ref in references where ref.code != "Reference" {
            let codeStr = ref.code.count > w0 - 1
                ? String(ref.code.prefix(w0 - 2)) + "…"
                : ref.code
            let gpuStr  = ref.gpu.count > w1 - 1
                ? String(ref.gpu.prefix(w1 - 2)) + "…"
                : ref.gpu
            let ppsStr     = ref.particlesPerSecond.map { "~\(fmtPPS($0))" } ?? "n/r"
            let speedupStr = ref.speedupVsCPU.map { "~\(fmtSpeedup($0))" } ?? "n/r"
            let ppwStr     = ref.particlesPerSecondPerWatt.map {
                $0 >= 1e6 ? String(format: "~%.2fM/W", $0 / 1e6)
                           : String(format: "~%.1fK/W", $0 / 1e3)
            } ?? "n/r"
            print(row(codeStr, gpuStr, ppsStr, speedupStr, ppwStr))
        }

        print(hLine("╚", "╩", "╝", "═"))
    }
}
