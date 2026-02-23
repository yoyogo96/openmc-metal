// BenchmarkRunner.swift
// OpenMC-Metal
//
// Full benchmark suite runner.
// Executes all available benchmarks in sequence and reports aggregate timing.

import Metal
import Foundation

class BenchmarkRunner {
    let engine: MetalEngine

    init(engine: MetalEngine) {
        self.engine = engine
    }

    /// Run all benchmarks in sequence.
    func runAll() {
        let startTime = CFAbsoluteTimeGetCurrent()

        print("╔══════════════════════════════════════════════════════════════╗")
        print("║         OpenMC-Metal: Full Benchmark Suite                   ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print()
        printSystemInfo()
        print()

        // BENCHMARK 1: XSBench cross-section lookup
        print(String(repeating: "─", count: 60))
        print("BENCHMARK 1: XSBench Cross-Section Lookup")
        print(String(repeating: "─", count: 60))
        let xsbench = XSBenchmark(engine: engine)
        xsbench.run(numLookups: 10_000_000, numRuns: 5)
        print()

        // BENCHMARK 2: C5G7 Pincell k-eigenvalue
        print(String(repeating: "─", count: 60))
        print("BENCHMARK 2: C5G7 Pincell k-Eigenvalue")
        print(String(repeating: "─", count: 60))
        let pincell = PincellBenchmark(engine: engine)
        pincell.run(numParticles: 100_000, numBatches: 50, numInactive: 10)
        print()

        let totalTime = CFAbsoluteTimeGetCurrent() - startTime
        print(String(repeating: "═", count: 60))
        print(String(format: "Total benchmark time: %.1f seconds", totalTime))
        print(String(repeating: "═", count: 60))
    }

    /// Print system hardware information.
    func printSystemInfo() {
        print("System Information:")
        print("  GPU: \(engine.device.name)")
        print("  Unified Memory: \(engine.device.hasUnifiedMemory)")
        print("  Max Working Set: \(engine.device.recommendedMaxWorkingSetSize / 1024 / 1024) MB")
        print("  Max Threadgroup: \(engine.device.maxThreadsPerThreadgroup)")

        let osVersion = ProcessInfo.processInfo.operatingSystemVersion
        print("  macOS: \(osVersion.majorVersion).\(osVersion.minorVersion).\(osVersion.patchVersion)")
        print("  CPU Cores: \(ProcessInfo.processInfo.activeProcessorCount)")
        print("  RAM: \(ProcessInfo.processInfo.physicalMemory / 1024 / 1024 / 1024) GB")
    }
}
