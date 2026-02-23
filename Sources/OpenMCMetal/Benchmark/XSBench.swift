// XSBench.swift
// OpenMC-Metal
//
// XSBench GPU microbenchmark — measures raw cross-section lookup throughput.
//
// Inspired by the ANL XSBench proxy application (Tramm et al.):
//   "The XSBench Mini-app: Examining the Dominant Nuclear Data Lookup Operation
//    in Monte Carlo Particle Transport Simulations"
//
// This version exercises the flat C5G7 material buffer with 7-group data.
// Each "lookup" reads total XS + scatter matrix row for a random (material, group)
// pair, matching the work done by the xs_lookup transport kernel.
//
// Metal kernel: xsbench_lookup (defined in XSLookup.metal)

import Metal
import Foundation

// MARK: - Parameter struct (must match Metal counterpart byte-for-byte)

/// Parameters passed to the xsbench_lookup kernel.
/// 16 bytes — 4 x UInt32.
struct XSBenchParams {
    var numLookups:  UInt32
    var numMaterials: UInt32
    var numGroups:   UInt32
    var _pad:        UInt32 = 0
}

// MARK: - XSBenchmark

/// GPU cross-section lookup microbenchmark.
///
/// Usage:
/// ```swift
/// let bench = XSBenchmark(engine: engine)
/// bench.run()
/// ```
class XSBenchmark {
    let engine: MetalEngine

    init(engine: MetalEngine) {
        self.engine = engine
    }

    // -------------------------------------------------------------------------
    // MARK: Run
    // -------------------------------------------------------------------------

    /// Run the XSBench microbenchmark.
    ///
    /// - Parameters:
    ///   - numLookups:   Number of (material, group) lookups per timed run.
    ///   - numMaterials: Number of materials in the XS data (default: 7 for C5G7).
    ///   - numGroups:    Number of energy groups (default: 7 for C5G7).
    ///   - numRuns:      Number of timed iterations (excluding warmup).
    func run(numLookups:   Int = 10_000_000,
             numMaterials: Int = 7,
             numGroups:    Int = 7,
             numRuns:      Int = 5) {

        printHeader(numLookups: numLookups,
                    numMaterials: numMaterials,
                    numGroups: numGroups,
                    numRuns: numRuns)

        // ---- Cross-section data buffer ----
        guard let materialsBuffer = loadC5G7Materials(device: engine.device) else {
            print("ERROR: Failed to load C5G7 cross-section data into GPU buffer.")
            return
        }

        // ---- Random lookup index buffer ----
        // Layout: [matIdx0, groupIdx0, matIdx1, groupIdx1, ...]  (2 UInt32 per lookup)
        var lookupIndices = [UInt32](repeating: 0, count: numLookups * 2)
        for i in 0..<numLookups {
            lookupIndices[i * 2]     = UInt32.random(in: 0..<UInt32(numMaterials))
            lookupIndices[i * 2 + 1] = UInt32.random(in: 0..<UInt32(numGroups))
        }
        let lookupBuffer = engine.makeBuffer(lookupIndices)

        // ---- Output buffer (one Float per lookup) ----
        let outputBuffer = engine.makeBuffer(length: numLookups * MemoryLayout<Float>.stride)

        // ---- Parameter buffer ----
        let params = XSBenchParams(numLookups:   UInt32(numLookups),
                                   numMaterials: UInt32(numMaterials),
                                   numGroups:    UInt32(numGroups))
        let paramsBuffer = engine.makeBuffer([params])

        // ---- Pipeline ----
        let pipeline: MTLComputePipelineState
        do {
            pipeline = try engine.makePipeline(functionName: "xsbench_lookup")
        } catch {
            print("ERROR: Failed to create xsbench_lookup pipeline: \(error)")
            return
        }

        let buffers = [lookupBuffer, materialsBuffer, outputBuffer, paramsBuffer]

        // ---- Warmup ----
        print("Warming up...")
        engine.dispatchAndWait(pipeline: pipeline,
                               buffers: buffers,
                               gridSize: numLookups)
        print()

        // ---- Timed runs ----
        var times: [Double] = []
        times.reserveCapacity(numRuns)

        for run in 1...numRuns {
            let gpuTime = engine.dispatchAndWait(pipeline: pipeline,
                                                  buffers: buffers,
                                                  gridSize: numLookups)
            times.append(gpuTime)
            let throughput = Double(numLookups) / gpuTime / 1e6
            print(String(format: "  Run %d: %7.3f ms  (%6.2f M lookups/sec)",
                         run, gpuTime * 1000.0, throughput))
        }

        // ---- Statistics ----
        printResults(times: times, numLookups: numLookups, outputBuffer: outputBuffer)
    }

    // -------------------------------------------------------------------------
    // MARK: Private helpers
    // -------------------------------------------------------------------------

    private func printHeader(numLookups: Int, numMaterials: Int,
                             numGroups: Int, numRuns: Int) {
        print("╔══════════════════════════════════════════════╗")
        print("║       XSBench - Metal GPU Microbenchmark     ║")
        print("╚══════════════════════════════════════════════╝")
        print()
        print("Configuration:")
        print("  Lookups:    \(numLookups)")
        print("  Materials:  \(numMaterials)")
        print("  Groups:     \(numGroups)")
        print("  Runs:       \(numRuns)")
        print()
    }

    private func printResults(times: [Double], numLookups: Int, outputBuffer: MTLBuffer) {
        guard !times.isEmpty else { return }

        let n = Double(times.count)
        let mean = times.reduce(0.0, +) / n
        let variance: Double
        if times.count > 1 {
            variance = times.map { ($0 - mean) * ($0 - mean) }.reduce(0.0, +) / (n - 1.0)
        } else {
            variance = 0.0
        }
        let stdDev = variance.squareRoot()
        let throughput = Double(numLookups) / mean / 1e6

        // Spot-check output validity (first 100 entries should be > 0)
        let spotCheckCount = min(100, numLookups)
        let outPtr = outputBuffer.contents().bindMemory(to: Float.self, capacity: numLookups)
        var positiveCount = 0
        for i in 0..<spotCheckCount {
            if outPtr[i] > 0.0 { positiveCount += 1 }
        }

        print()
        print("Results:")
        print(String(format: "  Mean time:     %.3f +/- %.3f ms",
                     mean * 1000.0, stdDev * 1000.0))
        print(String(format: "  Throughput:    %.2f M lookups/sec", throughput))
        print(String(format: "  GPU:           %@", engine.device.name))
        print(String(format: "  Verification:  %d/%d spot checks positive",
                     positiveCount, spotCheckCount))
    }
}
