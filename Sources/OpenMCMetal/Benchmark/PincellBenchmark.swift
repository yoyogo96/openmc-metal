// PincellBenchmark.swift
// OpenMC-Metal
//
// C5G7 pincell k-eigenvalue benchmark.
//
// Ties together C5G7Geometry, C5G7Data cross sections, and the Simulation
// driver into a single runnable benchmark that reports k-eff and throughput,
// then compares against published GPU Monte Carlo codes.

import Metal
import Foundation

class PincellBenchmark {
    let engine: MetalEngine

    init(engine: MetalEngine) {
        self.engine = engine
    }

    func run(numParticles: UInt32 = 100_000,
             numBatches:   UInt32 = 50,
             numInactive:  UInt32 = 10,
             verbose:      Bool   = false) {

        print("╔══════════════════════════════════════════════════════╗")
        print("║   C5G7 Pincell Benchmark - Monte Carlo Transport     ║")
        print("╚══════════════════════════════════════════════════════╝")
        print()
        print(String(format: "Configuration:"))
        print(String(format: "  Particles/batch : %d", numParticles))
        print(String(format: "  Active batches  : %d", numBatches))
        print(String(format: "  Inactive batches: %d", numInactive))
        print()

        // Build geometry
        print("Building geometry...")
        let geometry = C5G7Geometry.buildPincell(engine: engine)
        print("  Cells: \(geometry.numCells), Surfaces: \(geometry.numSurfaces)")

        // Load cross sections
        print("Loading cross sections...")
        guard let materialsBuffer = loadC5G7Materials(device: engine.device) else {
            print("ERROR: Failed to load C5G7 cross-section data.")
            return
        }
        print("  Loaded 7 materials (C5G7 benchmark set)")

        // Generate initial source
        print("Generating source distribution...")
        let source = C5G7Geometry.sourceSampler(
            numParticles: Int(numParticles),
            numGroups: 7
        )
        print("  Generated \(source.count) source particles in UO2 fuel")
        print()

        // Create and run simulation
        do {
            let simulation = try Simulation(
                engine:          engine,
                geometry:        geometry,
                materialsBuffer: materialsBuffer,
                numParticles:    numParticles,
                numBatches:      numBatches,
                numInactive:     numInactive
            )

            print("Running simulation...")
            print()

            let result = simulation.run(source: source, verbose: verbose)

            print()

            // Print k-eff result summary
            result.printSummary(reference: C5G7Geometry.referenceKEff)

            // Build BenchmarkResult for comparison table
            let benchResult = BenchmarkResult(
                gpu:                 engine.device.name,
                benchmark:           "C5G7 Pincell (7-group)",
                particlesPerSecond:  result.particlesPerSecond,
                speedupVsCPU:        1.0,   // CPU comparison not yet implemented
                wallTimeSeconds:     result.wallTime,
                gpuTimeSeconds:      result.wallTime,
                numParticles:        Int(result.totalParticles),
                numBatches:          Int(numBatches),
                kEff:                result.kEff.mean,
                kEffStdDev:          result.kEff.stdDev
            )

            print()
            ReferenceDatabase.printComparisonTable(metalResult: benchResult)

        } catch {
            print("ERROR: Simulation failed: \(error)")
        }
    }
}
