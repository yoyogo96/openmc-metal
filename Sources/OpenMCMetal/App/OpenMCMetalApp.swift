// OpenMCMetalApp.swift
// OpenMC-Metal
//
// CLI entry point using swift-argument-parser.
//
// Subcommands:
//   openmc-metal info       — print GPU and system info
//   openmc-metal xsbench    — run XSBench cross-section lookup benchmark
//   openmc-metal c5g7       — run C5G7 pincell k-eigenvalue benchmark
//   openmc-metal benchmark  — run full benchmark suite

import ArgumentParser
import Metal
import Foundation

// MARK: - Root Command

@main
struct OpenMCMetal: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "openmc-metal",
        abstract: "Monte Carlo Neutron Transport on Apple Metal GPU",
        version: "0.1.0",
        subcommands: [
            Info.self,
            XSBenchCmd.self,
            C5G7Cmd.self,
            BenchmarkCmd.self
        ],
        defaultSubcommand: Info.self
    )
}

// MARK: - info

struct Info: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Print GPU and system information"
    )

    func run() throws {
        let engine = try MetalEngine()
        print()
        print("OpenMC-Metal v0.1.0")
        print("Monte Carlo Neutron Transport on Apple Metal GPU")
        print()
        engine.printDeviceInfo()

        // Struct layout verification
        print()
        print("=== Struct Layout Verification ===")
        print("  Particle:         \(MemoryLayout<Particle>.size) bytes  (stride: \(MemoryLayout<Particle>.stride), align: \(MemoryLayout<Particle>.alignment))")
        print("  GPUSurface:       \(MemoryLayout<GPUSurface>.size) bytes  (stride: \(MemoryLayout<GPUSurface>.stride), align: \(MemoryLayout<GPUSurface>.alignment))")
        print("  GPUCell:          \(MemoryLayout<GPUCell>.size) bytes  (stride: \(MemoryLayout<GPUCell>.stride), align: \(MemoryLayout<GPUCell>.alignment))")
        print("  GPUCellSurface:   \(MemoryLayout<GPUCellSurface>.size) bytes  (stride: \(MemoryLayout<GPUCellSurface>.stride), align: \(MemoryLayout<GPUCellSurface>.alignment))")
        print("  GPULattice:       \(MemoryLayout<GPULattice>.size) bytes  (stride: \(MemoryLayout<GPULattice>.stride), align: \(MemoryLayout<GPULattice>.alignment))")
        print("  MaterialXS:       \(MemoryLayout<MaterialXS>.size) bytes  (stride: \(MemoryLayout<MaterialXS>.stride), align: \(MemoryLayout<MaterialXS>.alignment))")
        print("  SimulationParams: \(MemoryLayout<SimulationParams>.size) bytes  (stride: \(MemoryLayout<SimulationParams>.stride), align: \(MemoryLayout<SimulationParams>.alignment))")
        print("===================================")
    }
}

// MARK: - xsbench

struct XSBenchCmd: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "xsbench",
        abstract: "Run XSBench cross-section lookup benchmark"
    )

    @Option(name: .long, help: "Number of (material, group) lookups per timed run")
    var lookups: Int = 10_000_000

    @Option(name: .long, help: "Number of timed runs (excluding warmup)")
    var runs: Int = 5

    func run() throws {
        let engine = try MetalEngine()
        let benchmark = XSBenchmark(engine: engine)
        benchmark.run(numLookups: lookups, numRuns: runs)
    }
}

// MARK: - c5g7

struct C5G7Cmd: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "c5g7",
        abstract: "Run C5G7 pincell k-eigenvalue benchmark"
    )

    @Option(name: .long, help: "Particles per batch")
    var particles: UInt32 = 100_000

    @Option(name: .long, help: "Number of active batches")
    var batches: UInt32 = 50

    @Option(name: .long, help: "Number of inactive (power iteration) batches")
    var inactive: UInt32 = 10

    @Flag(name: .long, help: "Print k-eff after every batch")
    var verbose: Bool = false

    func run() throws {
        let engine = try MetalEngine()
        let benchmark = PincellBenchmark(engine: engine)
        benchmark.run(numParticles: particles,
                      numBatches:   batches,
                      numInactive:  inactive,
                      verbose:      verbose)
    }
}

// MARK: - benchmark

struct BenchmarkCmd: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "benchmark",
        abstract: "Run full benchmark suite (XSBench + C5G7 pincell)"
    )

    func run() throws {
        let engine = try MetalEngine()
        let runner = BenchmarkRunner(engine: engine)
        runner.runAll()
    }
}
