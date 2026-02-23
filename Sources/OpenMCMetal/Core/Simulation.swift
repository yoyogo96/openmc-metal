// Simulation.swift
// OpenMC-Metal
//
// Main simulation driver that orchestrates the event-based GPU transport loop
// for k-eigenvalue calculations. Ties together all kernels:
//
//   xs_lookup -> distance_to_collision -> move_particle -> collision -> tally_score
//       ^                                                                   |
//       |___________________________________________________________________|
//       (alive particles loop back; dead particles exit)
//
// Batch loop:
//   1. Inactive batches: converge fission source, discard tallies
//   2. Active batches:   accumulate tallies, track k-eff per batch
//   3. Post-processing:  compute statistics on k-eff and tallies

import Metal
import Foundation
import simd

// MARK: - SimulationResult

/// Aggregated results from a completed k-eigenvalue simulation.
struct SimulationResult {
    let kEff: (mean: Double, stdDev: Double, ci95: (Double, Double))
    let fluxStats: [(mean: Double, stdDev: Double, relError: Double)]
    let fissionStats: [(mean: Double, stdDev: Double, relError: Double)]
    let wallTime: Double
    let particlesPerSecond: Double
    let totalParticles: UInt64
    let numActiveBatches: Int
    let gpuName: String

    func printSummary(reference: Double? = nil) {
        print("\n" + String(repeating: "=", count: 60))
        print("SIMULATION RESULTS")
        print(String(repeating: "=", count: 60))

        TallyOutput.printKEffResult(stats: kEff, reference: reference,
                                     numBatches: numActiveBatches, wallTime: wallTime)

        print(String(format: "\nPerformance: %.2f particles/sec", particlesPerSecond))
        print(String(format: "Total particles: %llu", totalParticles))
        print("GPU: \(gpuName)")

        let numGroups = 7
        if !fluxStats.isEmpty {
            let numCells = fluxStats.count / numGroups
            TallyOutput.printFluxSummary(stats: fluxStats, numCells: numCells,
                                          numGroups: numGroups)
        }

        if !fissionStats.isEmpty {
            let numCells = fissionStats.count / numGroups
            TallyOutput.printFissionSummary(stats: fissionStats, numCells: numCells,
                                             numGroups: numGroups)
        }
    }
}

// MARK: - SimulationError

enum SimulationError: Error, CustomStringConvertible {
    case pipelineInitFailed(String)
    case tallyInitFailed(String)

    var description: String {
        switch self {
        case .pipelineInitFailed(let msg): return "Pipeline init failed: \(msg)"
        case .tallyInitFailed(let msg):    return "Tally init failed: \(msg)"
        }
    }
}

// MARK: - Simulation

/// k-eigenvalue Monte Carlo simulation using Apple Metal GPU kernels.
///
/// Runs the power-iteration loop:
///   inactive batches -> discard tallies -> active batches -> collect statistics
///
/// Usage:
/// ```swift
/// let sim = try Simulation(engine: engine, geometry: geometry,
///                          materialsBuffer: matBuf,
///                          numParticles: 100_000, numBatches: 100, numInactive: 50)
/// let source = C5G7Geometry.sourceSampler(numParticles: 100_000)
/// let result = sim.run(source: source)
/// result.printSummary(reference: C5G7Geometry.referenceKEff)
/// ```
final class Simulation {
    let engine: MetalEngine
    let geometry: GeometryData
    let materialsBuffer: MTLBuffer
    let numParticles: Int
    let numBatches: Int      // total batches (inactive + active)
    let numInactive: Int     // inactive batches for source convergence
    let numGroups: Int

    // GPU subsystems
    private let kernels: TransportKernels
    private let tally: TallyManager
    private let fissionBank: FissionBank
    private let collisionPipeline: MTLComputePipelineState

    // Persistent GPU buffers
    private let particleBuffer: MTLBuffer
    private let paramsBuffer: MTLBuffer
    private let lostCountBuffer: MTLBuffer

    // Current simulation parameters (mutable for k-eff updates)
    private var params: SimulationParams

    // Per-batch k-eff estimates (active batches only)
    private(set) var batchKEff: [Double] = []

    // Shannon entropy history for convergence diagnostics
    private(set) var shannonEntropy: [Double] = []

    // MARK: Init

    /// Initialize the simulation with geometry, materials, and run parameters.
    ///
    /// - Parameters:
    ///   - engine:       Metal engine for GPU access.
    ///   - geometry:     Packed geometry buffers (surfaces, cells, cell-surfaces).
    ///   - materialsBuffer: Flat float buffer of C5G7 cross-section data.
    ///   - numParticles: Number of particles per batch.
    ///   - numBatches:   Total number of batches (inactive + active).
    ///   - numInactive:  Number of inactive (source convergence) batches.
    ///   - numGroups:    Number of energy groups (default 7 for C5G7).
    init(engine: MetalEngine,
         geometry: GeometryData,
         materialsBuffer: MTLBuffer,
         numParticles: UInt32 = 100_000,
         numBatches: UInt32 = 100,
         numInactive: UInt32 = 50,
         numGroups: UInt32 = 7) throws {

        self.engine          = engine
        self.geometry        = geometry
        self.materialsBuffer = materialsBuffer
        self.numParticles    = Int(numParticles)
        self.numBatches      = Int(numBatches)
        self.numInactive     = Int(numInactive)
        self.numGroups       = Int(numGroups)

        // Transport kernels (xs_lookup, distance_to_collision, move_particle)
        self.kernels = try TransportKernels(engine: engine)

        // Collision kernel pipeline (dispatched manually due to fission bank bindings)
        self.collisionPipeline = try engine.makePipeline(functionName: "collision")

        // Tally manager
        self.tally = try TallyManager(engine: engine,
                                       numCells: Int(geometry.numCells),
                                       numGroups: Int(numGroups))

        // Fission bank -- allow up to 3x particles to avoid overflow
        self.fissionBank = FissionBank(engine: engine,
                                        maxSites: Int(numParticles) * 3)

        // Particle buffer
        self.particleBuffer = engine.makeBuffer(
            length: Int(numParticles) * MemoryLayout<Particle>.stride)
        self.particleBuffer.label = "SimulationParticles"

        // SimulationParams
        self.params = SimulationParams(
            numParticles: numParticles,
            numBatches:   numBatches,
            numInactive:  numInactive,
            numGroups:    numGroups,
            numCells:     geometry.numCells,
            numSurfaces:  geometry.numSurfaces,
            numMaterials: 7,
            kEff:         1.0
        )
        self.paramsBuffer = engine.makeBuffer(length: MemoryLayout<SimulationParams>.stride)
        self.paramsBuffer.label = "SimulationParams"

        // Lost-particle counter
        self.lostCountBuffer = engine.makeBuffer(length: MemoryLayout<UInt32>.stride)
        self.lostCountBuffer.label = "LostParticleCount"

        writeParams()
    }

    // MARK: Run

    /// Execute the full k-eigenvalue simulation.
    ///
    /// - Parameters:
    ///   - source:  Initial source particles as (position, group) pairs.
    ///   - verbose: Print progress for every batch (not just every 10th).
    /// - Returns: Aggregated simulation results with statistics.
    func run(source: [(position: SIMD3<Float>, group: UInt32)],
             verbose: Bool = false) -> SimulationResult {

        let wallStart = CFAbsoluteTimeGetCurrent()
        var totalParticlesTransported: UInt64 = 0

        // Current fission source for the next batch
        var currentSource = source

        let activeBatches = numBatches - numInactive

        print("Starting k-eigenvalue simulation")
        print("  Particles per batch: \(numParticles)")
        print("  Inactive batches:    \(numInactive)")
        print("  Active batches:      \(activeBatches)")
        print("  Energy groups:       \(numGroups)")
        print("  Cells:               \(params.numCells)")
        print()

        batchKEff = []
        batchKEff.reserveCapacity(activeBatches)
        shannonEntropy = []

        for batchIdx in 0..<numBatches {
            let isActive = batchIdx >= numInactive

            // Reset per-batch state
            fissionBank.reset()
            tally.resetForBatch()
            memset(lostCountBuffer.contents(), 0, MemoryLayout<UInt32>.stride)

            // Initialize particles from current source
            seedFromSource(currentSource, batchIndex: UInt32(batchIdx))

            // Update params buffer with current k-eff
            writeParams()

            // Run transport loop until all particles are dead
            let iterations = transportBatch()

            // Read back fission bank
            let fissionSites = fissionBank.sites()
            let fissionCount = fissionSites.count

            // Compute batch k-eff: k_new = k_old * (fission_sites / num_particles)
            let batchK: Double
            if fissionCount > 0 {
                batchK = Double(params.kEff) * Double(fissionCount) / Double(numParticles)
            } else {
                batchK = Double(params.kEff)
            }

            // Collect tallies for active batches
            if isActive {
                tally.collectBatch()
                batchKEff.append(batchK)
            }

            // Shannon entropy for convergence diagnostics
            if !fissionSites.isEmpty {
                let entropy = KEigenvalue.shannonEntropy(
                    sites: fissionSites,
                    meshBins: (10, 10, 1),
                    bounds: (min: SIMD3<Float>(0, 0, 0),
                             max: SIMD3<Float>(1.26, 1.26, 1.0))
                )
                shannonEntropy.append(entropy)
            }

            // Update k-eff for next batch
            params.kEff = Float(batchK)

            // Prepare source for next batch from fission bank
            currentSource = resampleFissionBank(fissionSites)

            totalParticlesTransported += UInt64(numParticles)

            // After inactive phase, clear tally history
            if batchIdx == numInactive - 1 {
                tally.clearBatchHistory()
            }

            // Print progress
            let lostCount = lostCountBuffer.contents()
                .assumingMemoryBound(to: UInt32.self).pointee
            if verbose || batchIdx % 10 == 0 || batchIdx == numBatches - 1 {
                let status = isActive ? "Active  " : "Inactive"
                let convergedStr = KEigenvalue.isConverged(entropyHistory: shannonEntropy)
                    ? " [converged]" : ""
                print(String(format: "Batch %3d/%d [%@] k-eff = %.5f  fission = %d  lost = %d  iter = %d%@",
                             batchIdx + 1, numBatches, status, batchK,
                             fissionCount, lostCount, iterations, convergedStr))
            }
        }

        let wallTime = CFAbsoluteTimeGetCurrent() - wallStart
        let particlesPerSecond = Double(totalParticlesTransported) / wallTime

        // Compute statistics over active batches
        let kEffStats = Statistics.kEffStatistics(batchKEff: batchKEff)
        let fluxStats = Statistics.computeStats(batchValues: tally.batchFlux)
        let fissionStats = Statistics.computeStats(batchValues: tally.batchFission)

        return SimulationResult(
            kEff: kEffStats,
            fluxStats: fluxStats,
            fissionStats: fissionStats,
            wallTime: wallTime,
            particlesPerSecond: particlesPerSecond,
            totalParticles: totalParticlesTransported,
            numActiveBatches: activeBatches,
            gpuName: engine.device.name
        )
    }

    // MARK: - Private: Transport Loop

    /// Run a single batch's event-based transport loop until all particles are dead.
    /// Returns the number of iterations executed.
    private func transportBatch() -> Int {
        let maxSteps = 1000  // safety limit

        for step in 0..<maxSteps {
            let cmdBuffer = engine.commandQueue.makeCommandBuffer()!
            cmdBuffer.label = "Transport_\(step)"

            // 1. XS Lookup (particles with event == XS_LOOKUP -> DISTANCE)
            kernels.dispatchXSLookup(
                particles: particleBuffer,
                materials: materialsBuffer,
                cells:     geometry.cellBuffer,
                params:    paramsBuffer,
                count:     numParticles,
                commandBuffer: cmdBuffer
            )

            // 2. Distance to Collision (event == DISTANCE -> MOVE)
            kernels.dispatchDistanceToCollision(
                particles: particleBuffer,
                params:    paramsBuffer,
                count:     numParticles,
                commandBuffer: cmdBuffer
            )

            // 3. Move Particle (event == MOVE -> COLLIDE or XS_LOOKUP or DEAD)
            kernels.dispatchMove(
                particles:    particleBuffer,
                surfaces:     geometry.surfaceBuffer,
                cells:        geometry.cellBuffer,
                cellSurfaces: geometry.cellSurfaceBuffer,
                params:       paramsBuffer,
                lostCount:    lostCountBuffer,
                count:        numParticles,
                commandBuffer: cmdBuffer
            )

            // 4. Collision (event == COLLIDE -> TALLY)
            //    Buffer bindings match Collision.metal exactly:
            //    [0] particles, [1] materials, [2] fissionBank, [3] fissionCount, [4] params
            dispatchCollision(commandBuffer: cmdBuffer)

            // 5. Tally Score (event == TALLY -> XS_LOOKUP or DEAD)
            tally.dispatch(
                particles:     particleBuffer,
                params:        paramsBuffer,
                count:         numParticles,
                commandBuffer: cmdBuffer
            )

            cmdBuffer.commit()
            cmdBuffer.waitUntilCompleted()

            // Check if all particles are dead
            if allParticlesDead() { return step + 1 }
        }

        return maxSteps
    }

    /// Encode collision kernel into commandBuffer.
    ///
    /// Buffer bindings match Collision.metal:
    ///   [0] particles    (read-write)
    ///   [1] materials    (read-only, flat float array)
    ///   [2] fissionBank  (read-write, FissionSite array)
    ///   [3] fissionCount (read-write, atomic_uint counter)
    ///   [4] params       (read-only, SimParams)
    private func dispatchCollision(commandBuffer: MTLCommandBuffer) {
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(collisionPipeline)
        encoder.setBuffer(particleBuffer,          offset: 0, index: 0)
        encoder.setBuffer(materialsBuffer,         offset: 0, index: 1)
        encoder.setBuffer(fissionBank.buffer,      offset: 0, index: 2)
        encoder.setBuffer(fissionBank.countBuffer, offset: 0, index: 3)
        encoder.setBuffer(paramsBuffer,            offset: 0, index: 4)

        let gridSize = numParticles
        let maxThreads = collisionPipeline.maxTotalThreadsPerThreadgroup
        let threadgroupWidth = min(maxThreads, gridSize)
        encoder.dispatchThreads(
            MTLSize(width: gridSize, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadgroupWidth, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    /// Returns true when all particles have alive == 0.
    private func allParticlesDead() -> Bool {
        let ptr = particleBuffer.contents().bindMemory(
            to: Particle.self, capacity: numParticles)
        for i in 0..<numParticles {
            if ptr[i].alive != 0 { return false }
        }
        return true
    }

    // MARK: - Private: Source Seeding

    /// Initialize particles from a source distribution.
    ///
    /// Each particle gets:
    /// - Position from source bank (cycled if numParticles > source.count)
    /// - Isotropic direction via Philox RNG (reproducible per particle+batch)
    /// - Energy group from source
    /// - Initial cell from geometry test (C5G7 pincell: inside/outside fuel cylinder)
    private func seedFromSource(_ source: [(position: SIMD3<Float>, group: UInt32)],
                                batchIndex: UInt32) {
        let ptr = particleBuffer.contents().bindMemory(
            to: Particle.self, capacity: numParticles)

        for i in 0..<numParticles {
            let srcIdx = i % source.count
            var rng = PhiloxRNG(key: UInt32(i), counterHi: batchIndex)

            let dir = rng.sampleIsotropicDirection()

            var p = Particle(
                position: source[srcIdx].position,
                direction: dir
            )
            p.energyGroup      = source[srcIdx].group
            p.weight           = 1.0
            p.alive            = 1
            p.event            = ParticleEvent.xsLookup.rawValue
            p.rngCounter       = rng.counterLo
            p.rngKey           = UInt32(i) ^ (batchIndex &* 0x9E3779B9)
            p.boundarySurface  = -1
            p.distanceTraveled = 0
            p.fissionFlag      = 0

            // Determine initial cell from position.
            // C5G7 pincell: cell 0 = fuel (inside cylinder r=0.54 at center 0.63,0.63)
            //               cell 1 = moderator (outside cylinder)
            let cx: Float = 0.63
            let cy: Float = 0.63
            let r: Float = 0.54
            let dx = p.position.x - cx
            let dy = p.position.y - cy
            p.cellIndex = (dx * dx + dy * dy < r * r) ? 0 : 1

            ptr[i] = p
        }
    }

    // MARK: - Private: Fission Bank Resampling

    /// Resample fission bank sites to create a new source for the next batch.
    /// Falls back to the default C5G7 source sampler if the bank is empty.
    private func resampleFissionBank(
        _ sites: [FissionSite]
    ) -> [(position: SIMD3<Float>, group: UInt32)] {
        guard !sites.isEmpty else {
            return C5G7Geometry.sourceSampler(
                numParticles: numParticles, numGroups: numGroups)
        }

        var source: [(position: SIMD3<Float>, group: UInt32)] = []
        source.reserveCapacity(numParticles)

        for i in 0..<numParticles {
            let idx = i % sites.count
            source.append((position: sites[idx].position,
                           group: sites[idx].energyGroup))
        }
        return source
    }

    // MARK: - Private: Parameter Update

    /// Write current params struct to the GPU buffer.
    private func writeParams() {
        let ptr = paramsBuffer.contents()
            .assumingMemoryBound(to: SimulationParams.self)
        ptr.pointee = params
    }
}
