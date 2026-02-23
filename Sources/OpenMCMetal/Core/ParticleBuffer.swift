import Metal
import Foundation

// MARK: - ParticleBuffer

/// Manages GPU-side particle state and associated buffers.
/// Handles allocation, CPU-side initialization, and alive-count tracking.
class ParticleBuffer {
    let engine: MetalEngine
    let count: Int

    /// Main particle data buffer on GPU.
    let particleBuffer: MTLBuffer

    /// Atomic counter buffer for tracking alive particles.
    /// Single UInt32 value, updated atomically by kernels.
    let aliveCountBuffer: MTLBuffer

    /// Initialize particle buffer with the given capacity.
    init(engine: MetalEngine, count: Int) {
        self.engine = engine
        self.count = count

        let bufferSize = MemoryLayout<Particle>.stride * count
        self.particleBuffer = engine.makeBuffer(length: bufferSize)
        self.particleBuffer.label = "ParticleBuffer(\(count))"

        self.aliveCountBuffer = engine.makeBuffer(length: MemoryLayout<UInt32>.stride)
        self.aliveCountBuffer.label = "AliveCount"

        // Initialize alive count to particle count
        resetAliveCount()
    }

    /// Reset alive count to total particle count.
    func resetAliveCount() {
        let ptr = aliveCountBuffer.contents().assumingMemoryBound(to: UInt32.self)
        ptr.pointee = UInt32(count)
    }

    /// Read the current alive count from the buffer.
    func readAliveCount() -> UInt32 {
        let ptr = aliveCountBuffer.contents().assumingMemoryBound(to: UInt32.self)
        return ptr.pointee
    }

    /// Get a pointer to the particle array for CPU-side read/write.
    func particlePointer() -> UnsafeMutableBufferPointer<Particle> {
        let ptr = particleBuffer.contents().assumingMemoryBound(to: Particle.self)
        return UnsafeMutableBufferPointer(start: ptr, count: count)
    }

    // MARK: - CPU-Side Initialization

    /// Initialize particles on the CPU with a uniform source distribution.
    /// Each particle gets a unique RNG key based on its index and batch number.
    func initializeParticlesOnCPU(
        origin: SIMD3<Float> = .zero,
        extent: SIMD3<Float> = SIMD3<Float>(1, 1, 1),
        batchNumber: UInt32 = 0,
        kEff: Float = 1.0
    ) {
        let particles = particlePointer()

        for i in 0..<count {
            var rng = PhiloxRNG(key: UInt32(i), counterHi: batchNumber)

            // Sample uniform position within the source region
            let x = origin.x + rng.uniform() * extent.x
            let y = origin.y + rng.uniform() * extent.y
            let z = origin.z + rng.uniform() * extent.z

            // Sample isotropic direction
            let dir = rng.sampleIsotropicDirection()

            var p = Particle(
                position: SIMD3<Float>(x, y, z),
                direction: dir
            )
            p.energyGroup = 0
            p.weight = 1.0
            p.alive = 1
            p.event = ParticleEvent.xsLookup.rawValue
            p.cellIndex = 0
            p.rngCounter = rng.counterLo
            p.rngKey = UInt32(i)
            p.boundarySurface = -1
            p.distanceTraveled = 0
            p.fissionFlag = 0

            particles[i] = p
        }

        resetAliveCount()
    }

    /// Initialize particles from a fission bank (source sites from previous batch).
    func initializeFromFissionBank(
        sites: [(position: SIMD3<Float>, energyGroup: UInt32)],
        batchNumber: UInt32,
        kEff: Float
    ) {
        let particles = particlePointer()
        let numSites = sites.count

        for i in 0..<count {
            var rng = PhiloxRNG(key: UInt32(i), counterHi: batchNumber)

            // Sample from fission bank with replacement
            let siteIndex = Int(rng.next()) % max(numSites, 1)
            let site = numSites > 0 ? sites[siteIndex] : (position: SIMD3<Float>.zero, energyGroup: UInt32(0))

            let dir = rng.sampleIsotropicDirection()

            var p = Particle(
                position: site.position,
                direction: dir
            )
            p.energyGroup = site.energyGroup
            p.weight = 1.0
            p.alive = 1
            p.event = ParticleEvent.xsLookup.rawValue
            p.cellIndex = 0
            p.rngCounter = rng.counterLo
            p.rngKey = UInt32(i)
            p.boundarySurface = -1
            p.distanceTraveled = 0
            p.fissionFlag = 0

            particles[i] = p
        }

        resetAliveCount()
    }

    // MARK: - GPU-Side Initialization

    /// Dispatch the init_particles kernel on GPU.
    /// Requires the kernel to be compiled in the Metal library.
    func initializeParticlesOnGPU(params: SimulationParams) throws -> Double {
        let pipeline = try engine.makePipeline(functionName: "init_particles")
        let paramsBuffer = engine.makeBuffer(value: params)

        resetAliveCount()

        return engine.dispatchAndWait(
            pipeline: pipeline,
            buffers: [particleBuffer, paramsBuffer],
            gridSize: count
        )
    }

    // MARK: - Diagnostics

    /// Print summary statistics of current particle state.
    func printStats() {
        let particles = particlePointer()
        var alive = 0
        var eventCounts = [UInt32: Int]()

        for i in 0..<count {
            let p = particles[i]
            if p.alive != 0 { alive += 1 }
            eventCounts[p.event, default: 0] += 1
        }

        print("Particle Buffer Stats:")
        print("  Total:  \(count)")
        print("  Alive:  \(alive)")
        print("  Dead:   \(count - alive)")
        print("  Events:")
        for (event, cnt) in eventCounts.sorted(by: { $0.key < $1.key }) {
            let name = ParticleEvent(rawValue: event).map { "\($0)" } ?? "unknown(\(event))"
            print("    \(name): \(cnt)")
        }
    }
}
