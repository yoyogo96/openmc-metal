import Metal
import simd

// MARK: - FissionSite

/// A single fission site stored in the fission bank.
/// Memory layout must match the Metal `FissionSite` struct in Collision.metal byte-for-byte:
///   float3 position (16 bytes — Metal float3 padded to 16), uint energyGroup, 3x uint pad.
/// Total: 32 bytes (8 x 4-byte fields).
struct FissionSite {
    // Position (float3 in Metal = 16 bytes: x, y, z + 1 pad float)
    var posX: Float = 0;  var posY: Float = 0;  var posZ: Float = 0;  var _pad: Float = 0
    var energyGroup: UInt32 = 0
    var _pad2: UInt32 = 0
    var _pad3: UInt32 = 0
    var _pad4: UInt32 = 0

    /// Convenience SIMD3 position accessor.
    var position: SIMD3<Float> {
        get { SIMD3<Float>(posX, posY, posZ) }
        set { posX = newValue.x; posY = newValue.y; posZ = newValue.z }
    }

    init(position: SIMD3<Float> = .zero, energyGroup: UInt32 = 0) {
        self.posX = position.x
        self.posY = position.y
        self.posZ = position.z
        self.energyGroup = energyGroup
    }
}

// MARK: - FissionBank

/// Manages GPU buffers for the fission bank produced by the collision kernel.
/// The bank holds up to `maxSites` fission sites written by `collision` kernel.
/// An atomic counter buffer tracks how many sites have been written this batch.
final class FissionBank {
    let engine: MetalEngine

    /// GPU buffer holding `FissionSite` structs.
    private(set) var buffer: MTLBuffer

    /// GPU buffer holding a single `UInt32` atomic counter (number of sites banked).
    private(set) var countBuffer: MTLBuffer

    /// Maximum number of fission sites the bank can hold.
    let maxSites: Int

    // MARK: Init

    init(engine: MetalEngine, maxSites: Int) {
        self.engine = engine
        self.maxSites = maxSites
        self.buffer = engine.makeBuffer(length: maxSites * MemoryLayout<FissionSite>.stride)
        self.countBuffer = engine.makeBuffer(length: MemoryLayout<UInt32>.stride)
        reset()
    }

    // MARK: Control

    /// Zero the atomic counter before a new batch.
    func reset() {
        let ptr = countBuffer.contents().bindMemory(to: UInt32.self, capacity: 1)
        ptr.pointee = 0
    }

    /// Number of fission sites written by the GPU this batch (clamped to maxSites).
    var count: UInt32 {
        let ptr = countBuffer.contents().bindMemory(to: UInt32.self, capacity: 1)
        return min(ptr.pointee, UInt32(maxSites))
    }

    // MARK: Readback

    /// Read all banked fission sites back to the CPU.
    func sites() -> [FissionSite] {
        let n = Int(count)
        guard n > 0 else { return [] }
        let ptr = buffer.contents().bindMemory(to: FissionSite.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    // MARK: Reseed

    /// Reseed the GPU particle buffer from banked fission sites.
    /// Writes `count` particles back into `particleBuffer`, cycling through sites
    /// if `numParticles` > `count`. Resets the fission counter afterwards.
    func reseedParticles(
        particleBuffer: MTLBuffer,
        numParticles: Int,
        batchIndex: UInt32
    ) {
        let bankSites = sites()
        guard !bankSites.isEmpty else { return }

        let ptr = particleBuffer.contents().bindMemory(to: Particle.self, capacity: numParticles)
        for i in 0..<numParticles {
            let site = bankSites[i % bankSites.count]
            var p = Particle()
            p.position    = site.position
            p.energyGroup = site.energyGroup
            p.weight      = 1.0
            p.alive       = 1
            p.event       = ParticleEvent.xsLookup.rawValue
            p.rngKey      = UInt32(i) ^ (batchIndex &* 0x9E3779B9)
            p.rngCounter  = 0
            ptr[i] = p
        }
        reset()
    }
}
