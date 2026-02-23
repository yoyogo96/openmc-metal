import Foundation

// MARK: - Philox-2x32-10 RNG (CPU Reference Implementation)
//
// This implements the same Philox-2x32-10 algorithm as the Metal shader in Common.metal.
// Used for CPU-side validation, source sampling, and testing.
//
// Philox is a counter-based RNG: given a (counter, key) pair it produces a
// deterministic pseudo-random output. This makes it ideal for GPU parallelism
// since each thread can have its own counter without shared state.

/// Constants matching the Metal implementation exactly.
private let PHILOX_M: UInt32 = 0xD2511F53
private let PHILOX_W: UInt32 = 0x9E3779B9

/// Conversion factor: 1.0 / 2^32
private let UINT32_TO_FLOAT: Float = 2.3283064365e-10

// MARK: - Core Algorithm

/// Single Philox round: mix counter halves using multiply-and-XOR.
@inline(__always)
func philox2x32Round(_ ctr: (UInt32, UInt32), key: UInt32) -> (UInt32, UInt32) {
    let product = UInt64(PHILOX_M) &* UInt64(ctr.0)
    let hi = UInt32(product >> 32)
    let lo = UInt32(truncatingIfNeeded: product)
    return (hi ^ key ^ ctr.1, lo)
}

/// Full Philox-2x32-10: 10 rounds of mixing.
/// Matches `philox2x32_10` in Common.metal.
func philox2x32_10(counter: (UInt32, UInt32), key: UInt32) -> (UInt32, UInt32) {
    var ctr = counter
    var k = key
    for _ in 0..<10 {
        ctr = philox2x32Round(ctr, key: k)
        k = k &+ PHILOX_W
    }
    return ctr
}

// MARK: - PhiloxRNG State

/// CPU-side Philox RNG state, matching the GPU per-particle RNG state.
struct PhiloxRNG {
    /// Low word of the counter (incremented each draw)
    var counterLo: UInt32
    /// High word of the counter (typically batch/particle ID)
    var counterHi: UInt32
    /// Key (typically particle index or unique seed)
    var key: UInt32

    /// Initialize with a unique key and optional counter values.
    init(key: UInt32, counterHi: UInt32 = 0, counterLo: UInt32 = 0) {
        self.key = key
        self.counterHi = counterHi
        self.counterLo = counterLo
    }

    /// Draw a uniform random Float in [0, 1).
    /// Advances the counter by 1, matching `philox_uniform` in Common.metal.
    mutating func uniform() -> Float {
        let result = philox2x32_10(counter: (counterLo, counterHi), key: key)
        counterLo &+= 1
        return Float(result.0) * UINT32_TO_FLOAT
    }

    /// Draw a raw UInt32 random value.
    mutating func next() -> UInt32 {
        let result = philox2x32_10(counter: (counterLo, counterHi), key: key)
        counterLo &+= 1
        return result.0
    }

    /// Sample an isotropic direction on the unit sphere.
    mutating func sampleIsotropicDirection() -> SIMD3<Float> {
        let cosTheta = 2.0 * uniform() - 1.0
        let sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta))
        let phi = 2.0 * Float.pi * uniform()
        return SIMD3<Float>(
            sinTheta * cos(phi),
            sinTheta * sin(phi),
            cosTheta
        )
    }

    /// Sample from a discrete distribution using CDF.
    /// Returns the index of the selected bin.
    mutating func sampleDiscrete(cdf: [Float]) -> Int {
        let xi = uniform()
        for (i, cumProb) in cdf.enumerated() {
            if xi < cumProb { return i }
        }
        return cdf.count - 1
    }
}
