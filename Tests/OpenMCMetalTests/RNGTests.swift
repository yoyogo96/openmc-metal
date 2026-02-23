#if canImport(XCTest)
import XCTest
@testable import OpenMCMetal

final class RNGTests: XCTestCase {

    func testPhiloxDeterministic() {
        // Same key + counter must produce the same sequence
        var rng1 = PhiloxRNG(key: 12345, counterHi: 42, counterLo: 0)
        var rng2 = PhiloxRNG(key: 12345, counterHi: 42, counterLo: 0)

        for _ in 0..<100 {
            XCTAssertEqual(rng1.uniform(), rng2.uniform())
        }
    }

    func testPhiloxRange() {
        // All values should be in [0, 1)
        var rng = PhiloxRNG(key: 42, counterHi: 0, counterLo: 0)
        for _ in 0..<10_000 {
            let val = rng.uniform()
            XCTAssertGreaterThanOrEqual(val, 0.0)
            XCTAssertLessThan(val, 1.0)
        }
    }

    func testPhiloxUniformity() {
        // Chi-squared test for uniformity across 10 bins
        let numSamples = 100_000
        let numBins = 10
        var bins = [Int](repeating: 0, count: numBins)
        var rng = PhiloxRNG(key: 42, counterHi: 0, counterLo: 0)

        for _ in 0..<numSamples {
            let val = rng.uniform()
            let bin = min(Int(val * Float(numBins)), numBins - 1)
            bins[bin] += 1
        }

        // Chi-squared statistic
        let expected = Double(numSamples) / Double(numBins)
        var chiSq = 0.0
        for count in bins {
            let diff = Double(count) - expected
            chiSq += diff * diff / expected
        }

        // 9 degrees of freedom: chi-sq < 21.67 at p=0.01
        XCTAssertLessThan(chiSq, 21.67,
            "RNG uniformity test failed (chi-sq = \(chiSq)); expected < 21.67")
    }

    func testPhiloxDifferentKeys() {
        // Different keys must not produce identical sequences
        var rng1 = PhiloxRNG(key: 1, counterHi: 0, counterLo: 0)
        var rng2 = PhiloxRNG(key: 2, counterHi: 0, counterLo: 0)

        var allSame = true
        for _ in 0..<100 {
            if rng1.uniform() != rng2.uniform() {
                allSame = false
                break
            }
        }
        XCTAssertFalse(allSame, "Different keys should produce different sequences")
    }

    func testPhiloxNextReturnsUInt32() {
        // next() returns raw UInt32, not Float
        var rng = PhiloxRNG(key: 7, counterHi: 0, counterLo: 0)
        // Just verify it does not crash and returns different values
        let a = rng.next()
        let b = rng.next()
        // Two consecutive draws from the same counter sequence should differ
        // (astronomically unlikely to collide for any key)
        XCTAssertNotEqual(a, b, "Consecutive next() calls should return different values")
    }

    func testPhiloxCounterAdvances() {
        // Each call to uniform() should advance counterLo by 1
        var rng = PhiloxRNG(key: 99, counterHi: 0, counterLo: 0)
        _ = rng.uniform()
        XCTAssertEqual(rng.counterLo, 1)
        _ = rng.uniform()
        XCTAssertEqual(rng.counterLo, 2)
    }

    func testPhiloxIsotropicDirectionOnSphere() {
        // sampleIsotropicDirection should return unit vectors
        var rng = PhiloxRNG(key: 17, counterHi: 0, counterLo: 0)
        for _ in 0..<1_000 {
            let dir = rng.sampleIsotropicDirection()
            let len = sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z)
            XCTAssertEqual(len, 1.0, accuracy: 1e-5,
                "sampleIsotropicDirection should return a unit vector, got length \(len)")
        }
    }

    func testPhiloxSampleDiscrete() {
        // sampleDiscrete with a uniform CDF over 3 bins
        // CDF: [1/3, 2/3, 1.0]
        let cdf: [Float] = [1.0/3.0, 2.0/3.0, 1.0]
        var rng = PhiloxRNG(key: 55, counterHi: 0, counterLo: 0)
        var counts = [Int](repeating: 0, count: 3)

        let trials = 30_000
        for _ in 0..<trials {
            let idx = rng.sampleDiscrete(cdf: cdf)
            XCTAssertGreaterThanOrEqual(idx, 0)
            XCTAssertLessThan(idx, 3)
            counts[idx] += 1
        }

        // Each bin should be roughly 10000 ± a generous tolerance
        for bin in 0..<3 {
            let ratio = Double(counts[bin]) / Double(trials)
            XCTAssertEqual(ratio, 1.0/3.0, accuracy: 0.02,
                "Bin \(bin) had fraction \(ratio), expected ~0.333")
        }
    }
}
#endif
