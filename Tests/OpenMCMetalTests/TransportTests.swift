#if canImport(XCTest)
import XCTest
@testable import OpenMCMetal

final class TransportTests: XCTestCase {

    // MARK: - Cross-section loading

    func testC5G7MaterialCount() {
        let materials = C5G7Data.allMaterials()
        XCTAssertEqual(materials.count, 7, "C5G7 benchmark has exactly 7 materials")
    }

    func testUO2TotalXSGroup0() {
        let uo2 = C5G7Data.uo2()
        XCTAssertEqual(uo2.total[0], 0.212033, accuracy: 1e-5,
            "UO2 total XS group 0 should match NEA benchmark value")
    }

    func testUO2TotalXSAllGroups() {
        let uo2 = C5G7Data.uo2()
        XCTAssertEqual(uo2.total.count, 7, "total XS array should have 7 entries")
        // Spot-check a few published values
        XCTAssertEqual(uo2.total[1], 0.332717, accuracy: 1e-5, "UO2 total XS group 1")
        XCTAssertEqual(uo2.total[6], 0.531440, accuracy: 1e-5, "UO2 total XS group 6")
    }

    func testModeratorZeroFission() {
        let water = C5G7Data.moderator()
        for g in 0..<7 {
            XCTAssertEqual(water.fission[g], 0.0,
                "Moderator fission XS group \(g) should be zero")
        }
    }

    func testModeratorZeroNuFission() {
        let water = C5G7Data.moderator()
        for g in 0..<7 {
            XCTAssertEqual(water.nuFission[g], 0.0,
                "Moderator nu-fission XS group \(g) should be zero")
        }
    }

    func testGuideTubeZeroFission() {
        let gt = C5G7Data.guideTube()
        for g in 0..<7 {
            XCTAssertEqual(gt.fission[g], 0.0,
                "Guide tube fission XS group \(g) should be zero")
        }
    }

    func testFissionChamberHasFission() {
        let fc = C5G7Data.fissionChamber()
        // Group 0 fission XS should be non-zero
        XCTAssertGreaterThan(fc.fission[0], 0.0,
            "Fission chamber should have non-zero fission XS in group 0")
    }

    // MARK: - Chi (fission spectrum) normalization

    func testChiNormalizationUO2() {
        let uo2 = C5G7Data.uo2()
        let chiSum = uo2.chi.reduce(0.0, +)
        XCTAssertEqual(Double(chiSum), 1.0, accuracy: 0.001,
            "UO2 chi should sum to ~1.0, got \(chiSum)")
    }

    func testChiNormalizationAllFissileMaterials() {
        let materials = C5G7Data.allMaterials()
        for (i, mat) in materials.enumerated() {
            let chiSum = mat.chi.reduce(0.0, +)
            if chiSum > 0.01 {
                // Material is fissile — chi must normalise to 1
                XCTAssertEqual(Double(chiSum), 1.0, accuracy: 0.002,
                    "Material \(i) chi sums to \(chiSum), expected ~1.0")
            } else {
                // Non-fissile (guide tube, moderator) — chi must be all zeros
                XCTAssertEqual(Double(chiSum), 0.0, accuracy: 1e-9,
                    "Non-fissile material \(i) chi should be zero, got \(chiSum)")
            }
        }
    }

    // MARK: - Scatter matrix consistency

    func testScatterMatrixRowSums() {
        // The sum of outscatter from group g must not exceed total XS (within tolerance).
        let materials = C5G7Data.allMaterials()
        for (i, mat) in materials.enumerated() {
            for g in 0..<7 {
                var rowSum: Float = 0
                for gp in 0..<7 {
                    rowSum += mat.scatter[g * 7 + gp]
                }
                XCTAssertLessThanOrEqual(rowSum, mat.total[g] + 1e-4,
                    "Material \(i) group \(g): scatter row sum (\(rowSum)) exceeds total XS (\(mat.total[g]))")
            }
        }
    }

    func testScatterMatrixNonNegative() {
        let materials = C5G7Data.allMaterials()
        for (i, mat) in materials.enumerated() {
            for j in 0..<49 {
                XCTAssertGreaterThanOrEqual(mat.scatter[j], 0.0,
                    "Material \(i) scatter[\(j)] = \(mat.scatter[j]) is negative")
            }
        }
    }

    func testScatterMatrixDimensions() {
        let materials = C5G7Data.allMaterials()
        for (i, mat) in materials.enumerated() {
            XCTAssertEqual(mat.scatter.count, 49,
                "Material \(i) scatter matrix should be 7x7 = 49 elements")
        }
    }

    // MARK: - GPU buffer packing

    func testMaterialGPUBufferSize() {
        // Each material should pack to exactly 77 floats (7+49+7+7+7)
        let materials = C5G7Data.allMaterials()
        for (i, mat) in materials.enumerated() {
            let packed = mat.toGPUBuffer()
            XCTAssertEqual(packed.count, 77,
                "Material \(i) GPU buffer should be 77 floats, got \(packed.count)")
        }
    }

    func testAllMaterialsGPUBufferSize() {
        // 7 materials * 77 floats = 539 floats total
        let materials = C5G7Data.allMaterials()
        let totalFloats = materials.reduce(0) { $0 + $1.toGPUBuffer().count }
        XCTAssertEqual(totalFloats, 539,
            "All 7 materials should pack to 539 floats, got \(totalFloats)")
    }

    // MARK: - Statistics (Welford)

    func testWelfordMeanSingleBin() {
        let batches: [[Double]] = [[1.0], [3.0], [5.0]]
        let stats = TallyStatistics.computeStats(batchValues: batches)
        XCTAssertEqual(stats[0].mean, 3.0, accuracy: 1e-10,
            "Mean of [1, 3, 5] should be 3.0")
    }

    func testWelfordMeanTwoBins() {
        let batches: [[Double]] = [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ]
        let stats = TallyStatistics.computeStats(batchValues: batches)
        XCTAssertEqual(stats[0].mean, 3.0, accuracy: 1e-10, "Bin 0 mean of [1,3,5] should be 3.0")
        XCTAssertEqual(stats[1].mean, 4.0, accuracy: 1e-10, "Bin 1 mean of [2,4,6] should be 4.0")
    }

    func testWelfordStdDevKnown() {
        // For values [1, 3, 5]: sample variance = ((1-3)^2+(3-3)^2+(5-3)^2)/2 = 4
        // std dev of mean = sqrt(4/3) ≈ 1.1547
        let batches: [[Double]] = [[1.0], [3.0], [5.0]]
        let stats = TallyStatistics.computeStats(batchValues: batches)
        let expected = sqrt(4.0 / 3.0)
        XCTAssertEqual(stats[0].stdDev, expected, accuracy: 1e-10)
    }

    func testWelfordRelativeError() {
        let batches: [[Double]] = [[1.0], [3.0], [5.0]]
        let stats = TallyStatistics.computeStats(batchValues: batches)
        let expectedRelError = stats[0].stdDev / abs(stats[0].mean)
        XCTAssertEqual(stats[0].relError, expectedRelError, accuracy: 1e-10)
    }

    func testWelfordSingleBatchZeroStdDev() {
        let batches: [[Double]] = [[2.0]]
        let stats = TallyStatistics.computeStats(batchValues: batches)
        XCTAssertEqual(stats[0].mean, 2.0, accuracy: 1e-10)
        XCTAssertEqual(stats[0].stdDev, 0.0, accuracy: 1e-10,
            "Single-batch std dev should be 0")
    }

    func testWelfordEmptyBatches() {
        let stats = TallyStatistics.computeStats(batchValues: [])
        XCTAssertTrue(stats.isEmpty, "Empty input should return empty results")
    }

    // MARK: - k-eff statistics

    func testKEffMean() {
        let kEffValues: [Double] = [1.0, 1.1, 0.9, 1.05, 0.95]
        let stats = TallyStatistics.kEffStatistics(batchKEff: kEffValues)
        XCTAssertEqual(stats.mean, 1.0, accuracy: 0.01,
            "Mean k-eff of [1.0,1.1,0.9,1.05,0.95] should be ~1.0")
    }

    func testKEffStdDevPositive() {
        let kEffValues: [Double] = [1.0, 1.1, 0.9, 1.05, 0.95]
        let stats = TallyStatistics.kEffStatistics(batchKEff: kEffValues)
        XCTAssertGreaterThan(stats.stdDev, 0, "Std dev should be positive for non-constant sequence")
    }

    func testKEffCI95Ordering() {
        let kEffValues: [Double] = [1.0, 1.1, 0.9, 1.05, 0.95]
        let stats = TallyStatistics.kEffStatistics(batchKEff: kEffValues)
        XCTAssertLessThan(stats.ci95.0, stats.mean, "Lower 95% CI bound should be below mean")
        XCTAssertGreaterThan(stats.ci95.1, stats.mean, "Upper 95% CI bound should be above mean")
    }

    func testKEffCI95Width() {
        // CI width = 2 * 1.96 * stdDev
        let kEffValues: [Double] = [1.0, 1.1, 0.9, 1.05, 0.95]
        let stats = TallyStatistics.kEffStatistics(batchKEff: kEffValues)
        let expectedWidth = 2 * 1.96 * stats.stdDev
        let actualWidth = stats.ci95.1 - stats.ci95.0
        XCTAssertEqual(actualWidth, expectedWidth, accuracy: 1e-10)
    }

    func testKEffSingleValue() {
        let stats = TallyStatistics.kEffStatistics(batchKEff: [1.2])
        XCTAssertEqual(stats.mean, 1.2, accuracy: 1e-10)
        XCTAssertEqual(stats.stdDev, 0.0, accuracy: 1e-10)
    }

    func testKEffEmpty() {
        let stats = TallyStatistics.kEffStatistics(batchKEff: [])
        XCTAssertEqual(stats.mean, 0.0, accuracy: 1e-10)
        XCTAssertEqual(stats.stdDev, 0.0, accuracy: 1e-10)
    }

    // MARK: - Figure of merit

    func testFigureOfMerit() {
        // FOM = 1 / (R^2 * T); R=0.01 (1%), T=100s → FOM=10000
        let fom = TallyStatistics.figureOfMerit(relError: 0.01, wallTime: 100.0)
        XCTAssertEqual(fom, 10_000.0, accuracy: 1e-6)
    }

    func testFigureOfMeritZeroRelError() {
        let fom = TallyStatistics.figureOfMerit(relError: 0.0, wallTime: 100.0)
        XCTAssertEqual(fom, 0.0, "FOM should be 0 when relative error is 0")
    }

    func testFigureOfMeritZeroTime() {
        let fom = TallyStatistics.figureOfMerit(relError: 0.01, wallTime: 0.0)
        XCTAssertEqual(fom, 0.0, "FOM should be 0 when wall time is 0")
    }

    // MARK: - Batch means helper

    func testBatchMeans() {
        let batches: [[Double]] = [
            [1.0, 10.0],
            [3.0, 20.0],
            [5.0, 30.0]
        ]
        let means = TallyStatistics.batchMeans(batchValues: batches, size: 2)
        XCTAssertEqual(means[0], 3.0, accuracy: 1e-10, "Bin 0 mean should be 3.0")
        XCTAssertEqual(means[1], 20.0, accuracy: 1e-10, "Bin 1 mean should be 20.0")
    }
}
#endif
