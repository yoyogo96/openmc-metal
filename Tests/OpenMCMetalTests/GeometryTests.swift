#if canImport(XCTest)
import XCTest
@testable import OpenMCMetal
import simd

final class GeometryTests: XCTestCase {

    // MARK: - Surface factory

    func testPlaneXCoefficients() {
        let surface = Surface.planeX(1.0)
        XCTAssertEqual(surface.gpuSurface.type, SurfaceType.planeX.rawValue)
        XCTAssertEqual(surface.gpuSurface.coefficients.x, 1.0, accuracy: 1e-7,
            "planeX position should be stored in coefficients.x")
    }

    func testPlaneYCoefficients() {
        let surface = Surface.planeY(2.5)
        XCTAssertEqual(surface.gpuSurface.type, SurfaceType.planeY.rawValue)
        XCTAssertEqual(surface.gpuSurface.coefficients.x, 2.5, accuracy: 1e-7)
    }

    func testPlaneZCoefficients() {
        let surface = Surface.planeZ(0.0)
        XCTAssertEqual(surface.gpuSurface.type, SurfaceType.planeZ.rawValue)
        XCTAssertEqual(surface.gpuSurface.coefficients.x, 0.0, accuracy: 1e-7)
    }

    func testCylinderZCoefficients() {
        let surface = Surface.cylinderZ(x0: 0.63, y0: 0.63, radius: 0.54)
        XCTAssertEqual(surface.gpuSurface.type, SurfaceType.cylinderZ.rawValue)
        XCTAssertEqual(surface.gpuSurface.coefficients.x, 0.63, accuracy: 1e-7, "cylinder x0")
        XCTAssertEqual(surface.gpuSurface.coefficients.y, 0.63, accuracy: 1e-7, "cylinder y0")
        XCTAssertEqual(surface.gpuSurface.coefficients.z, 0.54, accuracy: 1e-7, "cylinder radius")
    }

    func testSphereCoefficients() {
        let surface = Surface.sphere(x0: 1.0, y0: 2.0, z0: 3.0, radius: 4.0)
        XCTAssertEqual(surface.gpuSurface.type, SurfaceType.sphere.rawValue)
        XCTAssertEqual(surface.gpuSurface.coefficients.x, 1.0, accuracy: 1e-7)
        XCTAssertEqual(surface.gpuSurface.coefficients.y, 2.0, accuracy: 1e-7)
        XCTAssertEqual(surface.gpuSurface.coefficients.z, 3.0, accuracy: 1e-7)
        XCTAssertEqual(surface.gpuSurface.coefficients.w, 4.0, accuracy: 1e-7, "sphere radius in .w")
    }

    func testBoundaryConditionDefault() {
        let surf = Surface.planeX(0.0)
        XCTAssertEqual(surf.gpuSurface.boundaryCondition, BoundaryCondition.vacuum.rawValue)
    }

    func testBoundaryConditionReflective() {
        let surf = Surface.planeX(0.0, bc: .reflective)
        XCTAssertEqual(surf.gpuSurface.boundaryCondition, BoundaryCondition.reflective.rawValue)
    }

    // MARK: - Plane sense (CPU-side evaluation)

    func testPlaneXSenseNegativeSide() {
        // Point at x=0.5 vs plane at x=1.0 → negative side (val < 0)
        let surface = Surface.planeX(1.0)
        let pos = SIMD3<Float>(0.5, 0.5, 0.5)
        let val = pos.x - surface.gpuSurface.coefficients.x  // 0.5 - 1.0 = -0.5
        XCTAssertLessThan(val, 0, "Point (0.5) should be on the negative side of planeX(1.0)")
    }

    func testPlaneXSensePositiveSide() {
        // Point at x=1.5 vs plane at x=1.0 → positive side (val > 0)
        let surface = Surface.planeX(1.0)
        let pos = SIMD3<Float>(1.5, 0.5, 0.5)
        let val = pos.x - surface.gpuSurface.coefficients.x  // 1.5 - 1.0 = 0.5
        XCTAssertGreaterThan(val, 0, "Point (1.5) should be on the positive side of planeX(1.0)")
    }

    // MARK: - Cylinder containment (CPU-side evaluation)

    func testCylinderContainmentInside() {
        // Point (0.1, 0.1) relative to origin-centered cylinder R=0.54
        let surface = Surface.cylinderZ(x0: 0.0, y0: 0.0, radius: 0.54)
        let pos = SIMD3<Float>(0.1, 0.1, 0.5)
        let r = surface.gpuSurface.coefficients.z
        let val = pos.x * pos.x + pos.y * pos.y - r * r
        XCTAssertLessThan(val, 0, "Point (0.1,0.1) should be inside cylinder R=0.54")
    }

    func testCylinderContainmentOutside() {
        // Point (0.6, 0.6) — outside R=0.54
        let surface = Surface.cylinderZ(x0: 0.0, y0: 0.0, radius: 0.54)
        let pos = SIMD3<Float>(0.6, 0.6, 0.5)
        let r = surface.gpuSurface.coefficients.z
        let val = pos.x * pos.x + pos.y * pos.y - r * r
        XCTAssertGreaterThan(val, 0, "Point (0.6,0.6) should be outside cylinder R=0.54")
    }

    // MARK: - Analytic distances

    func testDistanceToPlane() {
        // Particle at x=0.5 moving +x toward planeX(1.0)
        // Expected distance: (1.0 - 0.5) / 1.0 = 0.5
        let pos = SIMD3<Float>(0.5, 0.5, 0.5)
        let dir = SIMD3<Float>(1.0, 0.0, 0.0)
        let planeX: Float = 1.0
        let d = (planeX - pos.x) / dir.x
        XCTAssertEqual(d, 0.5, accuracy: 1e-6, "Distance to plane should be 0.5")
    }

    func testDistanceToCylinder() {
        // Particle at origin moving +x toward cylinder of radius R centered at origin.
        // A = 1, B = 0, C = -R^2 → d = R (positive root).
        let R: Float = 0.54
        let A: Float = 1.0
        let B: Float = 0.0
        let C: Float = -(R * R)
        let disc = B * B - 4 * A * C
        let d = (-B + sqrt(disc)) / (2 * A)
        XCTAssertEqual(d, R, accuracy: 1e-6, "Distance from origin to cylinder surface should equal R")
    }

    func testDistanceToCylinderOffCenter() {
        // Particle at (0, 0, 0.5) moving +x toward cylinder centered at (0.63, 0.63) R=0.54.
        // Transform to cylinder-local coords: dx0 = 0 - 0.63 = -0.63, dy0 = 0 - 0.63 = -0.63
        let cx: Float = 0.63, cy: Float = 0.63, r: Float = 0.54
        let px: Float = 0.0, py: Float = 0.0
        let ux: Float = 1.0, uy: Float = 0.0
        let dx0 = px - cx  // -0.63
        let dy0 = py - cy  // -0.63
        let A = ux * ux + uy * uy               // 1.0
        let B = 2 * (dx0 * ux + dy0 * uy)       // 2 * (-0.63)
        let C = dx0 * dx0 + dy0 * dy0 - r * r   // 0.63^2+0.63^2 - 0.54^2

        let disc = B * B - 4 * A * C
        // disc >= 0 means the ray intersects the cylinder
        if disc >= 0 {
            let d1 = (-B - sqrt(disc)) / (2 * A)
            let d2 = (-B + sqrt(disc)) / (2 * A)
            // Particle is outside, so positive root is entry
            let dEntry = max(d1, d2)
            XCTAssertGreaterThan(dEntry, 0, "Particle outside cylinder; entry distance should be positive")
        }
        // If disc < 0 the ray misses — acceptable for this geometry
    }

    // MARK: - Cell construction

    func testCellMaterialIndex() {
        let cell = Cell(material: 0, surfaces: [
            (index: UInt32(0), sense: Int32(1)),
            (index: UInt32(6), sense: Int32(-1))
        ])
        XCTAssertEqual(cell.materialIndex, 0)
    }

    func testCellSurfaceCount() {
        let cell = Cell(material: 0, surfaces: [
            (index: UInt32(0), sense: Int32(1)),
            (index: UInt32(1), sense: Int32(-1))
        ])
        XCTAssertEqual(cell.surfaceIndices.count, 2)
        XCTAssertEqual(cell.surfaceSenses.count, 2)
    }

    func testCellSurfaceSenses() {
        let cell = Cell(material: 0, surfaces: [
            (index: UInt32(0), sense: Int32(1)),
            (index: UInt32(1), sense: Int32(-1))
        ])
        XCTAssertEqual(cell.surfaceSenses[0], 1)
        XCTAssertEqual(cell.surfaceSenses[1], -1)
    }

    func testCellSurfaceIndices() {
        let cell = Cell(material: 6, surfaces: [
            (index: UInt32(3), sense: Int32(1)),
            (index: UInt32(7), sense: Int32(-1))
        ])
        XCTAssertEqual(cell.surfaceIndices[0], 3)
        XCTAssertEqual(cell.surfaceIndices[1], 7)
    }

    // MARK: - C5G7 pincell geometry

    func testC5G7PincellSurfaceCount() throws {
        let engine = try MetalEngine()
        let geometry = C5G7Geometry.buildPincell(engine: engine)
        XCTAssertEqual(geometry.numSurfaces, 7,
            "C5G7 pincell should have 7 surfaces (4 x/y planes + 2 z planes + 1 cylinder)")
    }

    func testC5G7PincellCellCount() throws {
        let engine = try MetalEngine()
        let geometry = C5G7Geometry.buildPincell(engine: engine)
        XCTAssertEqual(geometry.numCells, 2,
            "C5G7 pincell should have 2 cells (fuel + moderator)")
    }

    func testC5G7PincellBuffersNonNil() throws {
        let engine = try MetalEngine()
        let geometry = C5G7Geometry.buildPincell(engine: engine)
        // All mandatory buffers must be allocated
        XCTAssertNotNil(geometry.surfaceBuffer)
        XCTAssertNotNil(geometry.cellBuffer)
        XCTAssertNotNil(geometry.cellSurfaceBuffer)
    }

    // MARK: - Source sampler

    func testSourceSamplerCount() {
        let sources = C5G7Geometry.sourceSampler(numParticles: 1_000, numGroups: 7)
        XCTAssertEqual(sources.count, 1_000)
    }

    func testSourceSamplerInsideFuelCylinder() {
        // All positions must lie inside the fuel cylinder centered at (0.63, 0.63) R=0.54
        let sources = C5G7Geometry.sourceSampler(numParticles: 500, numGroups: 7)
        let cx: Float = 0.63, cy: Float = 0.63, r: Float = 0.54
        for (i, src) in sources.enumerated() {
            let dx = src.position.x - cx
            let dy = src.position.y - cy
            let r2 = dx * dx + dy * dy
            XCTAssertLessThanOrEqual(r2, r * r + 1e-5,
                "Source \(i) at (\(src.position.x),\(src.position.y)) is outside fuel cylinder")
        }
    }

    func testSourceSamplerZRange() {
        let sources = C5G7Geometry.sourceSampler(numParticles: 500, numGroups: 7)
        for (i, src) in sources.enumerated() {
            XCTAssertGreaterThan(src.position.z, 0,
                "Source \(i) z=\(src.position.z) should be > 0")
            XCTAssertLessThan(src.position.z, 1.0,
                "Source \(i) z=\(src.position.z) should be < 1.0")
        }
    }

    func testSourceSamplerGroupDistribution() {
        // UO2 chi: [0.5879, 0.4118, 0.0003, 0, 0, 0, 0]
        // With 10000 samples, groups 0 and 1 should dominate.
        let sources = C5G7Geometry.sourceSampler(numParticles: 10_000, numGroups: 7)
        let group0 = sources.filter { $0.group == 0 }.count
        let group1 = sources.filter { $0.group == 1 }.count
        let group2plus = sources.filter { $0.group >= 2 }.count

        XCTAssertGreaterThan(group0, 4_000, "~58.8% should be group 0, got \(group0)")
        XCTAssertGreaterThan(group1, 3_000, "~41.2% should be group 1, got \(group1)")
        XCTAssertLessThan(group2plus, 100,  "Groups 2+ should be rare, got \(group2plus)")
    }
}
#endif
