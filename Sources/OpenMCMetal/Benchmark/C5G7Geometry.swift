// C5G7Geometry.swift
// OpenMC-Metal
//
// C5G7 benchmark pincell geometry builder.
//
// Builds the simplest C5G7 geometry: a single UO2 fuel pin (radius 0.54 cm)
// centered in a square moderator cell with pitch 1.26 cm and z-extent 0..1 cm.
// Reflective boundary conditions on all 6 outer planes simulate an infinite
// lattice — this is the standard "reflected pincell" benchmark.
//
// Reference: NEA/NSC C5G7 MOX Benchmark (2003), NEA/NSC/DOC(2003)16.

import Metal
import simd

struct C5G7Geometry {

    // -------------------------------------------------------------------------
    // MARK: Pincell geometry
    // -------------------------------------------------------------------------

    /// Build a single reflected UO2 pincell.
    ///
    /// Geometry:
    ///   - 7 surfaces: 4 x/y planes + 2 z planes + 1 fuel cylinder
    ///   - 2 cells: fuel (mat 0 = UO2) and moderator (mat 6 = water)
    ///   - Boundary condition: reflective on all 6 outer planes
    ///
    /// Surface index map:
    ///   0  planeX  x = 0.00   (reflective, left)
    ///   1  planeX  x = 1.26   (reflective, right)
    ///   2  planeY  y = 0.00   (reflective, bottom)
    ///   3  planeY  y = 1.26   (reflective, top)
    ///   4  planeZ  z = 0.00   (reflective, back)
    ///   5  planeZ  z = 1.00   (reflective, front)
    ///   6  cylinderZ center=(0.63,0.63) r=0.54  (internal, vacuum)
    static func buildPincell(engine: MetalEngine) -> GeometryData {

        // ---- Surfaces ----
        var surfaces: [Surface] = []

        // Outer bounding planes — all reflective (simulates infinite lattice)
        surfaces.append(Surface.planeX(0.0,  bc: .reflective))   // 0: left   x=0
        surfaces.append(Surface.planeX(1.26, bc: .reflective))   // 1: right  x=1.26
        surfaces.append(Surface.planeY(0.0,  bc: .reflective))   // 2: bottom y=0
        surfaces.append(Surface.planeY(1.26, bc: .reflective))   // 3: top    y=1.26
        surfaces.append(Surface.planeZ(0.0,  bc: .reflective))   // 4: back   z=0
        surfaces.append(Surface.planeZ(1.0,  bc: .reflective))   // 5: front  z=1.0

        // Fuel cylinder — internal surface, vacuum BC (never hit from outside)
        surfaces.append(
            Surface.cylinderZ(x0: 0.63, y0: 0.63, radius: 0.54) // 6: fuel pin
        )

        // Assign sequential IDs
        for i in 0..<surfaces.count {
            surfaces[i].id = i
        }

        // ---- Cells ----

        // Cell 0: UO2 fuel — inside the cylinder and inside the bounding box.
        // Senses: positive half-space of min planes, negative of max planes,
        //         negative half-space of the cylinder (inside).
        let fuel = Cell(
            material: 0,   // UO2 (material index 0 in C5G7Data)
            surfaces: [
                (index: UInt32(0), sense: Int32( 1)),  // x > 0.00
                (index: UInt32(1), sense: Int32(-1)),  // x < 1.26
                (index: UInt32(2), sense: Int32( 1)),  // y > 0.00
                (index: UInt32(3), sense: Int32(-1)),  // y < 1.26
                (index: UInt32(4), sense: Int32( 1)),  // z > 0.00
                (index: UInt32(5), sense: Int32(-1)),  // z < 1.00
                (index: UInt32(6), sense: Int32(-1)),  // inside cylinder
            ]
        )

        // Cell 1: Water moderator — outside the cylinder, inside the bounding box.
        let moderator = Cell(
            material: 6,   // Moderator (material index 6 in C5G7Data)
            surfaces: [
                (index: UInt32(0), sense: Int32( 1)),  // x > 0.00
                (index: UInt32(1), sense: Int32(-1)),  // x < 1.26
                (index: UInt32(2), sense: Int32( 1)),  // y > 0.00
                (index: UInt32(3), sense: Int32(-1)),  // y < 1.26
                (index: UInt32(4), sense: Int32( 1)),  // z > 0.00
                (index: UInt32(5), sense: Int32(-1)),  // z < 1.00
                (index: UInt32(6), sense: Int32( 1)),  // outside cylinder
            ]
        )

        return GeometryData.build(engine: engine,
                                  surfaces: surfaces,
                                  cells: [fuel, moderator])
    }

    // -------------------------------------------------------------------------
    // MARK: Source sampler
    // -------------------------------------------------------------------------

    /// Generate fission source particles uniformly distributed inside the fuel
    /// cylinder with energy groups sampled from the UO2 chi spectrum.
    ///
    /// Chi (UO2): [0.587910, 0.411760, 0.000340, 0, 0, 0, 0]
    ///
    /// - Parameters:
    ///   - numParticles: Number of source particles to generate.
    ///   - numGroups:    Total number of energy groups (must be >= 3 for UO2 chi).
    /// - Returns: Array of (position, energyGroup) pairs.
    static func sourceSampler(
        numParticles: Int,
        numGroups: Int = 7
    ) -> [(position: SIMD3<Float>, group: UInt32)] {

        var sources: [(position: SIMD3<Float>, group: UInt32)] = []
        sources.reserveCapacity(numParticles)

        var rng = SystemRandomNumberGenerator()

        // Fuel cylinder geometry
        let cx: Float = 0.63
        let cy: Float = 0.63
        let r:  Float = 0.54
        let r2: Float = r * r

        // Bounding box for the cylinder
        let xMin = cx - r  // 0.09
        let xMax = cx + r  // 1.17
        let yMin = cy - r  // 0.09
        let yMax = cy + r  // 1.17

        // Cumulative chi CDF for UO2 (groups 0..6)
        // chi = [0.587910, 0.411760, 0.000340, 0, 0, 0, 0]
        let chiCDF: [Float] = [
            0.58791,          // group 0: 0..0.58791
            0.58791 + 0.41176, // group 1: ..0.99967
            1.0               // group 2: ..1.0
        ]

        var generated = 0
        while generated < numParticles {
            // Uniform rejection sampling inside cylinder
            let x = Float.random(in: xMin...xMax, using: &rng)
            let y = Float.random(in: yMin...yMax, using: &rng)
            let dx = x - cx
            let dy = y - cy
            guard dx * dx + dy * dy <= r2 else { continue }

            // Sample z uniformly in (0, 1)
            let z = Float.random(in: 0.001...0.999, using: &rng)

            // Sample energy group from chi CDF
            let xi = Float.random(in: 0.0...1.0, using: &rng)
            let group: UInt32
            if xi < chiCDF[0] {
                group = 0
            } else if xi < chiCDF[1] {
                group = 1
            } else {
                group = 2
            }

            sources.append((position: SIMD3<Float>(x, y, z), group: group))
            generated += 1
        }

        return sources
    }

    // -------------------------------------------------------------------------
    // MARK: Reference values
    // -------------------------------------------------------------------------

    /// Approximate k-eff reference for the reflected UO2 pincell (7-group C5G7).
    /// This is not the official NEA benchmark value (which requires a full assembly);
    /// it is an approximate value for a single-pin infinite-lattice calculation.
    static let referenceKEff: Double = 1.33007
}
