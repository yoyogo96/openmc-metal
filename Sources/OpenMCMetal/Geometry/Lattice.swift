import simd

/// Rectangular lattice for modelling repeating structures such as fuel assemblies.
///
/// Elements are indexed (i, j, k) where i is the X dimension, j is Y, k is Z.
/// The flat `universeMap` array stores cell offsets in row-major order:
///   index = k * (nx * ny) + j * nx + i
///
/// GPULattice is defined in Types.swift.
struct Lattice {
    /// World-space origin (corner) of the lattice at element (0,0,0).
    var origin: SIMD3<Float>
    /// Pitch (element size) along each axis.
    var pitch: SIMD3<Float>
    /// Number of elements along each axis: (nx, ny, nz).
    var dimensions: SIMD3<UInt32>
    /// Flat map from lattice element index to the cell offset in the global cell array.
    /// Length must equal nx * ny * nz.
    var universeMap: [UInt32]

    /// Create a lattice.
    ///
    /// - Parameters:
    ///   - origin: Lower-left-front corner of the lattice in world space.
    ///   - pitch:  Size of one lattice element along each axis.
    ///   - nx:     Number of elements along X.
    ///   - ny:     Number of elements along Y.
    ///   - nz:     Number of elements along Z.
    ///   - map:    2-D array (outer = layers along Z, inner = rows along Y×X) of cell offsets.
    ///             Each inner array must have length nx; the outer array must have length ny * nz.
    ///             For a 2-D lattice set nz = 1 and supply a single-element outer array.
    init(
        origin: SIMD3<Float>,
        pitch: SIMD3<Float>,
        nx: UInt32,
        ny: UInt32,
        nz: UInt32,
        map: [[UInt32]]
    ) {
        self.origin     = origin
        self.pitch      = pitch
        self.dimensions = SIMD3<UInt32>(nx, ny, nz)
        self.universeMap = map.flatMap { $0 }
    }

    // MARK: - Convenience Accessors

    /// Return the cell offset for the lattice element at (i, j, k).
    func cellOffset(i: UInt32, j: UInt32, k: UInt32) -> UInt32 {
        let nx = dimensions.x
        let ny = dimensions.y
        let flatIndex = k * (nx * ny) + j * nx + i
        return universeMap[Int(flatIndex)]
    }

    /// Total number of lattice elements.
    var count: Int {
        Int(dimensions.x) * Int(dimensions.y) * Int(dimensions.z)
    }
}
