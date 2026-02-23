import Metal
import simd

/// Swift-side cell definition using CSG half-space intersection.
/// A particle is inside a cell if it satisfies all surface sense conditions.
///
/// GPUCell, GPUCellSurface, and MetalEngine are defined elsewhere in the module.
struct Cell {
    /// Index into the materials array for this cell.
    var materialIndex: UInt32
    /// Ordered list of surface indices (into the global surfaces array) bounding this cell.
    var surfaceIndices: [UInt32]
    /// Sense for each surface: +1 = positive half-space, -1 = negative half-space.
    var surfaceSenses: [Int32]

    /// Create a cell from a material index and an array of (surfaceIndex, sense) pairs.
    init(material: UInt32, surfaces: [(index: UInt32, sense: Int32)]) {
        self.materialIndex = material
        self.surfaceIndices = surfaces.map { $0.index }
        self.surfaceSenses = surfaces.map { $0.sense }
    }
}

// MARK: - GeometryData

/// Packed Metal buffers ready to be passed to GPU kernels.
struct GeometryData {
    /// Buffer of GPUSurface structs, one per surface.
    let surfaceBuffer: MTLBuffer
    /// Buffer of GPUCell structs, one per cell.
    let cellBuffer: MTLBuffer
    /// Flat buffer of GPUCellSurface structs for all cells concatenated.
    let cellSurfaceBuffer: MTLBuffer
    /// Optional buffer of GPULattice structs (nil when no lattice is used).
    let latticeBuffer: MTLBuffer?
    /// Optional flat buffer mapping lattice element (i,j,k) -> cell offset (nil when no lattice).
    let universeMapBuffer: MTLBuffer?
    /// Total number of surfaces.
    let numSurfaces: UInt32
    /// Total number of cells.
    let numCells: UInt32

    // MARK: Build

    /// Pack surfaces, cells, and an optional lattice into GPU buffers.
    ///
    /// - Parameters:
    ///   - engine:   The MetalEngine used to allocate buffers.
    ///   - surfaces: All surfaces referenced by the cells.
    ///   - cells:    All cells in the geometry.
    ///   - lattice:  Optional rectangular lattice; pass nil for simple geometries.
    static func build(
        engine: MetalEngine,
        surfaces: [Surface],
        cells: [Cell],
        lattice: Lattice? = nil
    ) -> GeometryData {

        // ---- Surfaces ----
        let gpuSurfaces = surfaces.map { $0.gpuSurface }
        let surfaceBuf = engine.makeBuffer(gpuSurfaces)

        // ---- Cells + flat cell-surface pairs ----
        var gpuCells: [GPUCell] = []
        var cellSurfaces: [GPUCellSurface] = []
        var offset: UInt32 = 0

        for cell in cells {
            var gc = GPUCell()
            gc.materialIndex = cell.materialIndex
            gc.numSurfaces   = UInt32(cell.surfaceIndices.count)
            gc.surfaceOffset = offset
            gpuCells.append(gc)

            for i in 0..<cell.surfaceIndices.count {
                var cs = GPUCellSurface()
                cs.surfaceIndex = cell.surfaceIndices[i]
                cs.sense        = cell.surfaceSenses[i]
                cellSurfaces.append(cs)
            }
            offset += UInt32(cell.surfaceIndices.count)
        }

        let cellBuf        = engine.makeBuffer(gpuCells)
        let cellSurfaceBuf = engine.makeBuffer(cellSurfaces)

        // ---- Optional lattice ----
        var latticeBuf:     MTLBuffer? = nil
        var universeMapBuf: MTLBuffer? = nil

        if let lat = lattice {
            var gpuLat = GPULattice()
            gpuLat.origin     = lat.origin
            gpuLat.pitch      = lat.pitch
            gpuLat.dimensions = lat.dimensions
            latticeBuf = engine.makeBuffer([gpuLat])

            universeMapBuf = engine.makeBuffer(lat.universeMap)
        }

        return GeometryData(
            surfaceBuffer:    surfaceBuf,
            cellBuffer:       cellBuf,
            cellSurfaceBuffer: cellSurfaceBuf,
            latticeBuffer:    latticeBuf,
            universeMapBuffer: universeMapBuf,
            numSurfaces:      UInt32(surfaces.count),
            numCells:         UInt32(cells.count)
        )
    }
}
