import Metal
import simd

// MARK: - Simulation Constants

/// Maximum number of energy groups supported
let MAX_ENERGY_GROUPS: Int = 7

/// Maximum number of surfaces bounding a single cell
let MAX_SURFACES_PER_CELL: Int = 16

/// Small distance to bump particles past surfaces (cm)
let BUMP_DISTANCE: Float = 1.0e-8

// MARK: - Enums

/// Particle event state machine states
enum ParticleEvent: UInt32 {
    case initialize = 0
    case xsLookup   = 1
    case distance   = 2
    case move        = 3
    case collide     = 4
    case tally       = 5
    case dead        = 6
    case census      = 7
}

/// Surface geometry types
enum SurfaceType: UInt32 {
    case planeX    = 0
    case planeY    = 1
    case planeZ    = 2
    case cylinderZ = 3
    case sphere    = 4
}

/// Boundary conditions for surfaces
enum BoundaryCondition: UInt32 {
    case vacuum     = 0
    case reflective = 1
}

// MARK: - GPU Structs
// These structs MUST have identical memory layout to Metal counterparts in Common.metal.
//
// IMPORTANT: We use scalar Float/UInt32 fields instead of SIMD3/SIMD4 to guarantee
// exact byte-for-byte layout matching Metal. SIMD3<Float> in Swift has 16-byte alignment
// which causes unexpected padding when combined with other fields. Scalar fields give
// us 4-byte alignment and predictable layout.
//
// Computed SIMD3/SIMD4 properties are provided for convenience.

/// Particle state on GPU. 112 bytes (28 x 4-byte fields).
/// Layout: matches Metal `Particle` struct byte-for-byte.
struct Particle {
    // Position (float3 in Metal = 16 bytes: 3 floats + 1 pad)
    var posX: Float = 0;  var posY: Float = 0;  var posZ: Float = 0;  var _pad0: Float = 0

    // Direction (float3 in Metal = 16 bytes: 3 floats + 1 pad)
    var dirX: Float = 0;  var dirY: Float = 0;  var dirZ: Float = 1;  var _pad1: Float = 0

    var energyGroup: UInt32 = 0
    var weight: Float = 1.0
    var cellIndex: UInt32 = 0
    var alive: UInt32 = 1

    var event: UInt32 = 0               // ParticleEvent raw value
    var rngCounter: UInt32 = 0
    var rngKey: UInt32 = 0
    var distanceToCollision: Float = 0

    var distanceToBoundary: Float = 0
    var boundarySurface: Int32 = -1
    var distanceTraveled: Float = 0
    var xsTotal: Float = 0

    var xsScatter: Float = 0
    var xsFission: Float = 0
    var xsNuFission: Float = 0
    var xsAbsorption: Float = 0

    var fissionFlag: UInt32 = 0
    var materialIndex: UInt32 = 0
    var _pad2: UInt32 = 0
    var _pad3: UInt32 = 0

    // SIMD3 convenience accessors
    var position: SIMD3<Float> {
        get { SIMD3<Float>(posX, posY, posZ) }
        set { posX = newValue.x; posY = newValue.y; posZ = newValue.z }
    }

    var direction: SIMD3<Float> {
        get { SIMD3<Float>(dirX, dirY, dirZ) }
        set { dirX = newValue.x; dirY = newValue.y; dirZ = newValue.z }
    }

    init(position: SIMD3<Float> = .zero,
         direction: SIMD3<Float> = SIMD3<Float>(0, 0, 1)) {
        self.posX = position.x; self.posY = position.y; self.posZ = position.z
        self.dirX = direction.x; self.dirY = direction.y; self.dirZ = direction.z
    }
}

/// Surface definition for GPU. 32 bytes (8 x 4-byte fields).
/// Coefficients are overloaded by surface type:
///   planeX/Y/Z: position = coefficients.x
///   cylinderZ:  x0=coeff.x, y0=coeff.y, R=coeff.z
///   sphere:     x0=coeff.x, y0=coeff.y, z0=coeff.z, R=coeff.w
struct GPUSurface {
    var type: UInt32 = 0                // SurfaceType raw value
    var boundaryCondition: UInt32 = 0   // BoundaryCondition raw value
    var coeffX: Float = 0;  var coeffY: Float = 0;  var coeffZ: Float = 0;  var coeffW: Float = 0
    var _pad0: UInt32 = 0
    var _pad1: UInt32 = 0

    // SIMD4 convenience accessor
    var coefficients: SIMD4<Float> {
        get { SIMD4<Float>(coeffX, coeffY, coeffZ, coeffW) }
        set { coeffX = newValue.x; coeffY = newValue.y; coeffZ = newValue.z; coeffW = newValue.w }
    }
}

/// Cell definition for GPU. 16 bytes.
struct GPUCell {
    var materialIndex: UInt32 = 0
    var numSurfaces: UInt32 = 0
    var surfaceOffset: UInt32 = 0       // Index into GPUCellSurface array
    var _pad: UInt32 = 0
}

/// Cell-surface association with sense (+1 or -1). 8 bytes.
struct GPUCellSurface {
    var surfaceIndex: UInt32 = 0
    var sense: Int32 = 1                // +1 or -1
}

/// Lattice definition for structured geometry. 48 bytes (12 x 4-byte fields).
struct GPULattice {
    var originX: Float = 0;  var originY: Float = 0;  var originZ: Float = 0;  var _pad0: Float = 0
    var pitchX: Float = 0;   var pitchY: Float = 0;   var pitchZ: Float = 0;   var _pad1: Float = 0
    var dimX: UInt32 = 0;    var dimY: UInt32 = 0;    var dimZ: UInt32 = 0;    var _pad2: UInt32 = 0

    // SIMD convenience accessors
    var origin: SIMD3<Float> {
        get { SIMD3<Float>(originX, originY, originZ) }
        set { originX = newValue.x; originY = newValue.y; originZ = newValue.z }
    }

    var pitch: SIMD3<Float> {
        get { SIMD3<Float>(pitchX, pitchY, pitchZ) }
        set { pitchX = newValue.x; pitchY = newValue.y; pitchZ = newValue.z }
    }

    var dimensions: SIMD3<UInt32> {
        get { SIMD3<UInt32>(dimX, dimY, dimZ) }
        set { dimX = newValue.x; dimY = newValue.y; dimZ = newValue.z }
    }
}

/// Multi-group cross section data for a single material. 308 bytes.
/// Uses fixed-size tuples for C-compatible layout (7 groups, 7x7 scatter matrix).
struct MaterialXS {
    // Total cross section per group [7]
    var total: (Float, Float, Float, Float, Float, Float, Float)
        = (0, 0, 0, 0, 0, 0, 0)

    // Scatter matrix [7x7] row-major: scatter[from * 7 + to]
    var scatter: (
        Float, Float, Float, Float, Float, Float, Float,
        Float, Float, Float, Float, Float, Float, Float,
        Float, Float, Float, Float, Float, Float, Float,
        Float, Float, Float, Float, Float, Float, Float,
        Float, Float, Float, Float, Float, Float, Float,
        Float, Float, Float, Float, Float, Float, Float,
        Float, Float, Float, Float, Float, Float, Float
    ) = (
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0
    )

    // Fission cross section per group [7]
    var fission: (Float, Float, Float, Float, Float, Float, Float)
        = (0, 0, 0, 0, 0, 0, 0)

    // Nu * fission cross section per group [7]
    var nuFission: (Float, Float, Float, Float, Float, Float, Float)
        = (0, 0, 0, 0, 0, 0, 0)

    // Fission spectrum (chi) per group [7]
    var chi: (Float, Float, Float, Float, Float, Float, Float)
        = (0, 0, 0, 0, 0, 0, 0)
}

/// Simulation parameters passed to GPU kernels. 32 bytes.
struct SimulationParams {
    var numParticles: UInt32 = 100_000
    var numBatches: UInt32 = 100
    var numInactive: UInt32 = 50
    var numGroups: UInt32 = 7
    var numCells: UInt32 = 0
    var numSurfaces: UInt32 = 0
    var numMaterials: UInt32 = 0
    var kEff: Float = 1.0
}

// MARK: - Helper Extensions

extension MaterialXS {
    /// Access total cross section by group index.
    func totalXS(group: Int) -> Float {
        precondition(group >= 0 && group < MAX_ENERGY_GROUPS)
        return withUnsafeBytes(of: total) { ptr in
            ptr.assumingMemoryBound(to: Float.self)[group]
        }
    }

    /// Access scatter matrix element.
    func scatterXS(from fromGroup: Int, to toGroup: Int) -> Float {
        precondition(fromGroup >= 0 && fromGroup < MAX_ENERGY_GROUPS)
        precondition(toGroup >= 0 && toGroup < MAX_ENERGY_GROUPS)
        return withUnsafeBytes(of: scatter) { ptr in
            ptr.assumingMemoryBound(to: Float.self)[fromGroup * MAX_ENERGY_GROUPS + toGroup]
        }
    }

    /// Access fission cross section by group index.
    func fissionXS(group: Int) -> Float {
        precondition(group >= 0 && group < MAX_ENERGY_GROUPS)
        return withUnsafeBytes(of: fission) { ptr in
            ptr.assumingMemoryBound(to: Float.self)[group]
        }
    }

    /// Access nu*fission cross section by group index.
    func nuFissionXS(group: Int) -> Float {
        precondition(group >= 0 && group < MAX_ENERGY_GROUPS)
        return withUnsafeBytes(of: nuFission) { ptr in
            ptr.assumingMemoryBound(to: Float.self)[group]
        }
    }

    /// Access chi (fission spectrum) by group index.
    func chiValue(group: Int) -> Float {
        precondition(group >= 0 && group < MAX_ENERGY_GROUPS)
        return withUnsafeBytes(of: chi) { ptr in
            ptr.assumingMemoryBound(to: Float.self)[group]
        }
    }

    /// Set total cross section by group index.
    mutating func setTotalXS(group: Int, value: Float) {
        precondition(group >= 0 && group < MAX_ENERGY_GROUPS)
        withUnsafeMutableBytes(of: &total) { ptr in
            ptr.assumingMemoryBound(to: Float.self)[group] = value
        }
    }

    /// Set scatter matrix element.
    mutating func setScatterXS(from fromGroup: Int, to toGroup: Int, value: Float) {
        precondition(fromGroup >= 0 && fromGroup < MAX_ENERGY_GROUPS)
        precondition(toGroup >= 0 && toGroup < MAX_ENERGY_GROUPS)
        withUnsafeMutableBytes(of: &scatter) { ptr in
            ptr.assumingMemoryBound(to: Float.self)[fromGroup * MAX_ENERGY_GROUPS + toGroup] = value
        }
    }

    /// Set fission cross section by group index.
    mutating func setFissionXS(group: Int, value: Float) {
        precondition(group >= 0 && group < MAX_ENERGY_GROUPS)
        withUnsafeMutableBytes(of: &fission) { ptr in
            ptr.assumingMemoryBound(to: Float.self)[group] = value
        }
    }

    /// Set nu*fission cross section by group index.
    mutating func setNuFissionXS(group: Int, value: Float) {
        precondition(group >= 0 && group < MAX_ENERGY_GROUPS)
        withUnsafeMutableBytes(of: &nuFission) { ptr in
            ptr.assumingMemoryBound(to: Float.self)[group] = value
        }
    }

    /// Set chi (fission spectrum) by group index.
    mutating func setChiValue(group: Int, value: Float) {
        precondition(group >= 0 && group < MAX_ENERGY_GROUPS)
        withUnsafeMutableBytes(of: &chi) { ptr in
            ptr.assumingMemoryBound(to: Float.self)[group] = value
        }
    }
}

// MARK: - Buffer Helpers

extension Array {
    /// Create an MTLBuffer from this array using shared storage.
    func toMTLBuffer(device: MTLDevice) -> MTLBuffer? {
        guard !isEmpty else { return nil }
        return withUnsafeBytes { ptr in
            device.makeBuffer(bytes: ptr.baseAddress!, length: ptr.count, options: .storageModeShared)
        }
    }
}
