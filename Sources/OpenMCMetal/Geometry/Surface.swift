import Metal
import simd

/// Swift-side surface builder that wraps the GPU-side GPUSurface struct.
/// GPUSurface, SurfaceType, and BoundaryCondition are defined in Types.swift.
struct Surface {
    var gpuSurface: GPUSurface
    var id: Int

    // MARK: - Factory Methods

    /// Plane perpendicular to the X axis at x = x0.
    static func planeX(_ x0: Float, bc: BoundaryCondition = .vacuum) -> Surface {
        var s = GPUSurface()
        s.type = SurfaceType.planeX.rawValue
        s.coefficients = SIMD4<Float>(x0, 0, 0, 0)
        s.boundaryCondition = bc.rawValue
        return Surface(gpuSurface: s, id: 0)
    }

    /// Plane perpendicular to the Y axis at y = y0.
    static func planeY(_ y0: Float, bc: BoundaryCondition = .vacuum) -> Surface {
        var s = GPUSurface()
        s.type = SurfaceType.planeY.rawValue
        s.coefficients = SIMD4<Float>(y0, 0, 0, 0)
        s.boundaryCondition = bc.rawValue
        return Surface(gpuSurface: s, id: 0)
    }

    /// Plane perpendicular to the Z axis at z = z0.
    static func planeZ(_ z0: Float, bc: BoundaryCondition = .vacuum) -> Surface {
        var s = GPUSurface()
        s.type = SurfaceType.planeZ.rawValue
        s.coefficients = SIMD4<Float>(z0, 0, 0, 0)
        s.boundaryCondition = bc.rawValue
        return Surface(gpuSurface: s, id: 0)
    }

    /// Infinite cylinder aligned with the Z axis, centered at (x0, y0) with given radius.
    /// coefficients: (x0, y0, radius, unused)
    static func cylinderZ(x0: Float = 0, y0: Float = 0, radius: Float, bc: BoundaryCondition = .vacuum) -> Surface {
        var s = GPUSurface()
        s.type = SurfaceType.cylinderZ.rawValue
        s.coefficients = SIMD4<Float>(x0, y0, radius, 0)
        s.boundaryCondition = bc.rawValue
        return Surface(gpuSurface: s, id: 0)
    }

    /// Sphere centered at (x0, y0, z0) with given radius.
    /// coefficients: (x0, y0, z0, radius)
    static func sphere(x0: Float = 0, y0: Float = 0, z0: Float = 0, radius: Float, bc: BoundaryCondition = .vacuum) -> Surface {
        var s = GPUSurface()
        s.type = SurfaceType.sphere.rawValue
        s.coefficients = SIMD4<Float>(x0, y0, z0, radius)
        s.boundaryCondition = bc.rawValue
        return Surface(gpuSurface: s, id: 0)
    }
}
