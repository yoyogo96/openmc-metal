// TransportKernels.swift
// OpenMC-Metal
//
// Swift wrapper for the GPU transport kernels. Manages compute pipeline states
// and provides convenience dispatch methods for each stage of the transport loop:
//
//   init_particles -> xs_lookup -> distance_to_collision -> move_particle -> [collide/tally]
//                        ^                                       |
//                        |_______________________________________|
//                          (boundary crossing triggers re-lookup)

import Metal

// MARK: - TransportKernels

/// Manages Metal compute pipelines for all transport kernels and provides
/// typed dispatch methods that encode the correct buffer bindings.
class TransportKernels {
    let engine: MetalEngine

    // Pipeline states (one per kernel function)
    let initParticlesPipeline: MTLComputePipelineState
    let xsLookupPipeline: MTLComputePipelineState
    let distanceToCollisionPipeline: MTLComputePipelineState
    let moveParticlePipeline: MTLComputePipelineState

    /// Initialize all transport kernel pipelines from the Metal library.
    /// Throws if any kernel function is missing from the compiled shader library.
    init(engine: MetalEngine) throws {
        self.engine = engine
        self.initParticlesPipeline = try engine.makePipeline(functionName: "init_particles")
        self.xsLookupPipeline = try engine.makePipeline(functionName: "xs_lookup")
        self.distanceToCollisionPipeline = try engine.makePipeline(functionName: "distance_to_collision")
        self.moveParticlePipeline = try engine.makePipeline(functionName: "move_particle")
    }

    // MARK: - Dispatch: init_particles

    /// Encode particle initialization into `commandBuffer`.
    ///
    /// Buffer bindings:
    ///   [0] particles  (read-write)
    ///   [1] params     (read-only)
    func dispatchInitParticles(particles: MTLBuffer,
                               params: MTLBuffer,
                               count: Int,
                               commandBuffer: MTLCommandBuffer) {
        engine.dispatch(
            pipeline: initParticlesPipeline,
            buffers: [particles, params],
            gridSize: count,
            commandBuffer: commandBuffer
        )
    }

    // MARK: - Dispatch: xs_lookup

    /// Encode cross-section lookup into `commandBuffer`.
    ///
    /// Buffer bindings:
    ///   [0] particles  (read-write)
    ///   [1] materials  (read-only, flat float buffer: 77 floats per material)
    ///   [2] cells      (read-only)
    ///   [3] params     (read-only)
    func dispatchXSLookup(particles: MTLBuffer,
                          materials: MTLBuffer,
                          cells: MTLBuffer,
                          params: MTLBuffer,
                          count: Int,
                          commandBuffer: MTLCommandBuffer) {
        engine.dispatch(
            pipeline: xsLookupPipeline,
            buffers: [particles, materials, cells, params],
            gridSize: count,
            commandBuffer: commandBuffer
        )
    }

    // MARK: - Dispatch: distance_to_collision

    /// Encode distance-to-collision sampling into `commandBuffer`.
    ///
    /// Buffer bindings:
    ///   [0] particles  (read-write)
    ///   [1] params     (read-only)
    func dispatchDistanceToCollision(particles: MTLBuffer,
                                     params: MTLBuffer,
                                     count: Int,
                                     commandBuffer: MTLCommandBuffer) {
        engine.dispatch(
            pipeline: distanceToCollisionPipeline,
            buffers: [particles, params],
            gridSize: count,
            commandBuffer: commandBuffer
        )
    }

    // MARK: - Dispatch: move_particle

    /// Encode particle movement and boundary crossing into `commandBuffer`.
    ///
    /// Buffer bindings:
    ///   [0] particles     (read-write)
    ///   [1] surfaces      (read-only)
    ///   [2] cells         (read-only)
    ///   [3] cellSurfaces  (read-only)
    ///   [4] params        (read-only)
    ///   [5] lostCount     (read-write, atomic_uint counter)
    func dispatchMove(particles: MTLBuffer,
                      surfaces: MTLBuffer,
                      cells: MTLBuffer,
                      cellSurfaces: MTLBuffer,
                      params: MTLBuffer,
                      lostCount: MTLBuffer,
                      count: Int,
                      commandBuffer: MTLCommandBuffer) {
        engine.dispatch(
            pipeline: moveParticlePipeline,
            buffers: [particles, surfaces, cells, cellSurfaces, params, lostCount],
            gridSize: count,
            commandBuffer: commandBuffer
        )
    }

    // MARK: - Convenience: Full Transport Step

    /// Encode a complete transport step (XS lookup -> distance sampling -> move)
    /// into a single command buffer. This is the inner loop of the transport cycle.
    ///
    /// After this step, particles will be in one of:
    ///   - EVENT_COLLIDE (ready for collision physics)
    ///   - EVENT_XS_LOOKUP (crossed boundary, need another iteration)
    ///   - EVENT_DEAD (escaped or lost)
    func dispatchTransportStep(particles: MTLBuffer,
                               materials: MTLBuffer,
                               surfaces: MTLBuffer,
                               cells: MTLBuffer,
                               cellSurfaces: MTLBuffer,
                               params: MTLBuffer,
                               lostCount: MTLBuffer,
                               count: Int,
                               commandBuffer: MTLCommandBuffer) {
        dispatchXSLookup(particles: particles, materials: materials,
                         cells: cells, params: params,
                         count: count, commandBuffer: commandBuffer)

        dispatchDistanceToCollision(particles: particles, params: params,
                                    count: count, commandBuffer: commandBuffer)

        dispatchMove(particles: particles, surfaces: surfaces,
                     cells: cells, cellSurfaces: cellSurfaces,
                     params: params, lostCount: lostCount,
                     count: count, commandBuffer: commandBuffer)
    }
}
