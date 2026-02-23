import Metal
import Foundation

// MARK: - TallyManager

/// Manages GPU tally buffers and accumulates per-batch tally data on the CPU.
///
/// The GPU stores tally values as `atomic_uint` (float bits via CAS loop).
/// After each batch the manager reads those bits back, reinterprets them as
/// `Float`, and appends a `Double` copy to CPU-side batch arrays for statistics.
///
/// Tally array indexing: `[cellIndex * numGroups + energyGroup]`
final class TallyManager {
    let engine: MetalEngine
    let numCells: Int
    let numGroups: Int

    /// Total number of tally bins = numCells * numGroups.
    let tallySize: Int

    /// GPU buffer for flux tally (float bits stored as UInt32 for atomic ops).
    private(set) var fluxBuffer: MTLBuffer

    /// GPU buffer for fission-rate tally (float bits stored as UInt32).
    private(set) var fissionBuffer: MTLBuffer

    /// Compiled tally_score compute pipeline.
    let tallyPipeline: MTLComputePipelineState

    // CPU-side batch accumulation (Double precision to avoid FP32 round-off).
    /// Per-batch flux values: batchFlux[batch][cellIndex * numGroups + energyGroup]
    private(set) var batchFlux: [[Double]] = []
    /// Per-batch fission-rate values.
    private(set) var batchFission: [[Double]] = []

    // MARK: Init

    init(engine: MetalEngine, numCells: Int, numGroups: Int) throws {
        self.engine    = engine
        self.numCells  = numCells
        self.numGroups = numGroups
        self.tallySize = numCells * numGroups

        // GPU buffers: one UInt32 per bin (stores float bits).
        self.fluxBuffer    = engine.makeBuffer(length: tallySize * MemoryLayout<UInt32>.stride)
        self.fissionBuffer = engine.makeBuffer(length: tallySize * MemoryLayout<UInt32>.stride)

        self.tallyPipeline = try engine.makePipeline(functionName: "tally_score")
    }

    // MARK: Batch Lifecycle

    /// Zero GPU tally buffers before dispatching a new batch.
    func resetForBatch() {
        memset(fluxBuffer.contents(),    0, tallySize * MemoryLayout<UInt32>.stride)
        memset(fissionBuffer.contents(), 0, tallySize * MemoryLayout<UInt32>.stride)
    }

    /// Read GPU tallies into CPU Double arrays and append to the batch history.
    /// Call this after the batch command buffer has completed (GPU idle).
    func collectBatch() {
        let fluxPtr    = fluxBuffer.contents().bindMemory(to: UInt32.self,
                                                          capacity: tallySize)
        let fissionPtr = fissionBuffer.contents().bindMemory(to: UInt32.self,
                                                              capacity: tallySize)

        var flux    = [Double](repeating: 0, count: tallySize)
        var fission = [Double](repeating: 0, count: tallySize)

        for i in 0..<tallySize {
            // Reinterpret stored UInt32 bits as Float, then widen to Double.
            flux[i]    = Double(Float(bitPattern: fluxPtr[i]))
            fission[i] = Double(Float(bitPattern: fissionPtr[i]))
        }

        batchFlux.append(flux)
        batchFission.append(fission)
    }

    /// Discard all accumulated batch data (e.g., end of inactive cycles).
    func clearBatchHistory() {
        batchFlux    = []
        batchFission = []
    }

    // MARK: Dispatch

    /// Encode `tally_score` kernel into `commandBuffer`.
    /// Buffers: [0] particles, [1] tallyFlux, [2] tallyFission, [3] params.
    func dispatch(
        particles: MTLBuffer,
        params: MTLBuffer,
        count: Int,
        commandBuffer: MTLCommandBuffer
    ) {
        guard count > 0 else { return }

        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(tallyPipeline)
        encoder.setBuffer(particles,     offset: 0, index: 0)
        encoder.setBuffer(fluxBuffer,    offset: 0, index: 1)
        encoder.setBuffer(fissionBuffer, offset: 0, index: 2)
        encoder.setBuffer(params,        offset: 0, index: 3)

        let maxThreads     = tallyPipeline.maxTotalThreadsPerThreadgroup
        let threadgroupW   = min(maxThreads, count)
        let threadgroupSize = MTLSize(width: threadgroupW, height: 1, depth: 1)
        let gridSize        = MTLSize(width: count,         height: 1, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }

    // MARK: Convenience Accessors

    /// Mean flux per bin across all collected batches.
    var meanFlux: [Double] {
        Statistics.batchMeans(batchValues: batchFlux, size: tallySize)
    }

    /// Mean fission rate per bin across all collected batches.
    var meanFission: [Double] {
        Statistics.batchMeans(batchValues: batchFission, size: tallySize)
    }
}
