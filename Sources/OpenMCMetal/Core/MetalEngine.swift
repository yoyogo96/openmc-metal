import Metal
import Foundation

// MARK: - Errors

enum MetalError: Error, CustomStringConvertible {
    case noDevice
    case noCommandQueue
    case noLibrary
    case functionNotFound(String)
    case bufferCreationFailed

    var description: String {
        switch self {
        case .noDevice:
            return "No Metal-compatible GPU device found"
        case .noCommandQueue:
            return "Failed to create Metal command queue"
        case .noLibrary:
            return "Failed to load compiled Metal shader library"
        case .functionNotFound(let name):
            return "Metal function '\(name)' not found in library"
        case .bufferCreationFailed:
            return "Failed to create Metal buffer"
        }
    }
}

// MARK: - MetalEngine

/// Central Metal device and pipeline manager.
/// Owns the device, command queue, and shader library.
/// Provides helpers for buffer creation and kernel dispatch.
class MetalEngine {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let library: MTLLibrary
    private var pipelineCache: [String: MTLComputePipelineState] = [:]

    /// Initialize Metal engine with the system default GPU.
    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalError.noDevice
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MetalError.noCommandQueue
        }
        self.commandQueue = queue

        // Compile shaders from embedded source (supports CLI tools without bundles)
        let options = MTLCompileOptions()
        options.languageVersion = .version3_0
        self.library = try device.makeLibrary(source: ShaderSource.allShaders, options: options)
    }

    // MARK: - Device Info

    /// Print GPU device capabilities.
    func printDeviceInfo() {
        print("=== Metal GPU Device Info ===")
        print("  Device:                   \(device.name)")
        print("  Max Threadgroup Size:     \(device.maxThreadsPerThreadgroup)")
        print("  Unified Memory:           \(device.hasUnifiedMemory)")
        let maxWorkingSetMB = device.recommendedMaxWorkingSetSize / (1024 * 1024)
        print("  Recommended Working Set:  \(maxWorkingSetMB) MB")
        print("  Max Buffer Length:        \(device.maxBufferLength / (1024 * 1024)) MB")
        print("  Registry ID:             \(device.registryID)")
        print("=============================")
    }

    // MARK: - Pipeline Management

    /// Get or create a compute pipeline for the named Metal function.
    func makePipeline(functionName: String) throws -> MTLComputePipelineState {
        if let cached = pipelineCache[functionName] {
            return cached
        }

        guard let function = library.makeFunction(name: functionName) else {
            throw MetalError.functionNotFound(functionName)
        }

        let pipeline = try device.makeComputePipelineState(function: function)
        pipelineCache[functionName] = pipeline
        return pipeline
    }

    // MARK: - Buffer Creation

    /// Create a shared-mode buffer from an array of values.
    func makeBuffer<T>(_ data: [T]) -> MTLBuffer {
        guard !data.isEmpty else {
            // Return a minimal buffer for empty arrays
            return device.makeBuffer(length: MemoryLayout<T>.stride, options: .storageModeShared)!
        }
        return data.withUnsafeBytes { ptr in
            device.makeBuffer(
                bytes: ptr.baseAddress!,
                length: ptr.count,
                options: .storageModeShared
            )!
        }
    }

    /// Create a shared-mode buffer from a single value.
    func makeBuffer<T>(value: T) -> MTLBuffer {
        var val = value
        return withUnsafeBytes(of: &val) { ptr in
            device.makeBuffer(
                bytes: ptr.baseAddress!,
                length: ptr.count,
                options: .storageModeShared
            )!
        }
    }

    /// Create an uninitialized shared-mode buffer of the given byte length.
    func makeBuffer(length: Int) -> MTLBuffer {
        guard length > 0 else {
            return device.makeBuffer(length: 1, options: .storageModeShared)!
        }
        return device.makeBuffer(length: length, options: .storageModeShared)!
    }

    // MARK: - Dispatch

    /// Encode a compute dispatch into a command buffer.
    /// If no command buffer is provided, a new one is created.
    /// Returns the command buffer (not yet committed).
    @discardableResult
    func dispatch(
        pipeline: MTLComputePipelineState,
        buffers: [MTLBuffer],
        gridSize: Int,
        commandBuffer: MTLCommandBuffer? = nil
    ) -> MTLCommandBuffer {
        let cmdBuffer = commandBuffer ?? commandQueue.makeCommandBuffer()!
        let encoder = cmdBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)

        for (i, buffer) in buffers.enumerated() {
            encoder.setBuffer(buffer, offset: 0, index: i)
        }

        let maxThreads = pipeline.maxTotalThreadsPerThreadgroup
        let threadgroupWidth = min(maxThreads, gridSize)
        let threadgroupSize = MTLSize(width: threadgroupWidth, height: 1, depth: 1)
        let gridSizeMTL = MTLSize(width: gridSize, height: 1, depth: 1)

        encoder.dispatchThreads(gridSizeMTL, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()

        return cmdBuffer
    }

    /// Dispatch a kernel and wait for completion. Returns GPU execution time in seconds.
    @discardableResult
    func dispatchAndWait(
        pipeline: MTLComputePipelineState,
        buffers: [MTLBuffer],
        gridSize: Int
    ) -> Double {
        let cmdBuffer = dispatch(pipeline: pipeline, buffers: buffers, gridSize: gridSize)
        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()

        if cmdBuffer.status == .error {
            print("GPU Error: \(cmdBuffer.error?.localizedDescription ?? "unknown")")
            return -1.0
        }

        return cmdBuffer.gpuEndTime - cmdBuffer.gpuStartTime
    }

    /// Dispatch multiple kernels in a single command buffer (batched submission).
    @discardableResult
    func dispatchBatch(
        operations: [(pipeline: MTLComputePipelineState, buffers: [MTLBuffer], gridSize: Int)]
    ) -> Double {
        let cmdBuffer = commandQueue.makeCommandBuffer()!

        for op in operations {
            let encoder = cmdBuffer.makeComputeCommandEncoder()!
            encoder.setComputePipelineState(op.pipeline)
            for (i, buffer) in op.buffers.enumerated() {
                encoder.setBuffer(buffer, offset: 0, index: i)
            }
            let maxThreads = op.pipeline.maxTotalThreadsPerThreadgroup
            let threadgroupWidth = min(maxThreads, op.gridSize)
            let threadgroupSize = MTLSize(width: threadgroupWidth, height: 1, depth: 1)
            let gridSizeMTL = MTLSize(width: op.gridSize, height: 1, depth: 1)
            encoder.dispatchThreads(gridSizeMTL, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }

        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()

        if cmdBuffer.status == .error {
            print("GPU Error: \(cmdBuffer.error?.localizedDescription ?? "unknown")")
            return -1.0
        }

        return cmdBuffer.gpuEndTime - cmdBuffer.gpuStartTime
    }
}
