"""Metal GPU compute engine using PyObjC bindings."""

import struct
import numpy as np

import Metal as MTL


class MetalEngine:
    """Central Metal device and pipeline manager.

    Wraps MTLDevice, MTLCommandQueue, and MTLLibrary.
    Provides helpers for buffer creation and kernel dispatch.
    """

    def __init__(self, shader_source: str | None = None):
        """Initialize Metal engine with the system default GPU."""
        from .shaders import load_shader_source

        self.device = MTL.MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("No Metal-compatible GPU device found")

        self.command_queue = self.device.newCommandQueue()
        if self.command_queue is None:
            raise RuntimeError("Failed to create Metal command queue")

        if shader_source is None:
            shader_source = load_shader_source()

        options = MTL.MTLCompileOptions.alloc().init()
        library, error = self.device.newLibraryWithSource_options_error_(
            shader_source, options, None
        )
        if library is None:
            err_msg = error.localizedDescription() if error else "unknown error"
            raise RuntimeError(f"Failed to compile Metal shaders: {err_msg}")
        self.library = library

        self._pipeline_cache: dict[str, object] = {}

    @property
    def gpu_name(self) -> str:
        return self.device.name()

    @property
    def has_unified_memory(self) -> bool:
        return self.device.hasUnifiedMemory()

    @property
    def max_threadgroup_size(self) -> int:
        size = self.device.maxThreadsPerThreadgroup()
        return size.width

    def print_device_info(self):
        """Print GPU device capabilities."""
        print("=== Metal GPU Device Info ===")
        print(f"  Device:                  {self.device.name()}")
        mts = self.device.maxThreadsPerThreadgroup()
        print(f"  Max Threadgroup Size:    {mts.width}x{mts.height}x{mts.depth}")
        print(f"  Unified Memory:          {self.device.hasUnifiedMemory()}")
        max_ws_mb = self.device.recommendedMaxWorkingSetSize() // (1024 * 1024)
        print(f"  Recommended Working Set: {max_ws_mb} MB")
        max_buf_mb = self.device.maxBufferLength() // (1024 * 1024)
        print(f"  Max Buffer Length:       {max_buf_mb} MB")
        print("=============================")

    # --- Pipeline Management ---

    def make_pipeline(self, function_name: str):
        """Get or create a compute pipeline for the named Metal function."""
        if function_name in self._pipeline_cache:
            return self._pipeline_cache[function_name]

        func = self.library.newFunctionWithName_(function_name)
        if func is None:
            raise RuntimeError(f"Metal function '{function_name}' not found in library")

        pipeline, error = self.device.newComputePipelineStateWithFunction_error_(
            func, None
        )
        if pipeline is None:
            err_msg = error.localizedDescription() if error else "unknown"
            raise RuntimeError(f"Pipeline creation failed for '{function_name}': {err_msg}")

        self._pipeline_cache[function_name] = pipeline
        return pipeline

    # --- Buffer Helpers ---

    @staticmethod
    def buffer_view(buf) -> memoryview:
        """Get a writable memoryview of an MTLBuffer's contents.

        Uses PyObjC's objc.varlist.as_buffer() to get a memoryview
        that supports struct.pack_into, numpy, and slice assignment.
        """
        return buf.contents().as_buffer(buf.length())

    # --- Buffer Creation ---

    def make_buffer_from_data(self, data: bytes):
        """Create a shared-mode MTLBuffer from bytes."""
        buf = self.device.newBufferWithBytes_length_options_(
            data, len(data), MTL.MTLResourceStorageModeShared
        )
        if buf is None:
            raise RuntimeError("Failed to create Metal buffer from data")
        return buf

    def make_buffer(self, length: int):
        """Create a zeroed shared-mode MTLBuffer of the given byte length."""
        length = max(length, 4)
        # Create from zero bytes (avoids ctypes interop issues with PyObjC)
        buf = self.device.newBufferWithBytes_length_options_(
            bytes(length), length, MTL.MTLResourceStorageModeShared
        )
        if buf is None:
            raise RuntimeError(f"Failed to create Metal buffer of length {length}")
        return buf

    def buffer_contents_as_array(self, buf, dtype=np.float32, count=None):
        """Get a numpy array from buffer contents."""
        mv = self.buffer_view(buf)
        if count is None:
            count = buf.length() // np.dtype(dtype).itemsize
        return np.frombuffer(mv, dtype=dtype, count=count).copy()

    def read_buffer_bytes(self, buf, length=None) -> bytes:
        """Read raw bytes from a buffer."""
        mv = self.buffer_view(buf)
        if length is None:
            length = buf.length()
        return bytes(mv[:length])

    def write_buffer_bytes(self, buf, data: bytes, offset: int = 0):
        """Write raw bytes to a buffer at given offset."""
        mv = self.buffer_view(buf)
        mv[offset:offset + len(data)] = data

    def zero_buffer(self, buf, length: int | None = None):
        """Zero out a buffer's contents."""
        mv = self.buffer_view(buf)
        n = length if length is not None else buf.length()
        mv[:n] = bytes(n)

    # --- Dispatch ---

    def dispatch(self, pipeline, buffers: list, grid_size: int,
                 command_buffer=None):
        """Encode a compute dispatch into a command buffer."""
        if command_buffer is None:
            command_buffer = self.command_queue.commandBuffer()

        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)

        for i, buf in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buf, 0, i)

        max_threads = pipeline.maxTotalThreadsPerThreadgroup()
        threadgroup_width = min(max_threads, grid_size)

        grid_size_mtl = MTL.MTLSizeMake(grid_size, 1, 1)
        threadgroup_size = MTL.MTLSizeMake(threadgroup_width, 1, 1)

        encoder.dispatchThreads_threadsPerThreadgroup_(grid_size_mtl, threadgroup_size)
        encoder.endEncoding()

        return command_buffer

    def dispatch_and_wait(self, pipeline, buffers: list, grid_size: int) -> float:
        """Dispatch a kernel and wait for completion. Returns GPU time in seconds."""
        cmd = self.dispatch(pipeline, buffers, grid_size)
        cmd.commit()
        cmd.waitUntilCompleted()

        status = cmd.status()
        if status == MTL.MTLCommandBufferStatusError:
            error = cmd.error()
            err_msg = error.localizedDescription() if error else "unknown"
            print(f"GPU Error: {err_msg}")
            return -1.0

        return cmd.GPUEndTime() - cmd.GPUStartTime()
