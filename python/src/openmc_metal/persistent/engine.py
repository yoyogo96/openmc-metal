"""PyObjC-based GPU engine for persistent kernel transport.

Uses PyObjC Metal bindings (already installed) to dispatch the
history-based persistent transport kernel. One kernel dispatch
per batch instead of ~600 in the event-based architecture.
"""

import ctypes
import numpy as np
import Metal as MTL


class PersistentEngine:
    """GPU engine for history-based persistent transport kernel."""

    def __init__(self):
        """Initialize Metal device and compile persistent shader."""
        from .shader import PERSISTENT_SHADER_SOURCE

        self._device = MTL.MTLCreateSystemDefaultDevice()
        if self._device is None:
            raise RuntimeError("No Metal-compatible GPU found")

        self._queue = self._device.newCommandQueue()

        options = MTL.MTLCompileOptions.alloc().init()
        library, error = self._device.newLibraryWithSource_options_error_(
            PERSISTENT_SHADER_SOURCE, options, None
        )
        if library is None:
            err_msg = error.localizedDescription() if error else "unknown"
            raise RuntimeError(f"Shader compilation failed: {err_msg}")

        func = library.newFunctionWithName_("transport_kernel")
        if func is None:
            raise RuntimeError("transport_kernel function not found in shader")

        self._pipeline, error = self._device.newComputePipelineStateWithFunction_error_(
            func, None
        )
        if self._pipeline is None:
            err_msg = error.localizedDescription() if error else "unknown"
            raise RuntimeError(f"Pipeline creation failed: {err_msg}")

    @property
    def gpu_name(self) -> str:
        return self._device.name()

    def make_buffer(self, size_bytes: int):
        """Create a shared-mode Metal buffer (zeroed)."""
        size_bytes = max(size_bytes, 4)
        buf = self._device.newBufferWithBytes_length_options_(
            bytes(size_bytes), size_bytes, MTL.MTLResourceStorageModeShared
        )
        if buf is None:
            raise RuntimeError(f"Failed to create buffer of {size_bytes} bytes")
        return buf

    def make_buffer_from_numpy(self, arr: np.ndarray):
        """Create a shared-mode Metal buffer from a numpy array."""
        data = arr.tobytes()
        buf = self._device.newBufferWithBytes_length_options_(
            data, len(data), MTL.MTLResourceStorageModeShared
        )
        if buf is None:
            raise RuntimeError("Failed to create buffer from numpy array")
        return buf

    def write_buffer(self, buf, arr: np.ndarray):
        """Write numpy array data into an existing Metal buffer (shared mode).

        Uses memoryview slice assignment for direct write into GPU-visible memory.
        Buffer must be large enough to hold arr.nbytes.
        """
        data = arr.tobytes()
        n = len(data)
        if n > buf.length():
            raise RuntimeError(f"Array ({n} bytes) exceeds buffer ({buf.length()} bytes)")
        mv = buf.contents().as_buffer(buf.length())
        mv[:n] = data

    def zero_buffer(self, buf):
        """Zero out all bytes of a Metal buffer."""
        n = buf.length()
        mv = buf.contents().as_buffer(n)
        mv[:] = bytes(n)

    def read_buffer(self, buf, dtype=np.float32, count=None):
        """Read numpy array from Metal buffer."""
        mv = buf.contents().as_buffer(buf.length())
        if count is None:
            count = buf.length() // np.dtype(dtype).itemsize
        return np.frombuffer(mv, dtype=dtype, count=count).copy()

    def dispatch_transport(self, num_particles: int, buffers: list):
        """Dispatch persistent transport kernel with given buffers.

        Args:
            num_particles: Number of particle threads to launch.
            buffers: List of Metal buffers in order [0..25].
        """
        cmd = self._queue.commandBuffer()
        encoder = cmd.computeCommandEncoder()
        encoder.setComputePipelineState_(self._pipeline)

        for i, buf in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buf, 0, i)

        max_threads = self._pipeline.maxTotalThreadsPerThreadgroup()
        tg_width = min(max_threads, num_particles)

        grid = MTL.MTLSizeMake(num_particles, 1, 1)
        threadgroup = MTL.MTLSizeMake(tg_width, 1, 1)

        encoder.dispatchThreads_threadsPerThreadgroup_(grid, threadgroup)
        encoder.endEncoding()

        cmd.commit()
        cmd.waitUntilCompleted()

        status = cmd.status()
        if status == MTL.MTLCommandBufferStatusError:
            error = cmd.error()
            err_msg = error.localizedDescription() if error else "unknown"
            raise RuntimeError(f"GPU kernel error: {err_msg}")
