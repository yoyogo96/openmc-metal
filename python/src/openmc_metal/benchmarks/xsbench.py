"""XSBench GPU microbenchmark - cross-section lookup throughput."""

import struct as struct_mod
import random
import time
import math

from ..types import XSBENCHPARAMS_FORMAT
from ..cross_sections import c5g7_materials_buffer


class XSBenchmark:
    """GPU cross-section lookup microbenchmark."""

    def __init__(self, engine):
        self.engine = engine

    def run(self, num_lookups: int = 10_000_000,
            num_materials: int = 7, num_groups: int = 7,
            num_runs: int = 5) -> dict:
        """Run the XSBench microbenchmark.

        Returns: dict with timing and throughput results.
        """
        print("=" * 50)
        print("   XSBench - Metal GPU Microbenchmark")
        print("=" * 50)
        print(f"\n  Lookups:    {num_lookups:,}")
        print(f"  Materials:  {num_materials}")
        print(f"  Groups:     {num_groups}")
        print(f"  Runs:       {num_runs}\n")

        # Materials buffer
        mat_data = c5g7_materials_buffer()
        materials_buffer = self.engine.make_buffer_from_data(mat_data)

        # Random lookup indices: [matIdx, groupIdx, matIdx, groupIdx, ...]
        lookup_data = struct_mod.pack(
            f'<{num_lookups * 2}I',
            *[v for _ in range(num_lookups)
              for v in (random.randint(0, num_materials - 1),
                        random.randint(0, num_groups - 1))]
        )
        lookup_buffer = self.engine.make_buffer_from_data(lookup_data)

        # Output buffer
        output_buffer = self.engine.make_buffer(num_lookups * 4)

        # Params buffer
        params_data = struct_mod.pack(XSBENCHPARAMS_FORMAT,
                                       num_lookups, num_materials, num_groups, 0)
        params_buffer = self.engine.make_buffer_from_data(params_data)

        # Pipeline
        pipeline = self.engine.make_pipeline("xsbench_lookup")
        buffers = [lookup_buffer, materials_buffer, output_buffer, params_buffer]

        # Warmup
        print("Warming up...")
        self.engine.dispatch_and_wait(pipeline, buffers, num_lookups)

        # Timed runs
        times = []
        for run_idx in range(1, num_runs + 1):
            gpu_time = self.engine.dispatch_and_wait(pipeline, buffers, num_lookups)
            times.append(gpu_time)
            throughput = num_lookups / gpu_time / 1e6
            print(f"  Run {run_idx}: {gpu_time*1000:7.3f} ms  ({throughput:6.2f} M lookups/sec)")

        # Statistics
        mean_time = sum(times) / len(times)
        if len(times) > 1:
            variance = sum((t - mean_time) ** 2 for t in times) / (len(times) - 1)
        else:
            variance = 0.0
        std_dev = math.sqrt(variance)
        mean_throughput = num_lookups / mean_time / 1e6

        # Verification: check output has positive values
        mv = self.engine.buffer_view(output_buffer)
        positive = 0
        check_count = min(100, num_lookups)
        for i in range(check_count):
            val = struct_mod.unpack_from('<f', mv, i * 4)[0]
            if val > 0:
                positive += 1

        print(f"\nResults:")
        print(f"  Mean time:     {mean_time*1000:.3f} +/- {std_dev*1000:.3f} ms")
        print(f"  Throughput:    {mean_throughput:.2f} M lookups/sec")
        print(f"  GPU:           {self.engine.gpu_name}")
        print(f"  Verification:  {positive}/{check_count} spot checks positive")

        return {
            'num_lookups': num_lookups,
            'num_materials': num_materials,
            'num_groups': num_groups,
            'runs': [{'gpu_time_ms': t * 1000, 'throughput_mlookups_sec': num_lookups / t / 1e6} for t in times],
            'mean_time_ms': mean_time * 1000,
            'std_dev_ms': std_dev * 1000,
            'mean_throughput_mlookups_sec': mean_throughput,
            'verification_pass': positive == check_count,
        }
