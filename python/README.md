# OpenMC-Metal: Monte Carlo Neutron Transport on Apple Metal GPU

A GPU-accelerated Monte Carlo neutron transport code targeting Apple Silicon using Metal Shading Language. OpenMC-Metal implements both event-based and history-based persistent kernel transport architectures for k-eigenvalue criticality calculations using C5G7 7-group cross sections, achieving **1.76M histories/sec with the persistent kernel**.

## Overview

OpenMC-Metal brings Monte Carlo neutron transport to Apple Silicon GPUs via the Metal compute pipeline. The implementation supports two GPU transport architectures: an event-based algorithm with **three fused GPU compute kernels** per transport step, and a **history-based persistent kernel** that dispatches a single long-running shader processing complete neutron histories end-to-end. The persistent kernel achieves **1.76M histories/sec** on the M4 Max assembly benchmark, a 15.7x improvement over the event-based design. Cross-section data follows the C5G7 7-group benchmark specification covering seven representative reactor materials.

The Python driver layer uses PyObjC to interface directly with the Metal API, avoiding the need for intermediate frameworks. Because Apple Silicon uses a unified memory architecture, neutron state buffers are shared between the CPU and GPU without explicit data transfers, which reduces latency and simplifies the host-device coordination logic.

## Features

- **GPU-accelerated multi-group Monte Carlo transport** using Metal compute shaders on Apple Silicon
- **3 fused compute kernels** (down from 5) with kernel fusion for XS lookup + distance sampling and collision + tally scoring
- **History-based persistent kernel** with single-dispatch transport, SoA memory layout, and zero-allocation batch loops
- **K-eigenvalue power iteration** with Shannon entropy convergence monitoring
- **C5G7 7-group benchmark cross sections** covering 7 materials (UO2, MOX 4.3%, MOX 7.0%, MOX 8.7%, fission chamber, guide tube, moderator)
- **Philox-2x32-10 counter-based RNG** with per-particle independent streams
- **Asynchronous GPU dispatch** — 5 steps per command buffer, sync every 25 steps
- **XSBench microbenchmark** for measuring cross-section lookup throughput in isolation
- **CPU baseline transport** for direct GPU speedup measurement under identical problem conditions

## Benchmark Results

Results on an Apple M4 Max (40 GPU cores) running C5G7 criticality benchmarks:

### C5G7 Pincell (100K particles/batch, 100 batches)

| Metric | Value |
|--------|-------|
| GPU | Apple M4 Max (40 GPU cores) |
| k-eff | 1.3256 +/- 0.0005 |
| Reference k-eff | 1.33007 |
| Delta | -448 pcm |
| GPU Throughput | 46,198 particles/sec |
| CPU Throughput (pure Python) | 3,907 particles/sec |
| CPU Throughput (Numba JIT, single-thread) | 231,819 particles/sec |
| GPU vs Numba CPU | 0.20x (CPU is 5x faster) |
| XSBench | 10,981 M lookups/sec |

### C5G7 17x17 Assembly (1M particles/batch, 150 batches)

| Metric | Value |
|--------|-------|
| k-eff | 1.2751 +/- 0.0001 |
| Throughput | 111,772 histories/sec |

### Persistent Kernel Results

| Metric | Pincell (1M) | Assembly (1M) |
|--------|-------------|---------------|
| k-eff | 1.3256 ± 0.0007 | 1.2762 ± 0.0003 |
| Throughput | 200,059 hist/sec | 1,762,612 hist/sec |
| vs Event-based | 4.3x | 15.7x |

### MC/DC Comparison (Morgan et al. 2025)

Direct comparison under similar conditions (C5G7 7-group, 1M particles/batch):

| Platform | Throughput (hist/sec) | TDP (W) | Efficiency (hist/sec/W) |
|----------|----------------------|---------|------------------------|
| **Persistent kernel — M4 Max** | **1,762,612** | **40** | **44,065** |
| Event-based — M4 Max | 111,772 | 40 | 2,794 |
| MC/DC — 1x V100 | 109,000 | 300 | 363 |
| MC/DC — 4x V100 | 437,000 | 1,200 | 364 |

The persistent kernel on the Apple M4 Max achieves **4.0x the throughput of 4x V100** while consuming **30x less power**, for an energy efficiency advantage of **121x** over the multi-GPU MC/DC configuration.

> **Note on CPU vs GPU performance**: The persistent kernel assembly benchmark (1,762,612 hist/sec) is **7.6x faster** than the Numba single-thread CPU baseline (231,819 particles/sec). The event-based pincell benchmark (46,198 particles/sec) remains slower than the Numba baseline due to branch divergence at small scale, but the persistent kernel eliminates this bottleneck by processing complete histories without inter-kernel synchronization.

## Architecture

### Event-Based Transport Loop (3 Fused Kernels)

The transport cycle uses three fused Metal compute kernels per step, reduced from an original five-kernel design via kernel fusion. This minimizes dispatch overhead while maintaining the event-based structure where all in-flight particles are processed through the same operation in parallel.

1. **`xs_lookup_and_distance`** (fused) — Reads each particle's material and energy group, retrieves cross sections from the material buffer, then immediately samples a collision distance from an exponential distribution. Fusing these two operations eliminates a kernel boundary for data that stays in the same thread.

2. **`move_particle`** — Advances each particle along its direction vector to the nearer of the sampled collision site or the next geometry boundary. Particles crossing boundaries trigger a material update via lattice-accelerated cell finding. This kernel remains standalone due to complex geometry divergence.

3. **`collision_and_tally`** (fused) — Applies collision physics (scatter, fission, or capture) and immediately scores track-length flux and fission rate tallies using atomic float CAS loops. Fission events deposit sites into the fission bank for the next generation.

### History-Based Persistent Kernel

The persistent kernel dispatches a single long-running Metal compute shader that processes complete neutron histories end-to-end without returning to the CPU between transport steps. Each GPU thread owns one neutron history from birth to absorption, iterating through geometry crossings, collisions, and tally scoring entirely within the shader. This eliminates the kernel launch overhead and global synchronization barriers that limit throughput in the event-based design.

Key design choices that enable the 15.7x speedup on the assembly benchmark:

- **Single-dispatch transport**: one `dispatchThreadgroups` call covers an entire batch; no per-step round-trips to the host.
- **Structure-of-Arrays (SoA) memory layout**: particle position, direction, energy group, and weight fields are stored in separate arrays rather than interleaved structs, improving GPU memory coalescing across threads.
- **Zero-allocation batch loops**: fission sites are written directly into a pre-allocated bank with atomic index increments; no dynamic allocation occurs inside the shader.

The pincell geometry (uniform material, minimal boundary crossings) limits the speedup to 4.3x because threads remain divergent across collision types. The assembly geometry (heterogeneous pins, frequent material crossings) exposes far more parallelism, yielding the full 15.7x gain.

### Unified Memory Advantage

Apple Silicon's unified memory architecture eliminates the PCIe transfer bottleneck present on discrete GPU systems. Particle state arrays, cross-section tables, geometry data, and tally buffers all reside in a single physical memory pool accessible by both the CPU and GPU without explicit copy operations. This allows the host code to inspect and modify simulation state between kernel dispatches with zero transfer overhead.

## Project Structure

```
src/openmc_metal/
  engine.py          - Metal GPU engine (PyObjC Metal API bindings)
  simulation.py      - K-eigenvalue simulation driver and power iteration
  cross_sections.py  - C5G7 7-group cross-section data tables
  geometry.py        - Geometry builder (pins, assemblies, core)
  collision.py       - Fission bank management and collision dispatch
  transport.py       - Transport kernel dispatch and buffer management
  tally.py           - Tally scoring and statistical reduction
  shaders.py         - Metal shader library loader and kernel cache
  persistent/
    shader.py        - Metal persistent transport shader (695 lines MSL)
    engine.py        - PyObjC GPU engine for persistent kernel
    simulation.py    - History-based simulation driver
  benchmarks/
    runner.py        - Benchmark CLI entry point
    xsbench.py       - XSBench cross-section lookup microbenchmark
    pincell.py       - C5G7 pincell criticality benchmark
    cpu_baseline.py  - CPU baseline transport for speedup comparison
  report/
    pdf_report.py    - PDF report generator (reportlab)
    charts.py        - Matplotlib chart generation
Sources/Shaders/
  Common.metal       - Shared type definitions and RNG (Philox-2x32-10)
  Geometry.metal     - Surface intersection and cell membership evaluation
  XSLookup.metal     - Cross-section lookup kernel
  Transport.metal    - Distance sampling and particle move kernels
  Collision.metal    - Scatter/fission/absorption physics kernel
  Tally.metal        - Flux and fission rate tally scoring kernel
```

## Requirements

- macOS with Apple Silicon (M1, M2, M3, or M4 family)
- Python 3.10 or later
- PyObjC (Metal GPU access via Objective-C bridge)
- reportlab (PDF report generation)
- matplotlib (benchmark charts)

## Installation

```bash
cd python
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

PyObjC will be installed automatically as a dependency. No separate Metal SDK installation is required; the framework is provided by macOS.

## Usage

```bash
# Run the full benchmark suite (pincell + XSBench + CPU baseline + PDF report)
python -m openmc_metal.benchmarks.runner

# Generate a PDF report from previously saved benchmark results
python -m openmc_metal.report.pdf_report benchmark_results.json -o report.pdf
```

The benchmark runner saves results to `benchmark_results.json` in the working directory. The PDF report includes k-eff convergence history, Shannon entropy trace, throughput comparison charts, and a summary table.

## References

1. Tramm, J. R., et al. (2024). "GPU-accelerated Monte Carlo particle transport with OpenMC." *Annals of Nuclear Energy*.
2. Morgan, L. M., et al. (2025). "MC/DC: A performant, scalable, and portable Monte Carlo neutron transport code using Numba." *Journal of Open Source Software*.
3. Biondo, E., et al. (2025). "Shift: A massively parallel Monte Carlo radiation transport package." *Nuclear Science and Engineering*.
4. Hamilton, S. P., and Evans, T. M. (2018). "Multigroup Monte Carlo on GPUs: Comparison of history- and event-based algorithms." *Annals of Nuclear Energy*, 113, 506-518.
5. NEA/NSC (2003). "Benchmark on Deterministic Transport Calculations Without Spatial Homogenisation: A 2-D/3-D MOX Fuel Assembly Benchmark." NEA/NSC/DOC(2003)16.
6. Romano, P. K., et al. (2015). "OpenMC: A state-of-the-art Monte Carlo code for research and development." *Annals of Nuclear Energy*, 82, 90-97.

## License

MIT License. See [LICENSE](LICENSE) for details.
