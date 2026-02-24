# OpenMC-Metal: Monte Carlo Neutron Transport on Apple Metal GPU

A GPU-accelerated Monte Carlo neutron transport code targeting Apple Silicon using Metal Shading Language. OpenMC-Metal implements an event-based transport algorithm with fused GPU kernels for k-eigenvalue criticality calculations using C5G7 7-group cross sections.

## Overview

OpenMC-Metal brings Monte Carlo neutron transport to Apple Silicon GPUs via the Metal compute pipeline. The implementation uses an event-based algorithm with **three fused GPU compute kernels** per transport step, optimized for Metal's unified memory architecture. Cross-section data follows the C5G7 7-group benchmark specification covering seven representative reactor materials.

The Python driver layer uses PyObjC to interface directly with the Metal API, avoiding the need for intermediate frameworks. Because Apple Silicon uses a unified memory architecture, neutron state buffers are shared between the CPU and GPU without explicit data transfers, which reduces latency and simplifies the host-device coordination logic.

## Features

- **GPU-accelerated multi-group Monte Carlo transport** using Metal compute shaders on Apple Silicon
- **3 fused compute kernels** (down from 5) with kernel fusion for XS lookup + distance sampling and collision + tally scoring
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

### MC/DC Comparison (Morgan et al. 2025)

Direct comparison under similar conditions (C5G7 7-group, 1M particles/batch):

| Platform | Throughput (hist/sec) | TDP (W) | Efficiency (hist/sec/W) |
|----------|----------------------|---------|------------------------|
| **This Work — M4 Max** | **111,772** | **40** | **2,794** |
| MC/DC — 1x V100 | 109,000 | 300 | 363 |
| MC/DC — 4x V100 | 437,000 | 1,200 | 364 |

The Apple M4 Max matches a single NVIDIA V100 in absolute throughput (1.03x) while achieving **7.7x better energy efficiency** (2,794 vs 363 hist/sec/W).

> **Note on CPU vs GPU performance**: A single Numba-JIT compiled CPU thread on the M4 Max (231,819 particles/sec) outperforms the Metal GPU pincell benchmark (46,198 particles/sec) by ~5x, due to branch divergence inherent in Monte Carlo transport. The GPU becomes competitive at scale (1M particles on the 17x17 assembly: 111,772 hist/sec) and offers significant energy efficiency advantages over discrete GPUs.

## Architecture

### Event-Based Transport Loop (3 Fused Kernels)

The transport cycle uses three fused Metal compute kernels per step, reduced from an original five-kernel design via kernel fusion. This minimizes dispatch overhead while maintaining the event-based structure where all in-flight particles are processed through the same operation in parallel.

1. **`xs_lookup_and_distance`** (fused) — Reads each particle's material and energy group, retrieves cross sections from the material buffer, then immediately samples a collision distance from an exponential distribution. Fusing these two operations eliminates a kernel boundary for data that stays in the same thread.

2. **`move_particle`** — Advances each particle along its direction vector to the nearer of the sampled collision site or the next geometry boundary. Particles crossing boundaries trigger a material update via lattice-accelerated cell finding. This kernel remains standalone due to complex geometry divergence.

3. **`collision_and_tally`** (fused) — Applies collision physics (scatter, fission, or capture) and immediately scores track-length flux and fission rate tallies using atomic float CAS loops. Fission events deposit sites into the fission bank for the next generation.

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
