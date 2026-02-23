# OpenMC-Metal: Monte Carlo Neutron Transport on Apple Metal GPU

A GPU-accelerated Monte Carlo neutron transport code targeting Apple Silicon using Metal Shading Language. OpenMC-Metal implements an event-based transport algorithm with k-eigenvalue criticality calculations and C5G7 7-group cross sections.

## Overview

OpenMC-Metal brings Monte Carlo neutron transport to Apple Silicon GPUs via the Metal compute pipeline. The implementation uses an event-based (or "history-based with event queues") algorithm decomposed into five GPU compute kernels, each handling a distinct phase of the transport cycle. Cross-section data follows the C5G7 7-group benchmark specification covering seven representative reactor materials.

The Python driver layer uses PyObjC to interface directly with the Metal API, avoiding the need for intermediate frameworks. Because Apple Silicon uses a unified memory architecture, neutron state buffers are shared between the CPU and GPU without explicit data transfers, which reduces latency and simplifies the host-device coordination logic.

## Features

- **GPU-accelerated multi-group Monte Carlo transport** using Metal compute shaders on Apple Silicon
- **K-eigenvalue power iteration** with Shannon entropy convergence monitoring
- **C5G7 7-group benchmark cross sections** covering 7 materials (UO2, MOX 4.3%, MOX 7.0%, MOX 8.7%, fission chamber, guide tube, moderator)
- **XSBench microbenchmark** for measuring cross-section lookup throughput in isolation
- **Comprehensive PDF benchmark report** with Matplotlib charts and tabulated results
- **CPU baseline transport** for direct GPU speedup measurement under identical problem conditions

## Benchmark Results

Results on an Apple M4 Max (40 GPU cores) running the C5G7 pincell criticality benchmark:

| Metric | Value |
|--------|-------|
| GPU | Apple M4 Max (40 GPU cores) |
| k-eff | 0.23192 +/- 0.00097 |
| Reference k-eff | 0.2327 |
| Delta | +77 pcm |
| GPU Throughput | 27,633 particles/sec |
| CPU Throughput | 4,171 particles/sec |
| GPU Speedup | 6.6x |
| XSBench | 11,372 M lookups/sec |

The computed k-eff of 0.23192 falls within 77 pcm of the published C5G7 reference value of 0.2327, confirming correctness of the multi-group physics implementation.

## Architecture

### Event-Based Transport Loop

The transport cycle is decomposed into five Metal compute kernels dispatched sequentially each iteration. This event-based structure allows the GPU to process all in-flight particles through the same operation in parallel before advancing to the next phase.

1. **`xs_lookup`** - Reads each particle's current material index and energy group, retrieves the corresponding total, scatter, fission, and absorption cross sections from the material buffer, and stores them in the particle state for use by downstream kernels.

2. **`distance_to_collision`** - Samples a collision distance from an exponential distribution using the total macroscopic cross section obtained in the lookup step. The sampled distance is stored alongside the particle direction for the move step.

3. **`move_particle`** - Advances each particle along its direction vector to the nearer of the sampled collision site or the next geometry boundary. Particles crossing boundaries trigger a material update. Particles that escape the geometry are terminated and flagged for bank replacement.

4. **`collision`** - Applies collision physics at the interaction site. Each particle is classified as a scatter, fission, or capture event. Scattered particles receive a new energy group and direction sampled from the scattering matrix. Fission events deposit sites into the fission bank for the next generation. Absorbed and leaked particles are retired.

5. **`tally_score`** - Scores the flux estimator and fission rate tally for the current track segment. Atomic adds accumulate contributions into shared tally buffers, which are reduced on the CPU at the end of each generation.

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
  Common.metal       - Shared type definitions and RNG (xoshiro128++)
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
