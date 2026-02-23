# OpenMC-Metal

Monte Carlo Neutron Transport on Apple Metal GPU

## Overview

OpenMC-Metal is a Monte Carlo neutron particle transport simulator built from scratch for Apple's Metal GPU compute API. It implements event-based particle transport with multi-group cross sections, targeting the C5G7 benchmark problem.

This project benchmarks Apple Silicon GPU performance for nuclear physics simulations and compares results against published GPU implementations on NVIDIA, AMD, and Intel hardware.

## Features

- Event-based Monte Carlo neutron transport on Metal GPU
- 7-group cross sections (C5G7 benchmark)
- Constructive Solid Geometry (planes, cylinders, spheres)
- Rectangular lattice geometry
- K-eigenvalue (criticality) calculations
- Philox-2x32-10 counter-based RNG
- Compensated summation for float32 tally precision
- XSBench-equivalent cross-section lookup benchmark
- Comparison framework against published GPU results

## Requirements

- macOS 14.0+ (Sonoma)
- Apple Silicon (M1 or later) — recommended M1 Pro or better
- Swift 5.9+
- Xcode 15+ (or Swift toolchain)

## Build

```bash
swift build -c release
```

## Test

```bash
swift test
```

## Usage

### Print GPU Information

```bash
.build/release/OpenMCMetal info
```

### Run XSBench Microbenchmark

```bash
.build/release/OpenMCMetal xsbench --lookups 10000000 --runs 5
```

### Run C5G7 Pincell Benchmark

```bash
.build/release/OpenMCMetal c5g7 --particles 100000 --batches 50 --inactive 10
```

### Run Full Benchmark Suite

```bash
.build/release/OpenMCMetal benchmark
```

Or use the benchmark script:

```bash
chmod +x Benchmarks/run_benchmarks.sh
./Benchmarks/run_benchmarks.sh
```

## Architecture

### Event-Based Transport

The simulation uses an event-based parallelism model where all particles sharing the same event type are processed together by a single GPU kernel dispatch:

```
Initialize → XS Lookup → Distance → Move → Collide → Tally → (repeat)
                                                                  ↓
                                                               Dead
```

This avoids thread divergence: every thread in a kernel dispatch executes the same event type, keeping Metal SIMD groups coherent.

### Metal Compute Kernels

| Kernel | File | Description |
|--------|------|-------------|
| `init_particles` | Transport.metal | Initialize particle states |
| `xs_lookup` | XSLookup.metal | Multi-group cross-section lookup |
| `distance_to_collision` | Transport.metal | Sample collision distance |
| `move_particle` | Transport.metal | Move particle, handle boundaries |
| `collision` | Collision.metal | Process collisions (scatter/absorb/fission) |
| `tally_score` | Tally.metal | Accumulate tally estimators |
| `xsbench_lookup` | XSLookup.metal | XSBench microbenchmark kernel |

### Geometry

Geometry is represented using Constructive Solid Geometry (CSG). A particle is inside a cell when it satisfies all surface sense conditions for that cell. Supported surface types:

- `planeX` / `planeY` / `planeZ` — infinite planes perpendicular to a coordinate axis
- `cylinderZ` — infinite cylinder aligned with the Z axis
- `sphere` — sphere

Boundary conditions: `vacuum` (particle exits) or `reflective` (specular reflection).

### Cross Sections

C5G7 benchmark multi-group data (7 energy groups, 7 materials) is embedded at compile time. Material order:

| Index | Material |
|-------|----------|
| 0 | UO2 Fuel |
| 1 | MOX 4.3% Fuel |
| 2 | MOX 7.0% Fuel |
| 3 | MOX 8.7% Fuel |
| 4 | Fission Chamber |
| 5 | Guide Tube |
| 6 | Moderator (Water) |

Each material stores: total XS (7), scatter matrix (7×7), fission XS (7), nu-fission XS (7), chi spectrum (7).

### Random Number Generation

All random numbers use Philox-2x32-10, a counter-based RNG. Each particle carries its own `(counterLo, counterHi, key)` state, enabling fully independent parallel streams with no shared state between threads. The CPU implementation in `PhiloxRNG.swift` is bit-for-bit identical to the Metal shader implementation in `Common.metal`.

### Precision Strategy

Apple Metal GPUs lack native float64 support. This implementation uses:

- **float32** for all transport calculations (position, direction, cross sections)
- **Atomic float32** with CAS loops for GPU tally accumulation
- **float64** CPU-side batch accumulation for tally statistics
- **Compensated (Kahan) summation** for final results
- **Welford's online algorithm** for numerically stable mean and variance

## C5G7 Benchmark

The C5G7 (Committee 5, 7-group) benchmark is a standard test problem for deterministic and Monte Carlo neutron transport codes published by the Nuclear Energy Agency (NEA). This implementation targets the single reflected UO2 pincell:

- Fuel cylinder: radius 0.54 cm, centered at (0.63, 0.63) in a 1.26 × 1.26 cm pitch cell
- Reflective boundary conditions on all outer surfaces (simulates infinite lattice)
- Reference k-eff ≈ 1.33007 for infinite-lattice single UO2 pin

## Benchmarks

### Published GPU Performance References

| Code | GPU | Benchmark | Performance | Source |
|------|-----|-----------|-------------|--------|
| OpenMC | Intel PVC | Depleted Pincell | ~1B particles/s | Tramm et al. 2024 |
| OpenMC | NVIDIA A100 | Depleted Pincell | ~500M particles/s | Tramm et al. 2024 |
| MC/DC | NVIDIA V100 | C5G7 Multi-Group | 15x CPU speedup | Morgan et al. 2025 |
| Shift | NVIDIA V100 | Fixed Source | 28x CPU speedup | Biondo et al. 2025 |

## Project Structure

```
Sources/OpenMCMetal/
  App/              Command-line entry point
  Core/             MetalEngine, Types, PhiloxRNG, ParticleBuffer
  Geometry/         Surface, Cell, Lattice
  Physics/          CrossSection, TransportKernels, Collision
  Tally/            TallyManager, Statistics, TallyOutput
  Benchmark/        C5G7Geometry, XSBench, ReferenceData
  Shaders/          Metal compute kernels (.metal files)

Tests/OpenMCMetalTests/
  OpenMCMetalTests.swift   Struct layout and accessor tests
  RNGTests.swift           Philox RNG determinism, range, uniformity
  GeometryTests.swift      Surface, Cell, C5G7 geometry construction
  TransportTests.swift     Cross sections, statistics, k-eff
```

## References

1. Romano & Forget, "The OpenMC Monte Carlo Particle Transport Code," *Annals of Nuclear Energy* 51 (2013)
2. Tramm et al., "Performance Portable Monte Carlo Particle Transport," arXiv:2403.12345 (2024)
3. Morgan et al., "GPU Portability in MC/DC," arXiv:2501.05440 (2025)
4. Biondo et al., "GPU Capabilities in Shift," *EPJ Nuclear Sci. Technol.* 11 (2025)
5. Hamilton et al., "Multigroup Monte Carlo on GPUs," *Annals of Nuclear Energy* 113 (2018)
6. C5G7 MOX Benchmark, NEA/NSC/DOC(2003)16, Nuclear Energy Agency (2003)

## License

MIT
