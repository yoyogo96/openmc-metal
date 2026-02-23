#!/bin/bash
set -e

echo "OpenMC-Metal Benchmark Runner"
echo "=============================="
echo ""

# Build in release mode
echo "Building in release mode..."
swift build -c release 2>&1

echo ""
echo "Running benchmarks..."
echo ""

# Run full benchmark suite
.build/release/OpenMCMetal benchmark

echo ""
echo "Benchmarks complete."
