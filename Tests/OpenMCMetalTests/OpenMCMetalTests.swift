import Testing
@testable import OpenMCMetal

@Test func philoxRNGDeterministic() {
    // Philox must produce identical output for identical (counter, key)
    var rng1 = PhiloxRNG(key: 42, counterHi: 0, counterLo: 0)
    var rng2 = PhiloxRNG(key: 42, counterHi: 0, counterLo: 0)

    for _ in 0..<100 {
        #expect(rng1.uniform() == rng2.uniform())
    }
}

@Test func philoxRNGRange() {
    var rng = PhiloxRNG(key: 123, counterHi: 0)
    for _ in 0..<10000 {
        let val = rng.uniform()
        #expect(val >= 0.0)
        #expect(val < 1.0)
    }
}

@Test func philoxDifferentKeys() {
    var rng1 = PhiloxRNG(key: 0)
    var rng2 = PhiloxRNG(key: 1)
    let v1 = rng1.uniform()
    let v2 = rng2.uniform()
    #expect(v1 != v2)
}

@Test func particleStructSize() {
    // 28 x 4-byte fields = 112 bytes
    #expect(MemoryLayout<Particle>.stride == 112, "Particle struct size mismatch - GPU layout will be wrong")
}

@Test func gpuSurfaceStructSize() {
    #expect(MemoryLayout<GPUSurface>.stride == 32, "GPUSurface struct size mismatch")
}

@Test func gpuCellStructSize() {
    #expect(MemoryLayout<GPUCell>.stride == 16, "GPUCell struct size mismatch")
}

@Test func gpuCellSurfaceStructSize() {
    #expect(MemoryLayout<GPUCellSurface>.stride == 8, "GPUCellSurface struct size mismatch")
}

@Test func simulationParamsStructSize() {
    #expect(MemoryLayout<SimulationParams>.stride == 32, "SimulationParams struct size mismatch")
}

@Test func materialXSAccessors() {
    var xs = MaterialXS()
    xs.setTotalXS(group: 0, value: 1.5)
    xs.setTotalXS(group: 6, value: 3.0)
    #expect(xs.totalXS(group: 0) == 1.5)
    #expect(xs.totalXS(group: 6) == 3.0)

    xs.setScatterXS(from: 0, to: 1, value: 0.5)
    #expect(xs.scatterXS(from: 0, to: 1) == 0.5)
    #expect(xs.scatterXS(from: 1, to: 0) == 0.0)
}

@Test func gpuLatticeStructSize() {
    // 3 x (float3 + pad) = 3 x 16 = 48, plus uint3 + pad = 16 -> 48+16 = 64?
    // Actually: float3(16) + float(4) = 16 aligned, float3(16) + float(4) = 16, uint3(12) + uint(4) = 16
    // Total = 48
    #expect(MemoryLayout<GPULattice>.stride == 48, "GPULattice struct size mismatch")
}

@Test func materialXSStructSize() {
    // 7 + 49 + 7 + 7 + 7 = 77 floats * 4 = 308 bytes
    #expect(MemoryLayout<MaterialXS>.stride == 308, "MaterialXS struct size mismatch")
}
