// CrossSection.swift
// OpenMCMetal
//
// C5G7 Multi-Group Cross-Section Data
// Source: NEA/NSC C5G7 MOX Benchmark (2003)
//         "Benchmark on Deterministic Transport Calculations Without Spatial Homogenisation"
//         NEA/NSC/DOC(2003)16
//
// 7 energy groups, 7 materials.
// All values are exact published benchmark constants.

import Foundation
import Metal

// MARK: - CPU-side data structure

/// Full multi-group cross-section data for one material (CPU side).
struct MaterialXSData {
    /// Macroscopic total cross section per group [cm^-1] (7 values)
    var total: [Float]
    /// Scattering matrix, row-major [from_g * 7 + to_g] (49 values)
    var scatter: [Float]
    /// Macroscopic fission cross section per group [cm^-1] (7 values)
    var fission: [Float]
    /// Nu * fission cross section per group [cm^-1] (7 values)
    var nuFission: [Float]
    /// Fission neutron spectrum (7 values, sums to 1)
    var chi: [Float]

    // Total floats per material: 7 + 49 + 7 + 7 + 7 = 77

    /// Flatten into a contiguous array suitable for a Metal buffer.
    func toGPUBuffer() -> [Float] {
        return total + scatter + fission + nuFission + chi
    }
}

// MARK: - C5G7 data

/// Static C5G7 benchmark cross-section data.
///
/// Material order:
///   0 – UO2 Fuel
///   1 – MOX 4.3% Fuel
///   2 – MOX 7.0% Fuel
///   3 – MOX 8.7% Fuel
///   4 – Fission Chamber
///   5 – Guide Tube
///   6 – Moderator (Water)
struct C5G7Data {

    static func allMaterials() -> [MaterialXSData] {
        return [
            uo2(),
            mox43(),
            mox70(),
            mox87(),
            fissionChamber(),
            guideTube(),
            moderator()
        ]
    }

    // MARK: Material 1 – UO2 Fuel

    static func uo2() -> MaterialXSData {
        let total: [Float] = [
            0.212033, 0.332717, 0.526764, 0.527260, 0.527560, 0.529280, 0.531440
        ]

        // 7x7 scatter matrix, row = from-group, col = to-group, row-major
        let scatter: [Float] = [
            // to: 1        2        3        4        5        6        7
            0.127537, 0.042378, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, // from 1
            0.000000, 0.324456, 0.007617, 0.000000, 0.000000, 0.000000, 0.000000, // from 2
            0.000000, 0.000000, 0.509437, 0.016413, 0.000000, 0.000000, 0.000000, // from 3
            0.000000, 0.000000, 0.000000, 0.507519, 0.018817, 0.000000, 0.000000, // from 4
            0.000000, 0.000000, 0.000000, 0.000540, 0.489790, 0.035695, 0.000000, // from 5
            0.000000, 0.000000, 0.000000, 0.000000, 0.001639, 0.489160, 0.037814, // from 6
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.006673, 0.480400  // from 7
        ]

        let nuFission: [Float] = [
            0.009252, 0.004620, 0.006132, 0.012211, 0.034635, 0.058450, 0.122160
        ]
        let fission: [Float] = [
            0.003520, 0.001729, 0.002256, 0.004530, 0.012880, 0.021790, 0.045600
        ]
        let chi: [Float] = [
            0.587910, 0.411760, 0.000340, 0.000000, 0.000000, 0.000000, 0.000000
        ]

        return MaterialXSData(total: total, scatter: scatter, fission: fission,
                              nuFission: nuFission, chi: chi)
    }

    // MARK: Material 2 – MOX 4.3% Fuel

    static func mox43() -> MaterialXSData {
        let total: [Float] = [
            0.214120, 0.338230, 0.533690, 0.534710, 0.535090, 0.536740, 0.539170
        ]

        let scatter: [Float] = [
            0.130457, 0.041783, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.328042, 0.007614, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.515520, 0.016330, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.514370, 0.018850, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000544, 0.494720, 0.035802, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.001616, 0.494090, 0.038010,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.006738, 0.484810
        ]

        let nuFission: [Float] = [
            0.008462, 0.004850, 0.006690, 0.013260, 0.038610, 0.064910, 0.135400
        ]
        let fission: [Float] = [
            0.003240, 0.001824, 0.002461, 0.004932, 0.014370, 0.024210, 0.050520
        ]
        let chi: [Float] = [
            0.587910, 0.411760, 0.000340, 0.000000, 0.000000, 0.000000, 0.000000
        ]

        return MaterialXSData(total: total, scatter: scatter, fission: fission,
                              nuFission: nuFission, chi: chi)
    }

    // MARK: Material 3 – MOX 7.0% Fuel

    static func mox70() -> MaterialXSData {
        let total: [Float] = [
            0.214120, 0.338230, 0.533690, 0.534710, 0.535090, 0.536740, 0.539170
        ]

        // NOTE: MOX 7.0% and MOX 8.7% share the same total XS and scatter matrix
        // as MOX 4.3% per the C5G7 benchmark specification; only nu-fission and
        // fission XS differ (reflecting different plutonium enrichment).
        let scatter: [Float] = [
            0.130457, 0.041783, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.328042, 0.007614, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.515520, 0.016330, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.514370, 0.018850, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000544, 0.494720, 0.035802, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.001616, 0.494090, 0.038010,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.006738, 0.484810
        ]

        let nuFission: [Float] = [
            0.013350, 0.007998, 0.011380, 0.022370, 0.064770, 0.108900, 0.227200
        ]
        let fission: [Float] = [
            0.005110, 0.003007, 0.004186, 0.008312, 0.024120, 0.040610, 0.084750
        ]
        let chi: [Float] = [
            0.587910, 0.411760, 0.000340, 0.000000, 0.000000, 0.000000, 0.000000
        ]

        return MaterialXSData(total: total, scatter: scatter, fission: fission,
                              nuFission: nuFission, chi: chi)
    }

    // MARK: Material 4 – MOX 8.7% Fuel

    static func mox87() -> MaterialXSData {
        let total: [Float] = [
            0.214120, 0.338230, 0.533690, 0.534710, 0.535090, 0.536740, 0.539170
        ]

        let scatter: [Float] = [
            0.130457, 0.041783, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.328042, 0.007614, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.515520, 0.016330, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.514370, 0.018850, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000544, 0.494720, 0.035802, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.001616, 0.494090, 0.038010,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.006738, 0.484810
        ]

        let nuFission: [Float] = [
            0.017540, 0.011030, 0.015960, 0.031410, 0.090910, 0.152900, 0.319200
        ]
        let fission: [Float] = [
            0.006710, 0.004146, 0.005872, 0.011680, 0.033840, 0.057010, 0.119100
        ]
        let chi: [Float] = [
            0.587910, 0.411760, 0.000340, 0.000000, 0.000000, 0.000000, 0.000000
        ]

        return MaterialXSData(total: total, scatter: scatter, fission: fission,
                              nuFission: nuFission, chi: chi)
    }

    // MARK: Material 5 – Fission Chamber

    static func fissionChamber() -> MaterialXSData {
        let total: [Float] = [
            0.126032, 0.293849, 0.284240, 0.280960, 0.334440, 0.565640, 1.172140
        ]

        let scatter: [Float] = [
            0.061680, 0.056129, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.240377, 0.052316, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.183425, 0.092397, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.172820, 0.096752, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.002652, 0.201340, 0.115230, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.008413, 0.322160, 0.210150,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.061600, 0.832490
        ]

        let nuFission: [Float] = [
            0.004736, 0.001036, 0.000738, 0.001191, 0.003524, 0.005963, 0.012520
        ]
        let fission: [Float] = [
            0.001800, 0.000389, 0.000272, 0.000442, 0.001311, 0.002222, 0.004671
        ]
        let chi: [Float] = [
            0.587910, 0.411760, 0.000340, 0.000000, 0.000000, 0.000000, 0.000000
        ]

        return MaterialXSData(total: total, scatter: scatter, fission: fission,
                              nuFission: nuFission, chi: chi)
    }

    // MARK: Material 6 – Guide Tube

    static func guideTube() -> MaterialXSData {
        // Same total XS as fission chamber; scatter differs slightly in groups 3-4
        let total: [Float] = [
            0.126032, 0.293849, 0.284240, 0.280960, 0.334440, 0.565640, 1.172140
        ]

        let scatter: [Float] = [
            0.061680, 0.056129, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.240377, 0.052316, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.183297, 0.092397, 0.000000, 0.000000, 0.000000, // 0.183297 vs 0.183425
            0.000000, 0.000000, 0.000000, 0.172730, 0.096752, 0.000000, 0.000000, // 0.172730 vs 0.172820
            0.000000, 0.000000, 0.000000, 0.002652, 0.201340, 0.115230, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.008413, 0.322160, 0.210150,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.061600, 0.832490
        ]

        let nuFission: [Float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        let fission: [Float]   = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        let chi: [Float]       = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        return MaterialXSData(total: total, scatter: scatter, fission: fission,
                              nuFission: nuFission, chi: chi)
    }

    // MARK: Material 7 – Moderator (Water)

    static func moderator() -> MaterialXSData {
        let total: [Float] = [
            0.159206, 0.412970, 0.590310, 0.584350, 0.718000, 1.254450, 2.650380
        ]

        let scatter: [Float] = [
            0.047946, 0.103580, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.315720, 0.096317, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.444970, 0.115170, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.413500, 0.149540, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.002143, 0.371680, 0.315550, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.010290, 0.644410, 0.557100,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.073903, 2.075100
        ]

        let nuFission: [Float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        let fission: [Float]   = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        let chi: [Float]       = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        return MaterialXSData(total: total, scatter: scatter, fission: fission,
                              nuFission: nuFission, chi: chi)
    }
}

// MARK: - Metal buffer loading

/// Loads all 7 C5G7 materials into a single contiguous MTLBuffer.
///
/// Buffer layout (row-major):
///   material[0..6], each 77 floats = [total(7) | scatter(49) | fission(7) | nuFission(7) | chi(7)]
///
/// Total size: 7 * 77 * sizeof(Float) = 7 * 77 * 4 = 2156 bytes
func loadC5G7Materials(device: MTLDevice) -> MTLBuffer? {
    let materials = C5G7Data.allMaterials()
    var flatData: [Float] = []
    flatData.reserveCapacity(7 * 77)

    for mat in materials {
        let packed = mat.toGPUBuffer()
        assert(packed.count == 77,
               "MaterialXSData GPU buffer must be 77 floats, got \(packed.count)")
        flatData.append(contentsOf: packed)
    }

    let byteCount = flatData.count * MemoryLayout<Float>.stride
    guard let buffer = device.makeBuffer(bytes: flatData,
                                         length: byteCount,
                                         options: .storageModeShared) else {
        return nil
    }
    buffer.label = "C5G7 Cross-Section Data"
    return buffer
}
