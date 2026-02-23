// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "OpenMCMetal",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.2.0")
    ],
    targets: [
        .executableTarget(
            name: "OpenMCMetal",
            dependencies: [
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ],
            path: "Sources/OpenMCMetal"
        ),
        .testTarget(
            name: "OpenMCMetalTests",
            dependencies: ["OpenMCMetal"],
            path: "Tests/OpenMCMetalTests"
        )
    ]
)
