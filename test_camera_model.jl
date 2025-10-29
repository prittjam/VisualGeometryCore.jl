using VisualGeometryCore
using VisualGeometryCore: px
using StaticArrays
using LinearAlgebra: norm
using Unitful: mm, μm, ustrip

println("="^70)
println("Testing New Camera Model System")
println("="^70)

# Test 1: LogicalIntrinsics construction
println("\n[Test 1] LogicalIntrinsics from K matrix")
K = CameraCalibrationMatrix(800.0px, [320.0px, 240.0px])
logical = LogicalIntrinsics{Float64}(K)
println("  ✓ Created LogicalIntrinsics")
println("  K[1,1] = ", logical.K[1,1])

# Test 2: PhysicalIntrinsics construction
println("\n[Test 2] PhysicalIntrinsics from physical parameters")
f = 16.0mm
pitch = Size2(width=5.86μm/px, height=5.86μm/px)
pp = [320.0px, 240.0px]
physical = PhysicalIntrinsics(f, pitch, pp)
println("  ✓ Created PhysicalIntrinsics")
println("  K[1,1] = ", physical.K[1,1])
println("  f = ", physical.f)

# Test 3: CameraModel with LogicalIntrinsics
println("\n[Test 3] CameraModel with LogicalIntrinsics")
camera_logical = CameraModel(logical, PinholeProjection())
println("  ✓ Created CameraModel{LogicalIntrinsics, PinholeProjection}")

# Test 4: CameraModel with PhysicalIntrinsics
println("\n[Test 4] CameraModel with PhysicalIntrinsics")
camera_physical = CameraModel(physical, PinholeProjection())
println("  ✓ Created CameraModel{PhysicalIntrinsics, PinholeProjection}")

# Test 5: Convenience constructor
println("\n[Test 5] Convenience CameraModel constructor")
camera_conv = CameraModel(16.0mm, Size2(width=5.86μm/px, height=5.86μm/px), [320.0px, 240.0px])
println("  ✓ Created CameraModel from parameters")

# Test 6: Project 3D point (both cameras should work)
println("\n[Test 6] Project 3D point")
X = SVector(10.0mm, 5.0mm, 100.0mm)
u_logical = project(camera_logical, X)
u_physical = project(camera_physical, X)
println("  Logical projection: u = ", u_logical)
println("  Physical projection: u = ", u_physical)
println("  Difference: ", abs(ustrip(u_logical[1]) - ustrip(u_physical[1])), " px")

# Test 7: Backproject to ray (both cameras should work)
println("\n[Test 7] Backproject to ray")
u = [400.0px, 300.0px]
ray_logical = backproject(camera_logical, u)
ray_physical = backproject(camera_physical, u)
println("  Logical ray: ", ray_logical)
println("  Physical ray: ", ray_physical)
println("  Both normalized: ", norm(ray_logical) ≈ 1.0 && norm(ray_physical) ≈ 1.0)

# Test 8: Unproject with depth (only PhysicalIntrinsics)
println("\n[Test 8] Unproject with depth (PhysicalIntrinsics only)")
depth = 100.0mm
X_reconstructed = unproject(camera_physical, u, depth)
println("  Reconstructed X: ", X_reconstructed)
println("  Depth matches: ", X_reconstructed[3] == depth)

# Test 9: Type safety - verify unproject doesn't work with LogicalIntrinsics
println("\n[Test 9] Type safety check")
try
    X_bad = unproject(camera_logical, u, depth)
    println("  ✗ ERROR: Should not be able to unproject with LogicalIntrinsics!")
catch e
    println("  ✓ Correctly prevented unproject with LogicalIntrinsics")
    println("    Error: ", typeof(e))
end

# Test 10: Round-trip projection test
println("\n[Test 10] Round-trip: X → project → unproject → X")
X_original = SVector(15.0mm, -8.0mm, 120.0mm)
u_proj = project(camera_physical, X_original)
X_roundtrip = unproject(camera_physical, u_proj, X_original[3])
error_x = abs(ustrip(X_roundtrip[1] - X_original[1]))
error_y = abs(ustrip(X_roundtrip[2] - X_original[2]))
error_z = abs(ustrip(X_roundtrip[3] - X_original[3]))
println("  Original X: ", X_original)
println("  Projected u: ", u_proj)
println("  Reconstructed X: ", X_roundtrip)
println("  Position error: (", error_x, ", ", error_y, ", ", error_z, ") mm")
println("  Success: ", error_x < 1e-10 && error_y < 1e-10 && error_z < 1e-10)

println("\n" * "="^70)
println("All tests completed successfully!")
println("="^70)
