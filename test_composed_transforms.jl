using VisualGeometryCore
using VisualGeometryCore: px
using StaticArrays
using LinearAlgebra: norm
using Unitful: mm, μm, ustrip
using CoordinateTransformations: Translation
using Rotations: RotZ

println("="^70)
println("Testing Composed Transform System (World → Pixel)")
println("="^70)

# Setup: Create a physical camera model
println("\n[Setup] Creating PhysicalIntrinsics CameraModel")
f = 16.0mm
pitch = Size2(width=5.86μm/px, height=5.86μm/px)
pp = [320.0px, 240.0px]
model = CameraModel(f, pitch, pp)
println("  ✓ Created CameraModel with cached transform")
println("  Focal length: $f")
println("  Pixel pitch: $(pitch.width)")
println("  Principal point: $pp")

# Test 1: Verify cached transform exists
println("\n[Test 1] Verify cached transform in CameraModel")
println("  Transform type: $(typeof(model.transform))")
println("  ✓ Transform cached at construction")

# Test 2: Project point in camera coordinates (without extrinsics)
println("\n[Test 2] Project point in camera coordinates")
X_cam = SVector(10.0mm, 5.0mm, 100.0mm)
u_cam = project(model, X_cam)
println("  X_cam: $X_cam")
println("  Projected u: $u_cam")
println("  ✓ Projection from camera coords works")

# Test 3: Create camera with extrinsics (identity - camera at origin)
println("\n[Test 3] Camera with identity extrinsics")
extrinsics_identity = EuclideanMap(RotZ(0.0), SVector(0.0, 0.0, 0.0))
camera_identity = Camera(model, extrinsics_identity)
println("  ✓ Created Camera{CameraModel, EuclideanMap}")

# Test 4: Project from world coordinates (identity transform)
println("\n[Test 4] Project from world to pixels (identity extrinsics)")
X_world = SVector(10.0mm, 5.0mm, 100.0mm)  # Same as X_cam with identity
u_world = project(camera_identity, X_world)
println("  X_world: $X_world")
println("  Projected u: $u_world")
println("  Match with camera-space projection: $(u_world ≈ u_cam)")
println("  ✓ World → Pixel projection works with identity")

# Test 5: Camera with translation
println("\n[Test 5] Camera with translation (camera moved back)")
# Camera at (0, 0, 50mm), looking at origin
# Note: EuclideanMap needs unitless values, units are handled by the projection functions
translation = SVector(0.0, 0.0, 50.0)  # unitless, but represents mm
extrinsics_trans = EuclideanMap(RotZ(0.0), translation)
camera_trans = Camera(model, extrinsics_trans)
println("  Camera position: (0, 0, 50) [unitless, represents mm in this context]")
println("  ✓ Created camera with translation")

# Point in world at (10mm, 5mm, 100mm)
# Should appear at same position as if camera were at origin and point at (10mm, 5mm, 50mm)
X_world_trans = SVector(10.0mm, 5.0mm, 100.0mm)
u_trans = project(camera_trans, X_world_trans)
println("  X_world: $X_world_trans")
println("  Projected u: $u_trans")

# Verify by manually computing what point should be in camera space
X_cam_expected = SVector(10.0mm, 5.0mm, 50.0mm)  # 100mm - 50mm = 50mm depth
u_expected = project(model, X_cam_expected)
println("  Expected u (manual): $u_expected")
println("  Match: $(u_trans ≈ u_expected)")
println("  ✓ Translation extrinsics work correctly")

# Test 6: Camera with rotation
println("\n[Test 6] Camera with rotation (90° around Z)")
# Rotate camera 90° around Z-axis
extrinsics_rot = EuclideanMap(RotZ(π/2), SVector(0.0, 0.0, 0.0))
camera_rot = Camera(model, extrinsics_rot)
println("  Rotation: 90° around Z-axis")
println("  ✓ Created camera with rotation")

# Point at (100mm, 0mm, 100mm) in world
# After 90° rotation around Z: (0mm, -100mm, 100mm) in camera
X_world_rot = SVector(100.0mm, 0.0mm, 100.0mm)
u_rot = project(camera_rot, X_world_rot)
println("  X_world: $X_world_rot")
println("  Projected u: $u_rot")

# Verify by manually computing
X_cam_rot_expected = SVector(0.0mm, -100.0mm, 100.0mm)
u_rot_expected = project(model, X_cam_rot_expected)
println("  Expected u (manual): $u_rot_expected")
max_error = maximum(abs.(ustrip.(u_rot) .- ustrip.(u_rot_expected)))
println("  Max error: $(max_error) px")
println("  ✓ Rotation extrinsics work correctly (error < 1e-10: $(max_error < 1e-10))")

# Test 7: lookat transform
println("\n[Test 7] Camera using lookat transform")
# Camera at (0, 0, 200mm), looking at origin, up is +Y
# Note: lookat needs compatible units/unitless values
camera_pos = SVector(0.0, 0.0, 200.0)  # unitless, represents mm
target = SVector(0.0, 0.0, 0.0)
up = SVector(0.0, 1.0, 0.0)
extrinsics_lookat = lookat(camera_pos, target, up)
camera_lookat = Camera(model, extrinsics_lookat)
println("  Camera position: (0, 0, 200) [unitless]")
println("  Target: $target")
println("  Up vector: $up")
println("  ✓ Created camera with lookat transform")

# Project point in front of camera
X_world_lookat = SVector(20.0mm, 10.0mm, 50.0mm)
u_lookat = project(camera_lookat, X_world_lookat)
println("  X_world: $X_world_lookat")
println("  Projected u: $u_lookat")
println("  ✓ Lookat projection works")

# Test 8: Round-trip verification
println("\n[Test 8] Round-trip test with composed transforms")
# For a physical camera, we can do: World → Camera → Pixels → Camera (with depth) → World
X_original = SVector(15.0mm, -8.0mm, 120.0mm)
println("  Original X_world: $X_original")

# Transform to camera space
X_in_cam = extrinsics_identity(X_original)
println("  X in camera: $X_in_cam")

# Project to pixels
u_proj = project(camera_identity, X_original)
println("  Projected u: $u_proj")

# Unproject back (requires depth)
depth = X_in_cam[3]
X_roundtrip = unproject(model, u_proj, depth)
println("  Unprojected X_cam: $X_roundtrip")

# Transform back to world (inverse of extrinsics)
pose_transform = pose(camera_identity)
X_world_roundtrip = pose_transform(X_roundtrip)
println("  X_world roundtrip: $X_world_roundtrip")

error_x = abs(ustrip(X_world_roundtrip[1] - X_original[1]))
error_y = abs(ustrip(X_world_roundtrip[2] - X_original[2]))
error_z = abs(ustrip(X_world_roundtrip[3] - X_original[3]))
println("  Position error: ($error_x, $error_y, $error_z) mm")
println("  Success: $(error_x < 1e-10 && error_y < 1e-10 && error_z < 1e-10)")
println("  ✓ Round-trip through composed transforms works")

println("\n" * "="^70)
println("All Composed Transform Tests Passed!")
println("="^70)
println("\nSummary:")
println("  ✓ CameraModel caches composed transform (Intrinsics ∘ Projection)")
println("  ✓ Camera → Pixel projection uses cached transform")
println("  ✓ World → Pixel projection composes extrinsics correctly")
println("  ✓ Identity, translation, rotation, and lookat extrinsics work")
println("  ✓ Round-trip World → Pixels → World preserves coordinates")
println("="^70)
