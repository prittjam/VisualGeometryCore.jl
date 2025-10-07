#!/usr/bin/env julia

"""
Demonstration of VisualGeometryCore ellipse and transform functionality.

This example shows how to:
1. Create ellipses with robust axis detection
2. Convert between ellipse and conic representations
3. Apply geometric transformations
4. Use GeometryBasics integration for visualization
"""

using VisualGeometryCore
using StaticArrays
using LinearAlgebra
using GeometryBasics
using CoordinateTransformations
using Rotations

println("🔗 VisualGeometryCore Ellipse & Transform Demo")
println("=" ^ 50)

# 1. Create ellipses with automatic axis correction
println("\n📐 Creating Ellipses:")
ellipse1 = Ellipse(SVector(0.0, 0.0), 3.0, 2.0, π/4)
println("  Standard ellipse: center=$(ellipse1.center), a=$(ellipse1.a), b=$(ellipse1.b), θ=$(ellipse1.θ)")

# This will automatically swap axes since b > a
ellipse2 = Ellipse(SVector(1.0, 2.0), 2.0, 4.0, π/6)  
println("  Auto-corrected:   center=$(ellipse2.center), a=$(ellipse2.a), b=$(ellipse2.b), θ=$(ellipse2.θ)")

# 2. Ellipse ⇄ Conic conversion with robust reconstruction
println("\n🔄 Ellipse ⇄ Conic Conversion:")
conic = HomogeneousConic(ellipse1)
recovered = Ellipse(conic)

center_error = norm(ellipse1.center - recovered.center)
a_error = abs(ellipse1.a - recovered.a)
b_error = abs(ellipse1.b - recovered.b)
θ_error = abs(ellipse1.θ - recovered.θ)

println("  Original:  $(ellipse1)")
println("  Recovered: $(recovered)")
println("  Errors: center=$(center_error), a=$(a_error), b=$(b_error), θ=$(θ_error)")

# 3. Geometric transformations
println("\n🔄 Geometric Transformations:")

# Translation
translation = Translation(SVector(5.0, -2.0))
translated_conic = push_conic(translation, conic)
translated_ellipse = Ellipse(translated_conic)
println("  After translation by [5, -2]: center=$(translated_ellipse.center)")

# Rotation
rotation = LinearMap(RotMatrix{2}(π/3))
rotated_conic = push_conic(rotation, conic)
rotated_ellipse = Ellipse(rotated_conic)
println("  After rotation by π/3: θ=$(rotated_ellipse.θ)")

# Scaling
scaling = LinearMap(@SMatrix [2.0 0.0; 0.0 1.5])
scaled_conic = push_conic(scaling, conic)
scaled_ellipse = Ellipse(scaled_conic)
println("  After scaling by [2, 1.5]: a=$(scaled_ellipse.a), b=$(scaled_ellipse.b)")

# 4. Homogeneous transform system
println("\n🏗️  Homogeneous Transform System:")
rot_hom = to_homogeneous(RotMatrix{2}(π/4))
trans_hom = to_homogeneous(Translation(SVector(3.0, 1.0)))
scale_hom = to_homogeneous(LinearMap(@SMatrix [1.5 0.0; 0.0 1.5]))

println("  Rotation matrix type: $(typeof(rot_hom))")
println("  Translation matrix type: $(typeof(trans_hom))")
println("  Scale matrix type: $(typeof(scale_hom))")

# Compose transforms
euclidean = rot_hom * trans_hom
affine = euclidean * scale_hom

println("  Euclidean composition type: $(typeof(euclidean))")
println("  Affine composition type: $(typeof(affine))")

# 5. GeometryBasics integration
println("\n🎨 GeometryBasics Integration:")
test_ellipse = Ellipse(SVector(2.0, 1.0), 4.0, 2.0, π/6)

# Interface methods
center = coordinates(test_ellipse)
major_radius = radius(test_ellipse)
println("  Center: $(center)")
println("  Major radius: $(major_radius)")

# Point decomposition for visualization
points_f64 = decompose(Point{2,Float64}, test_ellipse; resolution=12)
points_f32 = decompose(Point2f, test_ellipse; resolution=8)

println("  Generated $(length(points_f64)) Float64 points")
println("  Generated $(length(points_f32)) Float32 points")

# Verify points lie on ellipse
test_conic = HomogeneousConic(test_ellipse)
errors = Float64[]
for p in points_f64
    homog_p = SVector(p[1], p[2], 1.0)
    error = abs(homog_p' * SMatrix{3,3,Float64}(test_conic) * homog_p)
    push!(errors, error)
end
println("  Max conic equation error: $(maximum(errors)) (should be ≈ 0)")

# 6. Gradient computation
println("\n∇ Gradient Computation:")
grad_center = gradient(test_conic, test_ellipse.center)
boundary_point = SVector(test_ellipse.center[1] + test_ellipse.a, test_ellipse.center[2])
grad_boundary = gradient(test_conic, boundary_point)

println("  Gradient at center: $(grad_center) (should be ≈ [0,0])")
println("  Gradient at boundary: $(grad_boundary)")

println("\n✅ Demo completed successfully!")
println("\nKey features demonstrated:")
println("  • Robust ellipse construction with automatic axis ordering")
println("  • High-precision ellipse ⇄ conic conversion using eigenvalue decomposition")
println("  • Efficient homogeneous transform system with type preservation")
println("  • Seamless GeometryBasics integration for visualization")
println("  • Geometric transformations with proper conic push-forward")