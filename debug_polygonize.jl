#!/usr/bin/env julia

include("transforms_and_conics.jl")
using .Transforms2D, .Conics2D
using StaticArrays, LinearAlgebra, CoordinateTransformations, Rotations

println("🐛 Debug Polygonization")
println("=" ^ 30)

# Test with simple unit circle at origin
println("\n1. Unit circle at origin:")
unit_ellipse = Ellipse((@SVector [0.0, 0.0]), 1.0, 1.0, 0.0)
unit_conic = HomogeneousConic(unit_ellipse)
println("Ellipse: center=$(unit_ellipse.center), a=$(unit_ellipse.a), b=$(unit_ellipse.b), θ=$(unit_ellipse.θ)")
println("Conic matrix:")
display(unit_conic.C)

# Check if unit circle points satisfy the conic equation
test_points = [(@SVector [1.0, 0.0]), (@SVector [0.0, 1.0]), (@SVector [-1.0, 0.0]), (@SVector [0.0, -1.0])]
println("\nUnit circle points on conic?")
for p in test_points
    homog_p = (@SVector [p[1], p[2], 1.0])
    value = homog_p' * unit_conic.C * homog_p
    println("  $p: $(value) (should be ≈ 0)")
end

# Test polygonization of unit circle
polygon = polygonize(unit_conic; num_points=8)
println("\nPolygonized unit circle:")
for (i, p) in enumerate(polygon)
    homog_p = (@SVector [p[1], p[2], 1.0])
    value = homog_p' * unit_conic.C * homog_p
    println("  Point $i: $p, conic value: $value")
end

# Test with translated circle
println("\n\n2. Translated circle:")
translated_ellipse = Ellipse((@SVector [2.0, 1.0]), 1.0, 1.0, 0.0)
translated_conic = HomogeneousConic(translated_ellipse)
println("Ellipse: center=$(translated_ellipse.center), a=$(translated_ellipse.a), b=$(translated_ellipse.b), θ=$(translated_ellipse.θ)")

# Test polygonization
polygon2 = polygonize(translated_conic; num_points=8)
println("\nPolygonized translated circle:")
for (i, p) in enumerate(polygon2[1:4])  # Just first 4
    homog_p = (@SVector [p[1], p[2], 1.0])
    value = homog_p' * translated_conic.C * homog_p
    println("  Point $i: $p, conic value: $value")
end

# Let's also check the transform construction manually
println("\n\n3. Manual transform check:")
e = translated_ellipse
transform = CoordinateTransformations.Translation(e.center) ∘ 
           CoordinateTransformations.LinearMap(Rotations.RotMatrix{2,Float64}(e.θ)) ∘ 
           CoordinateTransformations.LinearMap(@SMatrix [e.a 0; 0 e.b])

# Apply to unit circle points
unit_pts = [(@SVector [1.0, 0.0]), (@SVector [0.0, 1.0])]
println("Unit circle points transformed:")
for p in unit_pts
    transformed = transform(p)
    homog_p = (@SVector [transformed[1], transformed[2], 1.0])
    value = homog_p' * translated_conic.C * homog_p
    println("  $p -> $transformed, conic value: $value")
end