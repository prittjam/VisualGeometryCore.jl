#!/usr/bin/env julia

include("transforms_and_conics.jl")
using .Transforms2D, .Conics2D
using StaticArrays, LinearAlgebra, CoordinateTransformations, Rotations

println("🐛 Debug Specific Problematic Ellipse")
println("=" ^ 40)

# This is the problematic ellipse from the verification
test_ellipse = Ellipse((@SVector [1.0, 2.0]), 3.0, 1.5, π/6)
println("Original ellipse: center=$(test_ellipse.center), a=$(test_ellipse.a), b=$(test_ellipse.b), θ=$(test_ellipse.θ)")

# Convert to conic
test_conic = HomogeneousConic(test_ellipse)
println("\nConic matrix:")
display(test_conic.C)

# Recover ellipse
recovered = Ellipse(test_conic)
println("\nRecovered ellipse: center=$(recovered.center), a=$(recovered.a), b=$(recovered.b), θ=$(recovered.θ)")

# Check roundtrip errors
println("\nRoundtrip errors:")
println("  Center: $(norm(recovered.center - test_ellipse.center))")
println("  a: $(abs(recovered.a - test_ellipse.a))")
println("  b: $(abs(recovered.b - test_ellipse.b))")
println("  θ: $(abs(recovered.θ - test_ellipse.θ))")

# Test a few points manually
println("\n\nManual point test:")
# Points on unit circle
unit_pts = [(@SVector [1.0, 0.0]), (@SVector [0.0, 1.0]), (@SVector [-1.0, 0.0]), (@SVector [0.0, -1.0])]

# Transform using our method
transform = CoordinateTransformations.Translation(test_ellipse.center) ∘ 
           CoordinateTransformations.LinearMap(Rotations.RotMatrix{2,Float64}(test_ellipse.θ)) ∘ 
           CoordinateTransformations.LinearMap(@SMatrix [test_ellipse.a 0; 0 test_ellipse.b])

for (i, p) in enumerate(unit_pts)
    transformed = transform(p)
    homog_p = (@SVector [transformed[1], transformed[2], 1.0])
    value = homog_p' * test_conic.C * homog_p
    println("  Unit point $i: $p -> $transformed, conic value: $value")
end

# Now test polygonization
println("\n\nPolygonization test:")
polygon = polygonize(test_conic; num_points=8)
for (i, p) in enumerate(polygon)
    homog_p = (@SVector [p[1], p[2], 1.0])
    value = homog_p' * test_conic.C * homog_p
    println("  Polygon point $i: $p, conic value: $value")
end