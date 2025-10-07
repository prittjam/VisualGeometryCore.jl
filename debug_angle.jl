#!/usr/bin/env julia

include("transforms_and_conics.jl")
using .Transforms2D, .Conics2D
using StaticArrays, LinearAlgebra, CoordinateTransformations, Rotations

println("🐛 Debug Angle Issue")
println("=" ^ 20)

# Test with different angles
angles = [0.0, π/6, π/4, π/3, π/2, 2π/3, 3π/4, 5π/6, π]

for θ_orig in angles
    println("\n--- Testing θ = $θ_orig ($(θ_orig * 180/π)°) ---")
    
    # Create ellipse
    ellipse = Ellipse((@SVector [0.0, 0.0]), 2.0, 1.0, θ_orig)
    
    # Convert to conic and back
    conic = HomogeneousConic(ellipse)
    recovered = Ellipse(conic)
    
    println("Original θ:  $θ_orig")
    println("Recovered θ: $(recovered.θ)")
    println("Difference:  $(abs(recovered.θ - θ_orig))")
    
    # Test if they represent the same ellipse by checking a few points
    transform_orig = CoordinateTransformations.LinearMap(Rotations.RotMatrix{2,Float64}(θ_orig)) ∘ 
                    CoordinateTransformations.LinearMap(@SMatrix [2.0 0; 0 1.0])
    
    transform_recovered = CoordinateTransformations.LinearMap(Rotations.RotMatrix{2,Float64}(recovered.θ)) ∘ 
                         CoordinateTransformations.LinearMap(@SMatrix [recovered.a 0; 0 recovered.b])
    
    # Test a few unit circle points
    test_pts = [(@SVector [1.0, 0.0]), (@SVector [0.0, 1.0])]
    max_diff = 0.0
    for p in test_pts
        pt1 = transform_orig(p)
        pt2 = transform_recovered(p)
        diff = norm(pt1 - pt2)
        max_diff = max(max_diff, diff)
    end
    println("Max point difference: $max_diff")
end