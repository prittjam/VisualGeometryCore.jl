#!/usr/bin/env julia

include("transforms_and_conics.jl")
using .Transforms2D, .Conics2D
using StaticArrays, LinearAlgebra, CoordinateTransformations, Rotations

println("🐛 Debug Eigendecomposition")
println("=" ^ 30)

test_ellipse = Ellipse((@SVector [1.0, 2.0]), 3.0, 1.5, π/6)
test_conic = HomogeneousConic(test_ellipse)

C = test_conic.C
println("Conic matrix C:")
display(C)

# Extract center
center = -C[1:2,1:2] \ C[1:2,3]
println("\nCenter: $center")

# Translate to center
Tc = @SMatrix [ 1.0 0.0 center[1]
                0.0 1.0 center[2]
                0.0 0.0 1.0 ]

println("\nTranslation matrix Tc:")
display(Tc)

# Get centered conic
C_centered = transpose(inv(Tc)) * C * inv(Tc)
println("\nCentered conic C_centered:")
display(C_centered)

# Extract 2x2 part and γ
A22 = C_centered[1:2, 1:2]
γ = C_centered[3,3]

println("\nA22 (2x2 part):")
display(A22)
println("\nγ (bottom-right): $γ")

# Eigendecomposition
eigen_result = eigen(A22)
λ = eigen_result.values
V = eigen_result.vectors

println("\nEigenvalues λ: $λ")
println("Eigenvectors V:")
display(V)

println("\nChecks:")
println("  γ < 0? $(γ < 0)")
println("  all(λ > 0)? $(all(λ .> 0))")

if γ < 0 && all(λ .> 0)
    axes = sqrt.(-γ ./ λ)
    println("  Semi-axes: $axes")
end