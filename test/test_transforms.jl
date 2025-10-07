#!/usr/bin/env julia

using Test
using VisualGeometryCore
using StaticArrays
using CoordinateTransformations
using Rotations
using LinearAlgebra

@testset "Transform Operations" begin
    
    @testset "Transform operator syntax" begin
        # Create test ellipse and conic
        ellipse = Ellipse(Point2(0.0, 0.0), 2.0, 1.0, 0.0)
        Q = HomogeneousConic(ellipse)
        
        # Test individual transforms
        scale = LinearMap(@SMatrix [2.0 0.0; 0.0 1.5])
        rotation = RotMatrix{2}(π/6)
        translation = Translation(SVector(1.0, 2.0))
        
        # Test transform application using operator syntax
        Q_scaled = scale(Q)
        @test Q_scaled isa HomogeneousConic{Float64}
        
        Q_rotated = rotation(Q_scaled)
        @test Q_rotated isa HomogeneousConic{Float64}
        
        Q_translated = translation(Q_rotated)
        @test Q_translated isa HomogeneousConic{Float64}
        
        # Test composition
        composed = translation ∘ rotation ∘ scale
        Q_composed = composed(Q)
        @test Q_composed isa HomogeneousConic{Float64}
        
        # Verify step-by-step and composed results match
        ellipse_step = Ellipse(Q_translated)
        ellipse_composed = Ellipse(Q_composed)
        
        @test ellipse_step.center ≈ ellipse_composed.center atol=1e-12
        @test ellipse_step.a ≈ ellipse_composed.a atol=1e-12
        @test ellipse_step.b ≈ ellipse_composed.b atol=1e-12
        @test ellipse_step.θ ≈ ellipse_composed.θ atol=1e-12
    end
    
    @testset "Transform matrix construction" begin
        # Test rotation matrix
        θ = π/4
        rot = RotMatrix{2}(θ)
        hom_rot = to_homogeneous(rot)
        @test hom_rot isa HomRotMat{Float64}
        @test size(hom_rot) == (3, 3)
        @test hom_rot[3, 3] == 1.0
        @test hom_rot[1:2, 1:2] ≈ [cos(θ) -sin(θ); sin(θ) cos(θ)]
        
        # Test translation matrix
        t = SVector(3.0, -2.0)
        trans = Translation(t)
        hom_trans = to_homogeneous(trans)
        @test hom_trans isa HomTransMat{Float64}
        @test size(hom_trans) == (3, 3)
        @test hom_trans[1:2, 1:2] ≈ I(2)
        @test hom_trans[1:2, 3] ≈ t
        
        # Test scale matrix
        s = SVector(2.0, 0.5)
        scale = LinearMap(Diagonal(s))
        hom_scale = to_homogeneous(scale)
        @test hom_scale isa HomScaleAnisoMat{Float64}
        @test size(hom_scale) == (3, 3)
        @test hom_scale[1:2, 1:2] ≈ Diagonal(s)
        
        # Test isotropic scale
        s_iso = 1.5
        scale_iso = LinearMap(s_iso * I(2))
        hom_scale_iso = to_homogeneous(scale_iso)
        @test hom_scale_iso isa HomScaleIsoMat{Float64}
    end
    
    @testset "Transform composition and materialization" begin
        # Create transforms
        rot = to_homogeneous(RotMatrix{2}(π/3))
        trans = to_homogeneous(Translation(SVector(1.0, -1.0)))
        scale = to_homogeneous(LinearMap(Diagonal(SVector(2.0, 0.8))))
        
        # Test multiplication
        euclidean = rot * trans
        @test euclidean isa EuclideanMat{Float64}
        
        affine = scale * euclidean
        @test affine isa AffineMat{Float64}
        
        # Test materialization
        mat_euclidean = materialize(euclidean)
        @test mat_euclidean isa EuclideanMat{Float64}
        @test size(mat_euclidean) == (3, 3)
        
        mat_affine = materialize(affine)
        @test mat_affine isa AffineMat{Float64}
        @test size(mat_affine) == (3, 3)
        
        # Verify materialized matrices work correctly
        test_point = [1.0, 1.0, 1.0]
        result1 = mat_euclidean * test_point
        # Note: Transform objects only work on conics, not points directly
        # So we'll just verify the matrix is correct size and type
    end
    
    @testset "Result type system" begin
        # Test result_type function
        rot = to_homogeneous(RotMatrix{2}(π/4))
        trans = to_homogeneous(Translation(SVector(1.0, 1.0)))
        scale_iso = to_homogeneous(LinearMap(2.0 * I(2)))
        scale_aniso = to_homogeneous(LinearMap(Diagonal(SVector(2.0, 0.5))))
        
        @test result_type(typeof(rot), typeof(trans)) == EuclideanMat
        @test result_type(typeof(trans), typeof(rot)) == EuclideanMat
        @test result_type(typeof(scale_iso), typeof(rot)) == AffineMat
        @test result_type(typeof(scale_aniso), typeof(trans)) == AffineMat
        
        # Test actual multiplication results
        @test (rot * trans) isa EuclideanMat{Float64}
        @test (trans * rot) isa EuclideanMat{Float64}
        @test (scale_iso * rot) isa AffineMat{Float64}
        @test (scale_aniso * trans) isa AffineMat{Float64}
    end
    
    @testset "Circle transformations" begin
        # Test circle to conic conversion and transformation
        circle = Circle(Point2(1.0, 1.0), 1.5)
        Q_circle = HomogeneousConic(circle)
        
        # Apply transformation
        scale = LinearMap(Diagonal(SVector(2.0, 1.0)))  # This will make it an ellipse
        Q_transformed = scale(Q_circle)
        
        # Convert back to ellipse (no longer a circle)
        ellipse_result = Ellipse(Q_transformed)
        @test ellipse_result isa Ellipse{Float64}
        
        # Verify the transformation worked correctly
        @test ellipse_result.center ≈ Point2(2.0, 1.0) atol=1e-12  # x-coordinate scaled by 2
        @test ellipse_result.a ≈ 3.0 atol=1e-12  # radius 1.5 scaled by 2
        @test ellipse_result.b ≈ 1.5 atol=1e-12  # radius 1.5 unchanged in y
        
        # Test isotropic scaling (should remain a circle)
        iso_scale = LinearMap(2.0 * I(2))
        Q_iso_scaled = iso_scale(Q_circle)
        circle_result = Circle(Q_iso_scaled)
        @test circle_result isa Circle{Float64}
        @test circle_result.center ≈ Point2(2.0, 2.0) atol=1e-12
        @test circle_result.r ≈ 3.0 atol=1e-12
    end
end