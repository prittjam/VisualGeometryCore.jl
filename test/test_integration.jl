#!/usr/bin/env julia

using Test
using VisualGeometryCore
using StaticArrays
using LinearAlgebra
using GeometryBasics
using CoordinateTransformations
using Rotations

@testset "VisualGeometryCore Integration" begin
    
    @testset "Basic Ellipse Operations" begin
        # Test ellipse creation and roundtrip
        ellipse = Ellipse(Point2(1.0, 2.0), 3.0, 2.0, π/4)
        @test ellipse isa Ellipse{Float64}
        @test ellipse.center == Point2(1.0, 2.0)
        @test ellipse.a == 3.0
        @test ellipse.b == 2.0

        # Test ellipse to conic conversion
        conic = HomEllipseMat(ellipse)
        @test conic isa HomEllipseMat{Float64}

        # Test conic to ellipse conversion
        recovered = Ellipse(conic)
        @test recovered isa Ellipse{Float64}

        # Check roundtrip accuracy
        @test norm(ellipse.center - recovered.center) < 1e-12
        @test abs(ellipse.a - recovered.a) < 1e-12
        @test abs(ellipse.b - recovered.b) < 1e-12
    end
    
    @testset "Transform Operations" begin
        # Test basic transform creation
        rot = to_homogeneous(RotMatrix{2}(π/4))
        @test rot isa HomRotMat{Float64}
        
        trans = to_homogeneous(Translation(SVector(2.0, 1.0)))
        @test trans isa HomTransMat{Float64}
        
        # Test transform multiplication
        euclidean = rot * trans
        @test euclidean isa EuclideanMat{Float64}
    end
    
    @testset "GeometryBasics Integration" begin
        ellipse = Ellipse(Point2(0.0, 0.0), 2.0, 1.0, 0.0)

        # Test interface methods
        @test ellipse.center == Point2(0.0, 0.0)
        @test radius(ellipse) == 2.0

        # Test point generation via coordinates
        points = GeometryBasics.coordinates(ellipse, 8)
        @test length(points) == 8
        @test all(p -> p isa Point{2,Float64}, points)

        # Test that decompose also works (calls coordinates internally)
        points2 = GeometryBasics.decompose(Point{2,Float64}, ellipse)
        @test length(points2) == 32  # default nvertices
        @test all(p -> p isa Point{2,Float64}, points2)

        # Verify points lie on ellipse
        conic = HomEllipseMat(ellipse)
        for p in points
            homog_p = SVector(p[1], p[2], 1.0)
            error = abs(homog_p' * SMatrix{3,3,Float64}(conic) * homog_p)
            @test error < 1e-14
        end
    end
    
    @testset "Conic Transformations" begin
        ellipse = Ellipse(Point2(0.0, 0.0), 3.0, 2.0, 0.0)
        conic = HomEllipseMat(ellipse)

        # Test translation using homography
        H = to_homogeneous(Translation(SVector(5.0, -2.0)))
        H_mat = PlanarHomographyMat{Float64}(Tuple(SMatrix{3,3,Float64}(H)))
        translated_conic = H_mat(conic)
        translated_ellipse = Ellipse(translated_conic)

        @test norm(translated_ellipse.center - Point2(5.0, -2.0)) < 1e-12
        @test abs(translated_ellipse.a - ellipse.a) < 1e-12
        @test abs(translated_ellipse.b - ellipse.b) < 1e-12
    end
end