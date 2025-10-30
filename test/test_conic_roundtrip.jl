#!/usr/bin/env julia
# Test round-trip conversions for Circle and Ellipse conic matrices

using Test
using VisualGeometryCore
using GeometryBasics: Point2, Circle
using LinearAlgebra

@testset "Conic Round-Trip Conversions" begin

    @testset "Circle → HomCircleMat → Circle" begin
        # Test various circles
        circles = [
            Circle(Point2(0.0, 0.0), 1.0),          # Unit circle at origin
            Circle(Point2(5.0, 3.0), 2.5),          # Offset circle
            Circle(Point2(-10.0, 20.0), 7.3),       # Negative coordinates
            Circle(Point2(100.0, 200.0), 50.0)      # Large circle
        ]

        for circle_orig in circles
            # Forward: Circle → HomCircleMat
            Q = HomCircleMat(circle_orig)
            @test Q isa HomCircleMat{Float64}
            @test conic_trait(Q) isa CircleTrait

            # Backward: HomCircleMat → Circle
            circle_recovered = Circle(Q)
            @test circle_recovered isa Circle{Float64}

            # Check round-trip accuracy
            @test isapprox(circle_recovered.center[1], circle_orig.center[1], atol=1e-10)
            @test isapprox(circle_recovered.center[2], circle_orig.center[2], atol=1e-10)
            @test isapprox(circle_recovered.r, circle_orig.r, atol=1e-10)

            # Verify conic equation: points on circle satisfy x'Qx = 0
            θ_samples = range(0, 2π, length=16)
            for θ in θ_samples
                p = circle_orig.center .+ circle_orig.r .* [cos(θ), sin(θ)]
                p_hom = [p[1], p[2], 1.0]
                residual = abs(dot(p_hom, Q * p_hom))
                @test residual < 1e-12
            end
        end
    end

    @testset "Ellipse → HomEllipseMat → Ellipse" begin
        # Test various ellipses
        ellipses = [
            Ellipse(Point2(0.0, 0.0), 2.0, 1.0, 0.0),         # Axis-aligned at origin
            Ellipse(Point2(5.0, 3.0), 4.0, 2.0, π/4),         # Rotated 45°
            Ellipse(Point2(-10.0, 20.0), 7.0, 3.5, π/6),      # Rotated 30°
            Ellipse(Point2(100.0, 200.0), 50.0, 30.0, π/3)    # Large rotated
        ]

        for ellipse_orig in ellipses
            # Forward: Ellipse → HomEllipseMat
            Q = HomEllipseMat(ellipse_orig)
            @test Q isa HomEllipseMat{Float64}
            @test conic_trait(Q) isa EllipseTrait

            # Backward: HomEllipseMat → Ellipse
            ellipse_recovered = Ellipse(Q)
            @test ellipse_recovered isa Ellipse{Float64}

            # Check round-trip accuracy
            @test isapprox(ellipse_recovered.center[1], ellipse_orig.center[1], atol=1e-10)
            @test isapprox(ellipse_recovered.center[2], ellipse_orig.center[2], atol=1e-10)
            @test isapprox(ellipse_recovered.a, ellipse_orig.a, atol=1e-10)
            @test isapprox(ellipse_recovered.b, ellipse_orig.b, atol=1e-10)

            # Angle comparison needs to handle wrapping (θ is periodic with period π)
            θ_diff = abs(mod(ellipse_recovered.θ - ellipse_orig.θ, π))
            θ_diff = min(θ_diff, π - θ_diff)  # Handle wraparound
            @test θ_diff < 1e-10

            # Verify conic equation: points on ellipse satisfy x'Qx = 0
            θ_samples = range(0, 2π, length=16)
            for θ in θ_samples
                # Parametric point on ellipse
                cos_angle = cos(ellipse_orig.θ)
                sin_angle = sin(ellipse_orig.θ)
                x_local = ellipse_orig.a * cos(θ)
                y_local = ellipse_orig.b * sin(θ)
                x = ellipse_orig.center[1] + cos_angle * x_local - sin_angle * y_local
                y = ellipse_orig.center[2] + sin_angle * x_local + cos_angle * y_local
                p_hom = [x, y, 1.0]
                residual = abs(dot(p_hom, Q * p_hom))
                @test residual < 1e-11
            end
        end
    end

    @testset "HomEllipseMat(::Circle) delegation" begin
        circle = Circle(Point2(3.0, 4.0), 5.0)

        # HomEllipseMat(circle) should return HomCircleMat
        Q = HomEllipseMat(circle)
        @test Q isa HomCircleMat{Float64}
        @test conic_trait(Q) isa CircleTrait

        # Should round-trip correctly
        circle_recovered = Circle(Q)
        @test isapprox(circle_recovered.center[1], circle.center[1], atol=1e-10)
        @test isapprox(circle_recovered.center[2], circle.center[2], atol=1e-10)
        @test isapprox(circle_recovered.r, circle.r, atol=1e-10)
    end

    @testset "Circle → HomCircleMat → Ellipse → Circle" begin
        circle = Circle(Point2(2.0, 3.0), 4.0)

        # Circle → HomCircleMat
        Q = HomCircleMat(circle)
        @test Q isa HomCircleMat{Float64}

        # HomCircleMat → Ellipse (should have a ≈ b ≈ r)
        ellipse = Ellipse(Q)
        @test ellipse isa Ellipse{Float64}
        @test isapprox(ellipse.a, ellipse.b, atol=1e-10)
        @test isapprox(ellipse.a, circle.r, atol=1e-10)
        @test isapprox(ellipse.center[1], circle.center[1], atol=1e-10)
        @test isapprox(ellipse.center[2], circle.center[2], atol=1e-10)

        # Ellipse → HomEllipseMat
        Q2 = HomEllipseMat(ellipse)
        @test Q2 isa HomEllipseMat{Float64}

        # HomEllipseMat → Circle (should work since a ≈ b)
        circle_recovered = Circle(Q2)
        @test circle_recovered isa Circle{Float64}
        @test isapprox(circle_recovered.center[1], circle.center[1], atol=1e-10)
        @test isapprox(circle_recovered.center[2], circle.center[2], atol=1e-10)
        @test isapprox(circle_recovered.r, circle.r, atol=1e-10)
    end
end
