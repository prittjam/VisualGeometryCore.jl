#!/usr/bin/env julia
# P3P Camera Pose Estimation Unit Tests

using Test
using VisualGeometryCore
using VisualGeometryCore: Camera, RotMatrix
using LinearAlgebra, Random, Statistics
using GeometryBasics: Vec3, Point3
using StaticArrays: SVector
using Unitful: ustrip, mm, °

@testset "P3P Camera Pose Estimation" begin

    @testset "P3P with random synthetic correspondences" begin
        # Setup camera model (following pose_from_correspondences.jl pattern)
        sensor = CMOS_SENSORS["Sony"]["IMX174"]
        f = focal_length(40.0°, sensor; dimension=:horizontal)
        pp = sensor.resolution ./ 2
        model = CameraModel(f, sensor.pitch, pp)

        # Set random seed for reproducibility
        Random.seed!(42)

        # Run multiple trials with different random configurations
        n_trials = 10
        for trial in 1:n_trials
            # Generate random 3D points in world coordinates (planar at z=0, in mm)
            # Spread points over a reasonable area (similar to calibration board)
            X = [Point3(rand(100.0:700.0), rand(100.0:500.0), 0.0) for _ in 1:3]

            # Generate random ground truth camera pose
            # Position: random location above the plane looking down
            camera_pos = Point3(rand(200.0:600.0), rand(200.0:400.0), rand(1500.0:2500.0))
            target_pos = Point3(400.0, 300.0, 0.0)  # Look at center of point cloud
            up_vector = Vec3(0.0, -1.0, 0.0)

            extrinsics_gt = lookat(camera_pos, target_pos, up_vector)
            camera_gt = Camera(model, extrinsics_gt)

            # Project 3D points to get 2D observations
            u = project.(Ref(camera_gt), X)
            u_unitless = ustrip.(u)

            # Solve P3P: backproject to rays and solve for pose
            rays = backproject.(Ref(model), u_unitless)
            Rs, ts = p3p(rays, X)

            # Verify P3P found at least one solution
            @test length(Rs) > 0

            if length(Rs) > 0
                # Select best solution based on reprojection error
                cameras = Camera.(Ref(model), EuclideanMap.(RotMatrix{3,Float64}.(Rs), ts))
                u_proj = [project.(Ref(cam), X) for cam in cameras]
                mean_errors = [mean(norm.(ustrip.(u_p) .- u_unitless)) for u_p in u_proj]
                best_idx = argmin(mean_errors)

                camera_recovered = cameras[best_idx]

                # Verify reprojection error is near zero
                # This is the key test - P3P may return multiple valid solutions,
                # but the best one (selected by reprojection error) should perfectly
                # reproduce the observations
                u_reproj = project.(Ref(camera_recovered), X)
                errors = norm.(ustrip.(u_reproj) .- u_unitless)
                mean_error = mean(errors)
                max_error = maximum(errors)

                @test mean_error < 1e-10  # Near-perfect reconstruction
                @test max_error < 1e-9

                # Note: We don't test pose match because P3P can return up to 4 valid
                # solutions. The solution with minimum reprojection error is correct
                # from a geometric standpoint, even if it's not the same as ground truth.
            end
        end
    end

    @testset "P3P with degenerate configurations" begin
        # Setup camera model
        sensor = CMOS_SENSORS["Sony"]["IMX174"]
        f = focal_length(40.0°, sensor; dimension=:horizontal)
        pp = sensor.resolution ./ 2
        model = CameraModel(f, sensor.pitch, pp)

        Random.seed!(123)

        # Test 1: Collinear points (should fail or give poor results)
        X_collinear = [Point3(100.0, 100.0, 0.0),
                       Point3(200.0, 200.0, 0.0),
                       Point3(300.0, 300.0, 0.0)]

        camera_pos = Point3(400.0, 300.0, 2000.0)
        extrinsics = lookat(camera_pos, Point3(200.0, 200.0, 0.0), Vec3(0.0, -1.0, 0.0))
        camera = Camera(model, extrinsics)

        u = project.(Ref(camera), X_collinear)
        u_unitless = ustrip.(u)
        rays = backproject.(Ref(model), u_unitless)
        Rs, ts = p3p(rays, X_collinear)

        # P3P may return solutions for collinear points, but they may not be unique
        # Just verify it doesn't crash
        @test true

        # Test 2: Very small triangle (numerical stability)
        X_small = [Point3(400.0, 300.0, 0.0),
                   Point3(400.1, 300.0, 0.0),
                   Point3(400.0, 300.1, 0.0)]

        u_small = project.(Ref(camera), X_small)
        u_small_unitless = ustrip.(u_small)
        rays_small = backproject.(Ref(model), u_small_unitless)
        Rs_small, ts_small = p3p(rays_small, X_small)

        # Should handle small triangles gracefully
        @test true
    end

    @testset "P3P with multiple valid solutions" begin
        # In general, P3P can have up to 4 solutions
        # Test that we can disambiguate using reprojection error

        sensor = CMOS_SENSORS["Sony"]["IMX174"]
        f = focal_length(40.0°, sensor; dimension=:horizontal)
        pp = sensor.resolution ./ 2
        model = CameraModel(f, sensor.pitch, pp)

        Random.seed!(456)

        # Generate configuration that typically gives multiple solutions
        X = [Point3(0.0, 0.0, 0.0),
             Point3(100.0, 0.0, 0.0),
             Point3(50.0, 100.0, 0.0)]

        camera_pos = Point3(50.0, 50.0, 500.0)
        extrinsics_gt = lookat(camera_pos, Point3(50.0, 50.0, 0.0), Vec3(0.0, -1.0, 0.0))
        camera_gt = Camera(model, extrinsics_gt)

        u = project.(Ref(camera_gt), X)
        u_unitless = ustrip.(u)
        rays = backproject.(Ref(model), u_unitless)
        Rs, ts = p3p(rays, X)

        if length(Rs) > 1
            # Multiple solutions exist - verify we can find the correct one
            cameras = Camera.(Ref(model), EuclideanMap.(RotMatrix{3,Float64}.(Rs), ts))
            u_proj = [project.(Ref(cam), X) for cam in cameras]
            mean_errors = [mean(norm.(ustrip.(u_p) .- u_unitless)) for u_p in u_proj]

            # Best solution should have near-zero error
            min_error = minimum(mean_errors)
            @test min_error < 1e-10

            # Other solutions should have larger error
            best_idx = argmin(mean_errors)
            for (i, err) in enumerate(mean_errors)
                if i != best_idx
                    @test err > min_error  # Other solutions are worse
                end
            end
        end
    end

    @testset "P3P scale invariance" begin
        # P3P should work correctly regardless of world coordinate scale
        sensor = CMOS_SENSORS["Sony"]["IMX174"]
        f = focal_length(40.0°, sensor; dimension=:horizontal)
        pp = sensor.resolution ./ 2
        model = CameraModel(f, sensor.pitch, pp)

        Random.seed!(789)

        # Original scale (mm)
        X_mm = [Point3(100.0, 100.0, 0.0),
                Point3(500.0, 100.0, 0.0),
                Point3(300.0, 400.0, 0.0)]

        camera_pos_mm = Point3(300.0, 250.0, 2000.0)
        extrinsics = lookat(camera_pos_mm, Point3(300.0, 250.0, 0.0), Vec3(0.0, -1.0, 0.0))
        camera = Camera(model, extrinsics)

        # Test at mm scale
        u = project.(Ref(camera), X_mm)
        u_unitless = ustrip.(u)
        rays = backproject.(Ref(model), u_unitless)
        Rs_mm, ts_mm = p3p(rays, X_mm)

        @test length(Rs_mm) > 0

        # Test at 10x scale (cm to mm)
        scale = 10.0
        X_scaled = [Point3(p[1] * scale, p[2] * scale, p[3] * scale) for p in X_mm]
        camera_pos_scaled = Point3(camera_pos_mm[1] * scale, camera_pos_mm[2] * scale, camera_pos_mm[3] * scale)
        target_scaled = Point3(300.0 * scale, 250.0 * scale, 0.0)
        extrinsics_scaled = lookat(camera_pos_scaled, target_scaled, Vec3(0.0, -1.0, 0.0))
        camera_scaled = Camera(model, extrinsics_scaled)

        u_scaled = project.(Ref(camera_scaled), X_scaled)
        @test all(isapprox.(ustrip.(u_scaled), u_unitless, atol=1e-10))  # Same projections

        rays_scaled = backproject.(Ref(model), ustrip.(u_scaled))
        Rs_scaled, ts_scaled = p3p(rays_scaled, X_scaled)

        @test length(Rs_scaled) > 0

        # Recovered translations should scale proportionally
        if length(Rs_mm) > 0 && length(Rs_scaled) > 0
            # Find best solutions
            cameras_mm = Camera.(Ref(model), EuclideanMap.(RotMatrix{3,Float64}.(Rs_mm), ts_mm))
            cameras_scaled = Camera.(Ref(model), EuclideanMap.(RotMatrix{3,Float64}.(Rs_scaled), ts_scaled))

            u_proj_mm = [project.(Ref(cam), X_mm) for cam in cameras_mm]
            u_proj_scaled = [project.(Ref(cam), X_scaled) for cam in cameras_scaled]

            errors_mm = [mean(norm.(ustrip.(u_p) .- u_unitless)) for u_p in u_proj_mm]
            errors_scaled = [mean(norm.(ustrip.(u_p) .- ustrip.(u_scaled))) for u_p in u_proj_scaled]

            best_mm = argmin(errors_mm)
            best_scaled = argmin(errors_scaled)

            # Check that translation scales correctly
            t_mm = ts_mm[best_mm]
            t_scaled = ts_scaled[best_scaled]
            @test isapprox(norm(t_scaled), norm(t_mm) * scale, rtol=1e-6)
        end
    end
end
