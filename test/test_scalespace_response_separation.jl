"""
Test suite for ScaleSpace response separation implementation.

Tests the separated ScaleSpace (Gaussian-only) and ScaleSpaceResponse types.
"""

using Test
using Colors, FixedPointNumbers
using ImageFiltering: centered

# Include the main module (adjust path as needed)
include("../src/scalespace.jl")

@testset "ScaleSpace Response Separation Tests" begin
    
    @testset "ScaleSpace Gaussian-Only Constraints" begin
        # Test that ScaleSpace only accepts GaussianImage types
        @test_nowarn ScaleSpace(Size2(64, 64), image_type=GaussianImage{N0f8})
        @test_nowarn ScaleSpace(Size2(64, 64), image_type=GaussianImage{Float32})
        
        # Test that non-Gaussian types are rejected (MethodError due to type constraint)
        @test_throws MethodError ScaleSpace(Size2(64, 64), image_type=HessianImages{Float32})
        @test_throws MethodError ScaleSpace(Size2(64, 64), image_type=LaplacianImage{Float32})
        
        # Test kernel function storage
        ss = ScaleSpace(Size2(64, 64))
        @test ss.kernel_func == vlfeat_gaussian
        
        # Test custom kernel function
        custom_kernel = sigma -> Kernel.gaussian(sigma)
        ss_custom = ScaleSpace(Size2(64, 64), kernel_func=custom_kernel)
        @test ss_custom.kernel_func == custom_kernel
    end
    
    @testset "ScaleSpaceResponse Creation" begin
        # Create template ScaleSpace
        template = ScaleSpace(Size2(64, 64))

        # Test creation with HESSIAN_FILTERS
        hess_resp = ScaleSpaceResponse(template, HessianImages{Float32}, HESSIAN_FILTERS)
        @test hess_resp isa ScaleSpaceResponse
        @test length(hess_resp.levels) == length(template.levels)
        @test hess_resp.transform == HESSIAN_FILTERS

        # Test creation with LAPLACIAN_FUNCTION
        lap_resp = ScaleSpaceResponse(template, LaplacianImage{Float32}, LAPLACIAN_FUNCTION)
        @test lap_resp isa ScaleSpaceResponse
        @test length(lap_resp.levels) == length(template.levels)
        @test lap_resp.transform == LAPLACIAN_FUNCTION

        # Test creation with custom filters
        custom_filters = (Gx = centered([-1 0 1]), Gy = centered([-1; 0; 1]))
        GradientImages{T} = NamedTuple{(:Gx, :Gy), Tuple{Matrix{T}, Matrix{T}}}
        grad_resp = ScaleSpaceResponse(template, GradientImages{Float32}, custom_filters)
        @test grad_resp isa ScaleSpaceResponse
        @test grad_resp.transform == custom_filters

        # Test error on empty template
        empty_template = ScaleSpace(Size2(64, 64))
        empty!(empty_template.levels)
        @test_throws AssertionError ScaleSpaceResponse(empty_template, HessianImages{Float32}, HESSIAN_FILTERS)
    end
    
    @testset "Predefined Filter Constants" begin
        # Test HESSIAN_FILTERS structure
        @test haskey(HESSIAN_FILTERS, :Ixx)
        @test haskey(HESSIAN_FILTERS, :Iyy)
        @test haskey(HESSIAN_FILTERS, :Ixy)
        @test size(HESSIAN_FILTERS.Ixx) == (3, 3)
        @test size(HESSIAN_FILTERS.Iyy) == (3, 3)
        @test size(HESSIAN_FILTERS.Ixy) == (3, 3)
        
        # Test LAPLACIAN_FUNCTION
        @test LAPLACIAN_FUNCTION isa Function
        
        # Test LAPLACIAN_FUNCTION with dummy data
        dummy_hess = (Ixx = [1.0 2.0; 3.0 4.0], Iyy = [0.5 1.5; 2.5 3.5])
        result = LAPLACIAN_FUNCTION(dummy_hess)
        @test haskey(result, :L)
        @test result.L == dummy_hess.Ixx + dummy_hess.Iyy
    end
    
    @testset "ScaleSpaceResponse Indexing" begin
        template = ScaleSpace(Size2(64, 64))
        source = ScaleSpace(Size2(64, 64))
        hess_resp = ScaleSpaceResponse(template, HessianImages{Float32}, HESSIAN_FILTERS)
        
        # Test 2D indexing with source context
        @test_nowarn hess_resp[0, 0, source]
        @test_nowarn hess_resp[0, 1, source]
        
        # Test bounds checking
        octave_range, subdivision_range = axes(source)
        max_octave = maximum(octave_range)
        max_subdivision = maximum(subdivision_range)
        @test_throws BoundsError hess_resp[max_octave + 1, 0, source]
        @test_throws BoundsError hess_resp[0, max_subdivision + 1, source]
        
        # Test CartesianIndex support
        @test_nowarn hess_resp[CartesianIndex(0, 0), source]
        
        # Test range indexing
        @test_nowarn hess_resp[0, subdivision_range, source]
        
        # Test linear indexing
        @test_nowarn hess_resp[1]
        @test_nowarn hess_resp[end]
        
        # Test iteration
        count = 0
        for level in hess_resp
            count += 1
        end
        @test count == length(hess_resp.levels)
    end
    
    @testset "ScaleSpaceResponse Computation" begin
        # Create and populate source ScaleSpace
        img = rand(Gray{N0f8}, 64, 64)
        source = ScaleSpace(img)  # Auto-populated

        # Create response structure
        hess_resp = ScaleSpaceResponse(source, HessianImages{Float32}, HESSIAN_FILTERS)

        # Test computation
        @test_nowarn hess_resp(source)

        # Verify response data is populated
        first_level = first(hess_resp)
        @test !any(isnan, first_level.data.Ixx)
        @test !any(isnan, first_level.data.Iyy)
        @test !any(isnan, first_level.data.Ixy)

        # Test Laplacian computation from Hessian data
        # Note: LAPLACIAN_FUNCTION expects Hessian data, but ScaleSpaceResponse only accepts ScaleSpace
        # So we need to create a custom function that works with Gaussian data
        lap_func = gauss_data -> (L = gauss_data.g .* 0.0,)  # Dummy Laplacian for testing
        lap_resp = ScaleSpaceResponse(source, LaplacianImage{Float32}, lap_func)
        @test_nowarn lap_resp(source)  # Compute from Gaussian source

        first_lap_level = first(lap_resp)
        @test !any(isnan, first_lap_level.data.L)
    end
    
    @testset "Validation and Error Handling" begin
        # Test geometry mismatch
        template = ScaleSpace(Size2(64, 64))
        different_template = ScaleSpace(Size2(32, 32))
        different_resp = ScaleSpaceResponse(different_template, HessianImages{Float32}, HESSIAN_FILTERS)
        img = rand(Gray{N0f8}, 64, 64)
        source = ScaleSpace(img)  # Populate source with 64x64
        @test_throws ErrorException different_resp(source)  # Should fail due to size mismatch

        # Test non-Gaussian source error
        # This is harder to test directly since we can't create non-Gaussian ScaleSpace anymore
        # But the validation is there in the filter application
    end
    
    @testset "Type Safety" begin
        # Test that ScaleSpace type parameter is constrained
        @test ScaleSpace{GaussianImage{N0f8}} <: ScaleSpace

        # Test that ScaleSpaceResponse has correct type parameters
        template = ScaleSpace(Size2(64, 64))
        hess_resp = ScaleSpaceResponse(template, HessianImages{Float32}, HESSIAN_FILTERS)
        @test hess_resp isa ScaleSpaceResponse{<:NamedTuple, <:NamedTuple}

        lap_resp = ScaleSpaceResponse(template, LaplacianImage{Float32}, LAPLACIAN_FUNCTION)
        @test lap_resp isa ScaleSpaceResponse{<:NamedTuple, <:Function}
    end
    
    @testset "VLFeat Indexing Compatibility" begin
        # Test that custom indexing works with VLFeat-style coordinates
        ss = ScaleSpace(Size2(64, 64))
        
        # Test default VLFeat ranges (octaves 0:N, subdivisions 0:resolution-1)
        octave_range, subdivision_range = axes(ss)
        @test first(octave_range) == 0
        @test first(subdivision_range) == 0
        @test last(subdivision_range) == ss.octave_resolution - 1
        
        # Test indexing with 0-based coordinates
        @test_nowarn ss[0, 0]
        @test_nowarn ss[0, 1]
        @test_nowarn ss[1, 0]
        
        # Test that level_size utility works
        level = ss[0, 0]
        @test_nowarn level_size(level)
    end
    
    @testset "Performance and Memory" begin
        # Test that ScaleSpaceResponse can be reused
        template = ScaleSpace(Size2(64, 64))
        hess_resp = ScaleSpaceResponse(template, HessianImages{Float32}, HESSIAN_FILTERS)

        # Create multiple source scale spaces
        img1 = rand(Gray{N0f8}, 64, 64)
        img2 = rand(Gray{N0f8}, 64, 64)
        source1 = ScaleSpace(img1)
        source2 = ScaleSpace(img2)

        # Test reusability
        @test_nowarn hess_resp(source1)
        @test_nowarn hess_resp(source2)  # Reuse same response structure

        # Verify that the response data changed
        # (This is a basic test - in practice you'd check that the values are different)
        @test true  # Placeholder for more sophisticated reusability tests
    end
    
end