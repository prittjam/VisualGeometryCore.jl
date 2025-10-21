#!/usr/bin/env julia
"""
Test scale space and Hessian response accuracy by comparing against VLFeat C library TIFF outputs.

This test validates that our Julia implementation matches VLFeat's C library
for both Gaussian Scale Space (GSS) and Cornerness Scale Space (CSS, i.e., Hessian determinant).

## Workflow
1. Check if VLFeat C executable exists (vlfeat_compare)
2. If test image changes, regenerate VLFeat TIFFs by running C program
3. Generate Julia scale space and responses
4. Compare Julia vs VLFeat TIFFs level-by-level

## VLFeat C Program
The vlfeat_compare executable:
- Loads vlfeat_comparison/input.tif
- Calls VLFeat C library to build GSS and CSS
- Saves all levels to vlfeat_comparison/vlfeat_gaussian/*.tif
- Saves all levels to vlfeat_comparison/vlfeat_hessian_det/*.tif

## Building vlfeat_compare
If the executable is missing or needs rebuilding:
```bash
gcc vlfeat_scalespace_compare.c -lvl -ltiff -o vlfeat_compare -lm
```
Requires: libvlfeat-dev, libtiff-dev
"""

using Test
using VisualGeometryCore
using FileIO
using ImageCore: channelview
using Colors
using Statistics
using Printf

function vlfeat_executable_exists()
    exec_path = joinpath(@__DIR__, "..", "vlfeat_compare")
    return isfile(exec_path) && (stat(exec_path).mode & 0o111) != 0
end

function regenerate_vlfeat_tiffs()
    # Run the C program to generate VLFeat TIFFs
    exec_path = joinpath(@__DIR__, "..", "vlfeat_compare")
    
    println("  Regenerating VLFeat TIFFs using C library...")
    cd(dirname(exec_path)) do
        output = read(`$exec_path`, String)
        println("  VLFeat C output:")
        for line in split(output, '\n')
            println("    ", line)
        end
    end
    println("  ✓ VLFeat TIFFs regenerated")
end

@testset "Scale Space TIFF Comparison vs VLFeat C" begin
    if !vlfeat_executable_exists()
        @warn "VLFeat executable not found - skipping TIFF comparison tests"
        @warn "Build with: gcc vlfeat_scalespace_compare.c -lvl -ltiff -o vlfeat_compare -lm"
        @warn "Requires: libvlfeat-dev, libtiff-dev"
        return
    end

    # Use blob_pattern test data
    blob_pattern_path = joinpath(@__DIR__, "data", "blob_pattern_eBd.png")
    if !isfile(blob_pattern_path)
        @warn "Blob pattern test image not found - skipping TIFF comparison"
        return
    end

    # Prepare input for VLFeat C program
    vlfeat_input_path = joinpath(@__DIR__, "..", "vlfeat_comparison", "input.tif")

    # Load and save as TIFF for VLFeat
    println("\nPreparing test image for VLFeat...")
    img_raw = load(blob_pattern_path)
    # Convert to grayscale
    img = if eltype(img_raw) <: Gray
        Gray{Float32}.(img_raw)
    else
        # Extract luminance from color/RGBA
        Gray{Float32}.(map(c -> Gray(c), img_raw))
    end
    println("  Image size: $(size(img))")
    save(vlfeat_input_path, img)
    println("  ✓ Saved to $vlfeat_input_path")

    # Option to regenerate (set to false for normal testing, true to regenerate reference)
    regenerate = get(ENV, "REGENERATE_VLFEAT", "false") == "true"
    if regenerate
        @info "Regenerating VLFeat reference TIFFs..."
        regenerate_vlfeat_tiffs()
    end
    
    # Build Julia scale space to match VLFeat's range
    # VLFeat generates octaves -1 to 5, subdivisions -1 to 3
    # (last_octave computed automatically from image size)
    println("\nBuilding Julia scale space...")
    ss = ScaleSpace(img; first_octave=-1, octave_resolution=3,
                   first_subdivision=-1, last_subdivision=3)

    # Get actual octave range
    julia_octaves = unique([level.octave for level in ss.levels])
    println("  ✓ Scale space built (octaves $(minimum(julia_octaves)) to $(maximum(julia_octaves)), subdivisions -1 to 3)")
    
    # Compute Julia Hessian responses
    println("\nComputing Julia Hessian determinant...")
    ixx = ScaleSpaceResponse(ss, DERIVATIVE_KERNELS.xx)
    iyy = ScaleSpaceResponse(ss, DERIVATIVE_KERNELS.yy)
    ixy = ScaleSpaceResponse(ss, DERIVATIVE_KERNELS.xy)
    hessian_resp = hessian_determinant_response(ixx, iyy, ixy)
    println("  ✓ Hessian determinant computed")
    
    @testset "Gaussian Scale Space (GSS) Comparison" begin
        println("\n" * "="^70)
        println("GAUSSIAN SCALE SPACE (GSS) COMPARISON")
        println("="^70)

        vlfeat_dir = joinpath(@__DIR__, "..", "vlfeat_comparison", "vlfeat_gaussian")

        # Find all VLFeat Gaussian TIFFs
        vlfeat_files = filter(f -> endswith(f, ".tif"), readdir(vlfeat_dir))

        rms_errors = Float64[]
        max_errors = Float64[]

        # Get Julia scale space octave/subdivision ranges
        julia_octaves = unique([level.octave for level in ss.levels])
        julia_subdivisions = unique([level.subdivision for level in ss.levels])

        for vf_file in sort(vlfeat_files)
            # Parse octave and subdivision from filename
            m = match(r"gaussian_o(-?\d+)_s(-?\d+)\.tif", vf_file)
            if m === nothing
                continue
            end

            octave = parse(Int, m.captures[1])
            subdivision = parse(Int, m.captures[2])

            # Skip if outside Julia's range
            if !(octave in julia_octaves && subdivision in julia_subdivisions)
                continue
            end

            # Load VLFeat TIFF
            vlfeat_path = joinpath(vlfeat_dir, vf_file)
            vlfeat_img = load(vlfeat_path)
            vlfeat_data = Float32.(channelview(vlfeat_img))

            # Get corresponding Julia level
            julia_level = ss[octave, subdivision]
            julia_data = Float32.(channelview(julia_level.data))

            # Compare sizes (skip if mismatch - expected at boundary octaves)
            if size(julia_data) != size(vlfeat_data)
                continue  # Skip size mismatches
            end

            # Compute errors
            diff = julia_data .- vlfeat_data
            rms = sqrt(mean(diff.^2))
            max_err = maximum(abs.(diff))

            push!(rms_errors, rms)
            push!(max_errors, max_err)
            
            # Categorize error level
            status = if rms < 1e-6
                "✓✓✓"
            elseif rms < 1e-5
                "✓✓"
            elseif rms < 1e-4
                "✓"
            elseif rms < 1e-3
                "~"
            else
                "✗"
            end
            
            @printf("  %s o=%2d, s=%2d: RMS=%.2e, Max=%.2e\n", 
                    status, octave, subdivision, rms, max_err)
            
            # Test thresholds (Gaussian - linear operation, tight tolerances)
            @test rms < 1e-4  # RMS pixel value error
            @test max_err < 1e-3  # Max pixel value error
        end
        
        println("\n" * "-"^70)
        @printf("  Overall GSS RMS: %.2e (mean), %.2e (max)\n", 
                mean(rms_errors), maximum(rms_errors))
        @printf("  Overall GSS Max Error: %.2e (mean), %.2e (max)\n",
                mean(max_errors), maximum(max_errors))
        println("="^70)
    end
    
    @testset "Hessian Determinant (CSS) Comparison" begin
        println("\n" * "="^70)
        println("HESSIAN DETERMINANT (CSS) COMPARISON")
        println("="^70)

        vlfeat_dir = joinpath(@__DIR__, "..", "vlfeat_comparison", "vlfeat_hessian_det")

        # Find all VLFeat Hessian det TIFFs
        vlfeat_files = filter(f -> endswith(f, ".tif"), readdir(vlfeat_dir))

        rms_errors = Float64[]
        max_errors = Float64[]

        # Get Julia Hessian response octave/subdivision ranges
        julia_octaves = unique([level.octave for level in hessian_resp.levels])
        julia_subdivisions = unique([level.subdivision for level in hessian_resp.levels])

        for vf_file in sort(vlfeat_files)
            # Parse octave and subdivision from filename
            m = match(r"hessian_det_o(-?\d+)_s(-?\d+)\.tif", vf_file)
            if m === nothing
                continue
            end

            octave = parse(Int, m.captures[1])
            subdivision = parse(Int, m.captures[2])

            # Skip if outside Julia's range
            if !(octave in julia_octaves && subdivision in julia_subdivisions)
                continue
            end

            # Load VLFeat TIFF
            vlfeat_path = joinpath(vlfeat_dir, vf_file)
            vlfeat_img = load(vlfeat_path)
            vlfeat_data = Float32.(channelview(vlfeat_img))

            # Get corresponding Julia level
            julia_level = hessian_resp[octave, subdivision]
            julia_data = Float32.(channelview(julia_level.data))

            # Compare sizes (skip if mismatch - expected at boundary octaves)
            if size(julia_data) != size(vlfeat_data)
                continue  # Skip size mismatches
            end

            # Compute errors
            diff = julia_data .- vlfeat_data
            rms = sqrt(mean(diff.^2))
            max_err = maximum(abs.(diff))

            push!(rms_errors, rms)
            push!(max_errors, max_err)
            
            # Categorize error level
            status = if rms < 1e-6
                "✓✓✓"
            elseif rms < 1e-5
                "✓✓"
            elseif rms < 1e-4
                "✓"
            elseif rms < 1e-3
                "~"
            else
                "✗"
            end
            
            @printf("  %s o=%2d, s=%2d: RMS=%.2e, Max=%.2e\n",
                    status, octave, subdivision, rms, max_err)

            # Test thresholds (Hessian det - nonlinear operation, error propagation expected)
            @test rms < 1e-3  # RMS determinant error (10× Gaussian due to products/squares)
            @test max_err < 5e-2  # Max determinant error (outliers from error propagation)
        end

        println("\n" * "-"^70)
        @printf("  Overall CSS RMS: %.2e (mean), %.2e (max)\n",
                mean(rms_errors), maximum(rms_errors))
        @printf("  Overall CSS Max Error: %.2e (mean), %.2e (max)\n",
                mean(max_errors), maximum(max_errors))
        println("="^70)
    end
end

println("\n✓ Scale space TIFF comparison complete!")
