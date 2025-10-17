"""
Validate Julia Hessian determinant responses against VLFeat CSS (Cornerness Scale Space).

VLFeat's CSS for Hessian detector contains the Hessian determinant responses.
"""

using VisualGeometryCore
using FileIO
using ImageCore: channelview
using Colors
using Statistics
using Printf

println("="^70)
println("Step 1: Create Julia scale space with Hessian detector settings")
println("="^70)

# Load input image
img = load("vlfeat_comparison/input.tif")

# Create scale space with VLFeat Hessian detector settings
ss = ScaleSpace(img; first_octave=-1, octave_resolution=3,
               first_subdivision=-1, last_subdivision=3)

octave_range, subdivision_range = axes(ss)
println("✓ Created ScaleSpace:")
println("  Octaves: $(first(octave_range)) to $(last(octave_range))")
println("  Subdivisions: $(first(subdivision_range)) to $(last(subdivision_range))")
println("  Total levels: $(length(ss))")
println()

println("="^70)
println("Step 2: Compute and save VLFeat-compatible Hessian determinant")
println("="^70)

# Create output directory
mkpath("vlfeat_comparison/julia_hessian_det")

# Compute determinant for each level using VLFeat-compatible function
println("Computing VLFeat-compatible Hessian determinant for each level...")
for level in ss
    # Compute step parameter: 2^octave (0.5 for octave -1, 1.0 for octave 0, etc.)
    step = 2.0^level.octave

    # Use VLFeat-compatible function with scale normalization
    det_data = vlfeat_hessian_det(level.data, level.sigma, step)

    # Save as TIFF
    filename = "vlfeat_comparison/julia_hessian_det/hessian_det_o$(level.octave)_s$(level.subdivision).tif"
    save(filename, Gray{Float32}.(det_data))

    min_val, max_val = extrema(det_data)
    println("  Saved: o=$(level.octave), s=$(level.subdivision), σ=$(round(level.sigma, digits=3)), step=$(step), range=[$min_val, $max_val]")
end

println()
println("✓ Saved $(length(ss)) Hessian determinant images with VLFeat scale normalization")
println()

println("="^70)
println("Step 3: Compare with VLFeat CSS (Hessian determinant)")
println("="^70)

# Check if VLFeat outputs exist
vlfeat_files = readdir("vlfeat_comparison/vlfeat_hessian_det", join=true)
if isempty(vlfeat_files)
    println("⚠ VLFeat Hessian determinant outputs not found!")
    println("The C program should have saved them to vlfeat_comparison/vlfeat_hessian_det/")
    exit(1)
end

rms_errors = Float64[]
max_errors = Float64[]

for o in -1:2
    for s in -1:3
        # Load VLFeat CSS (Hessian determinant)
        vlfeat_file = "vlfeat_comparison/vlfeat_hessian_det/hessian_det_o$(o)_s$(s).tif"
        if !isfile(vlfeat_file)
            continue
        end

        vlfeat_img = load(vlfeat_file)
        vlfeat_data = Float32.(channelview(vlfeat_img))

        # Load Julia Hessian determinant
        julia_file = "vlfeat_comparison/julia_hessian_det/hessian_det_o$(o)_s$(s).tif"
        julia_img = load(julia_file)
        julia_data = Float32.(channelview(julia_img))

        # Check dimensions match
        if size(julia_data) != size(vlfeat_data)
            println("  ✗ Size mismatch at o=$o, s=$s: Julia=$(size(julia_data)), VLFeat=$(size(vlfeat_data))")
            continue
        end

        # Exclude 1-pixel border (VLFeat only computes interior pixels)
        h, w = size(julia_data)
        julia_interior = julia_data[2:h-1, 2:w-1]
        vlfeat_interior = vlfeat_data[2:h-1, 2:w-1]

        # Compute errors on interior pixels only
        diff = julia_interior .- vlfeat_interior
        rms = sqrt(mean(diff.^2))
        max_err = maximum(abs.(diff))

        push!(rms_errors, rms)
        push!(max_errors, max_err)

        # Show per-level results
        status = rms < 1e-6 ? "✓✓✓" : (rms < 1e-5 ? "✓✓" : (rms < 1e-4 ? "✓" : (rms < 0.001 ? "~" : "✗")))
        @printf("%s o=%2d, s=%2d: RMS=%.6e, Max=%.6e\n", status, o, s, rms, max_err)
    end
end

println()
println("="^70)
println("Validation Summary:")
println("="^70)
println("Compared: $(length(rms_errors)) levels")

if !isempty(rms_errors)
    overall_rms = sqrt(mean(rms_errors.^2))
    mean_rms = mean(rms_errors)
    max_rms = maximum(rms_errors)
    overall_max = maximum(max_errors)

    @printf("Overall RMS: %.6e\n", overall_rms)
    @printf("Mean RMS:    %.6e\n", mean_rms)
    @printf("Max RMS:     %.6e\n", max_rms)
    @printf("Max error:   %.6e\n", overall_max)
    println()

    if overall_rms < 1e-6
        println("✓✓✓ EXCELLENT: RMS error < 1e-6 (TARGET ACHIEVED!)")
    elseif overall_rms < 1e-5
        println("✓✓ VERY GOOD: RMS error < 1e-5")
    elseif overall_rms < 1e-4
        println("✓ GOOD: RMS error < 1e-4")
    else
        println("✗ NEEDS INVESTIGATION: RMS error ≥ 1e-4")
    end
else
    println("\n⚠ No comparisons performed - check that VLFeat outputs exist")
end
