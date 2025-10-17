"""
Test Julia ScaleSpace with VLFeat Hessian detector settings.

VLFeat Hessian detector uses:
- first_octave = -1
- octave_resolution = 3
- first_subdivision = -1
- last_subdivision = 3 (octaveResolution)
"""

using VisualGeometryCore
using FileIO
using ImageCore: channelview
using Colors
using Statistics
using Printf

# Load input image
img = load("vlfeat_comparison/input.tif")

# Create scale space with VLFeat Hessian detector settings
println("Creating Julia scale space with Hessian detector settings:")
println("  first_octave = -1")
println("  octave_resolution = 3")
println("  first_subdivision = -1")
println("  last_subdivision = 3")
println()

ss = ScaleSpace(img; first_octave=-1, octave_resolution=3,
               first_subdivision=-1, last_subdivision=3)

println("Julia ScaleSpace created successfully!")
println("Geometry:")
octave_range, subdivision_range = axes(ss)
println("  Octaves: $(first(octave_range)) to $(last(octave_range))")
println("  Subdivisions: $(first(subdivision_range)) to $(last(subdivision_range))")
println("  Total levels: $(length(ss))")
println()

# Compare all overlapping levels
println("="^70)
println("Comparing with VLFeat outputs:")
println("="^70)

rms_errors = Float64[]
max_errors = Float64[]

for o in -1:2
    for s in -1:3
        # Try to load VLFeat level
        vlfeat_file = "vlfeat_comparison/vlfeat_gaussian/gaussian_o$(o)_s$(s).tif"
        if !isfile(vlfeat_file)
            continue
        end

        vlfeat_img = load(vlfeat_file)
        vlfeat_data = Float32.(channelview(vlfeat_img))

        # Get Julia level
        julia_level = ss[o, s]
        julia_data = Float32.(channelview(julia_level.data))

        # Compare
        diff = julia_data .- vlfeat_data
        rms = sqrt(mean(diff.^2))
        max_err = maximum(abs.(diff))

        push!(rms_errors, rms)
        push!(max_errors, max_err)

        status = rms < 1e-6 ? "✓✓✓" : (rms < 1e-5 ? "✓✓" : (rms < 1e-4 ? "✓" : (rms < 0.001 ? "~" : "✗")))
        @printf("%s o=%2d, s=%2d: RMS=%.6e, Max=%.6e\n", status, o, s, rms, max_err)
    end
end

println()
println("="^70)
println("Summary:")
println("="^70)
println("Levels compared: $(length(rms_errors))")

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
end
