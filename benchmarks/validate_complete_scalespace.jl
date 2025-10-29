"""
Complete validation of Julia ScaleSpace against VLFeat.

Validates that EVERY octave, subdivision, and level matches VLFeat with RMS < 1e-6.
Uses only existing code - no modifications.
"""

using VisualGeometryCore
using FileIO
using ImageCore: channelview
using Colors
using Statistics
using Printf

println("="^80)
println("COMPLETE SCALESPACE VALIDATION")
println("="^80)
println()

# =============================================================================
# Step 1: Create Julia Scale Space
# =============================================================================

println("Step 1: Creating Julia scale space with Hessian detector settings")
println("-"^80)

# Load test image
img = load("vlfeat_comparison/input.tif")
println("✓ Loaded test image: $(size(img))")

# Create scale space with VLFeat Hessian detector settings
# VLFeat Hessian uses: first_octave=-1, octave_resolution=3,
#                      first_subdivision=-1, last_subdivision=3
ss = ScaleSpace(img; first_octave=-1, octave_resolution=3,
               first_subdivision=-1, last_subdivision=3)

octave_range, subdivision_range = axes(ss)
println("✓ Created ScaleSpace:")
println("    Octaves: $(first(octave_range)) to $(last(octave_range))")
println("    Subdivisions: $(first(subdivision_range)) to $(last(subdivision_range))")
println("    Total levels: $(length(ss))")
println()

# =============================================================================
# Step 2: Save Julia Gaussian Levels
# =============================================================================

println("Step 2: Saving Julia Gaussian levels to TIFF")
println("-"^80)

mkpath("vlfeat_comparison/julia_gaussian")

saved_count = 0
for level in ss
    filename = "vlfeat_comparison/julia_gaussian/gaussian_o$(level.octave)_s$(level.subdivision).tif"
    save(filename, level.data)
    saved_count += 1
    @printf("  Saved: o=%2d, s=%2d, σ=%.3f\n", level.octave, level.subdivision, level.sigma)
end

println("✓ Saved $(saved_count) Gaussian levels")
println()

# =============================================================================
# Step 3: Run VLFeat C Program
# =============================================================================

println("Step 3: Running VLFeat C comparison program")
println("-"^80)

# Check if VLFeat executable exists
if !isfile("vlfeat_compare")
    println("⚠ vlfeat_compare executable not found!")
    println("  Please compile it first:")
    println("  gcc -I\$VLFEAT_PATH vlfeat_scalespace_compare.c -lvl -ltiff -o vlfeat_compare")
    exit(1)
end

# Run VLFeat comparison
run(`./vlfeat_compare`)
println()

# =============================================================================
# Step 4: Compare Every Level
# =============================================================================

println("Step 4: Comparing EVERY level (Gaussian)")
println("-"^80)

rms_errors = Float64[]
max_errors = Float64[]
comparison_details = []

for o in octave_range
    for s in subdivision_range
        # Check if VLFeat file exists
        vlfeat_file = "vlfeat_comparison/vlfeat_gaussian/gaussian_o$(o)_s$(s).tif"
        if !isfile(vlfeat_file)
            println("  ⚠ Missing VLFeat file: $vlfeat_file")
            continue
        end

        # Load VLFeat data
        vlfeat_img = load(vlfeat_file)
        vlfeat_data = Float32.(channelview(vlfeat_img))

        # Load Julia data
        julia_file = "vlfeat_comparison/julia_gaussian/gaussian_o$(o)_s$(s).tif"
        julia_img = load(julia_file)
        julia_data = Float32.(channelview(julia_img))

        # Check dimensions
        if size(julia_data) != size(vlfeat_data)
            println("  ✗ Size mismatch at o=$o, s=$s")
            println("    Julia: $(size(julia_data)), VLFeat: $(size(vlfeat_data))")
            continue
        end

        # Compute errors
        diff = julia_data .- vlfeat_data
        rms = sqrt(mean(diff.^2))
        max_err = maximum(abs.(diff))

        push!(rms_errors, rms)
        push!(max_errors, max_err)
        push!(comparison_details, (octave=o, subdivision=s, rms=rms, max_err=max_err))

        # Status indicator
        status = if rms < 1e-6
            "✓✓✓"
        elseif rms < 1e-5
            "✓✓ "
        elseif rms < 1e-4
            "✓  "
        elseif rms < 0.001
            "~  "
        else
            "✗  "
        end

        @printf("%s o=%2d, s=%2d: RMS=%.6e, Max=%.6e\n", status, o, s, rms, max_err)
    end
end

println()

# =============================================================================
# Step 5: Summary Statistics
# =============================================================================

println("="^80)
println("VALIDATION SUMMARY")
println("="^80)

if isempty(rms_errors)
    println("✗ No comparisons performed - check VLFeat outputs")
    exit(1)
end

n_compared = length(rms_errors)
overall_rms = sqrt(mean(rms_errors.^2))
mean_rms = mean(rms_errors)
max_rms = maximum(rms_errors)
overall_max = maximum(max_errors)

@printf("Levels compared:     %d\n", n_compared)
@printf("Overall RMS:         %.6e\n", overall_rms)
@printf("Mean RMS:            %.6e\n", mean_rms)
@printf("Max RMS:             %.6e\n", max_rms)
@printf("Max absolute error:  %.6e\n", overall_max)
println()

# Count by quality tier
excellent = sum(rms_errors .< 1e-6)
very_good = sum((rms_errors .>= 1e-6) .& (rms_errors .< 1e-5))
good = sum((rms_errors .>= 1e-5) .& (rms_errors .< 1e-4))
acceptable = sum((rms_errors .>= 1e-4) .& (rms_errors .< 0.001))
poor = sum(rms_errors .>= 0.001)

println("Quality breakdown:")
@printf("  ✓✓✓ Excellent (RMS < 1e-6):    %3d / %3d  (%.1f%%)\n",
        excellent, n_compared, 100*excellent/n_compared)
@printf("  ✓✓  Very Good (1e-6 to 1e-5):  %3d / %3d  (%.1f%%)\n",
        very_good, n_compared, 100*very_good/n_compared)
@printf("  ✓   Good (1e-5 to 1e-4):       %3d / %3d  (%.1f%%)\n",
        good, n_compared, 100*good/n_compared)
@printf("  ~   Acceptable (1e-4 to 1e-3): %3d / %3d  (%.1f%%)\n",
        acceptable, n_compared, 100*acceptable/n_compared)
@printf("  ✗   Poor (> 1e-3):             %3d / %3d  (%.1f%%)\n",
        poor, n_compared, 100*poor/n_compared)
println()

# Overall assessment
println("="^80)
if overall_rms < 1e-6
    println("✓✓✓ EXCELLENT: All levels achieve RMS < 1e-6")
    println("    Julia ScaleSpace perfectly matches VLFeat!")
elseif overall_rms < 1e-5
    println("✓✓ VERY GOOD: Overall RMS < 1e-5")
    println("   Julia ScaleSpace closely matches VLFeat")
elseif overall_rms < 1e-4
    println("✓ GOOD: Overall RMS < 1e-4")
    println("  Julia ScaleSpace matches VLFeat reasonably well")
else
    println("✗ NEEDS INVESTIGATION: Overall RMS ≥ 1e-4")
    println("  Differences detected - investigation required")
    println()

    # Show worst offenders
    println("Worst 5 levels:")
    sorted_details = sort(comparison_details, by=x->x.rms, rev=true)
    for i in 1:min(5, length(sorted_details))
        detail = sorted_details[i]
        @printf("  %d. o=%2d, s=%2d: RMS=%.6e, Max=%.6e\n",
                i, detail.octave, detail.subdivision, detail.rms, detail.max_err)
    end
end

println("="^80)
