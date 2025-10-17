"""
Validate Julia ScaleSpace implementation against VLFeat C implementation.

This script:
1. Creates a test image
2. Computes scale space using Julia
3. Saves Julia outputs to TIFF files
4. Runs the C VLFeat comparison program
5. Computes RMS error between Julia and VLFeat outputs
"""

using VisualGeometryCore
using Colors, FixedPointNumbers
using FileIO
using Statistics
using ImageCore: channelview

# Create output directories
mkpath("vlfeat_comparison/julia_gaussian")
mkpath("vlfeat_comparison/vlfeat_gaussian")
mkpath("vlfeat_comparison/vlfeat_hessian_det")

println("=== VLFeat vs Julia Scale Space Validation ===\n")

# Step 1: Create test image (64x64 is reasonable size for validation)
println("Step 1: Creating test image...")
width, height = 64, 64
img = rand(Gray{Float32}, height, width)

# Save input for C program
println("Saving input image...")
save("vlfeat_comparison/input.tif", img)
println("✓ Saved: vlfeat_comparison/input.tif ($(width)x$(height))\n")

# Step 2: Compute Julia scale space
println("Step 2: Computing Julia scale space...")
ss = ScaleSpace(img)

println("Julia ScaleSpace geometry:")
println("  Octaves: $(minimum(ss.levels.octave)) to $(maximum(ss.levels.octave))")
println("  Subdivisions: 0 to $(maximum(ss.levels.subdivision))")
println("  Total levels: $(length(ss.levels))")
println("  Sigma range: $(minimum(ss.levels.sigma)) to $(maximum(ss.levels.sigma))")
println()

# Step 3: Save Julia outputs
println("Step 3: Saving Julia Gaussian levels...")
for level in ss
    filename = "vlfeat_comparison/julia_gaussian/gaussian_o$(level.octave)_s$(level.subdivision).tif"
    save(filename, level.data)
    sz = level_size(level)
    println("  Saved: octave=$(level.octave), subdivision=$(level.subdivision), σ=$(round(level.sigma, digits=3)), size=$(sz.width)x$(sz.height)")
end
println("✓ Saved $(length(ss)) Julia levels\n")

# Step 4: Run C VLFeat program
println("Step 4: Running VLFeat C program...")
println("Note: You must compile and run vlfeat_compare manually:")
println("  gcc -I\$VLFEAT_ROOT -L\$VLFEAT_ROOT/bin/glnxa64 vlfeat_scalespace_compare.c -lvl -ltiff -o vlfeat_compare -lm")
println("  LD_LIBRARY_PATH=\$VLFEAT_ROOT/bin/glnxa64:\$LD_LIBRARY_PATH ./vlfeat_compare")
println()

# Check if VLFeat outputs exist
vlfeat_files = readdir("vlfeat_comparison/vlfeat_gaussian", join=true)
if isempty(vlfeat_files)
    println("⚠ VLFeat outputs not found. Run the C program first!")
    println("Stopping validation. Once C program runs, re-run this script for comparison.")
    exit(0)
end

# Step 5: Compare outputs
println("Step 5: Comparing Julia vs VLFeat outputs...")
println()

rms_errors = Float64[]
max_errors = Float64[]

for level in ss
    o, s = level.octave, level.subdivision
    julia_file = "vlfeat_comparison/julia_gaussian/gaussian_o$(o)_s$(s).tif"
    vlfeat_file = "vlfeat_comparison/vlfeat_gaussian/gaussian_o$(o)_s$(s).tif"

    if !isfile(vlfeat_file)
        println("  ⚠ Missing VLFeat file: $vlfeat_file")
        continue
    end

    # Load both images
    julia_img = load(julia_file)
    vlfeat_img = load(vlfeat_file)

    # Convert to Float32 arrays
    julia_data = Float32.(channelview(julia_img))
    vlfeat_data = Float32.(channelview(vlfeat_img))

    # Check dimensions match
    if size(julia_data) != size(vlfeat_data)
        println("  ✗ Size mismatch at o=$o, s=$s: Julia=$(size(julia_data)), VLFeat=$(size(vlfeat_data))")
        continue
    end

    # Compute errors
    diff = julia_data .- vlfeat_data
    rms = sqrt(mean(diff.^2))
    max_err = maximum(abs.(diff))

    push!(rms_errors, rms)
    push!(max_errors, max_err)

    # Show per-level results
    status = rms < 1e-6 ? "✓" : (rms < 1e-5 ? "⚠" : "✗")
    println("  $status o=$o, s=$s: RMS=$(rms), Max=$(max_err)")
end

println()
println("="^60)
println("Validation Summary:")
println("="^60)
println("Compared: $(length(rms_errors)) levels")

if !isempty(rms_errors)
    overall_rms = sqrt(mean(rms_errors.^2))
    mean_rms = mean(rms_errors)
    max_rms = maximum(rms_errors)
    overall_max = maximum(max_errors)

    println("\nRMS Error Statistics:")
    println("  Overall RMS: $(overall_rms)")
    println("  Mean RMS: $(mean_rms)")
    println("  Max RMS: $(max_rms)")
    println("\nMax Absolute Error Statistics:")
    println("  Overall Max: $(overall_max)")

    println()
    if overall_rms < 1e-6
        println("✓✓✓ EXCELLENT: RMS error < 1e-6 (target achieved!)")
    elseif overall_rms < 1e-5
        println("✓ GOOD: RMS error < 1e-5 (close to target)")
    elseif overall_rms < 1e-4
        println("⚠ ACCEPTABLE: RMS error < 1e-4 (may need tuning)")
    else
        println("✗ NEEDS WORK: RMS error ≥ 1e-4 (investigate differences)")
    end
else
    println("\n⚠ No comparisons performed - check that VLFeat outputs exist")
end
