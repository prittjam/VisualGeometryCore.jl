"""
Debug Hessian determinant by comparing pixel values directly.
"""

using VisualGeometryCore
using FileIO
using ImageCore: channelview
using Colors
using Statistics
using Printf

# Load input and create scale space
img = load("vlfeat_comparison/input.tif")
ss = ScaleSpace(img; first_octave=-1, octave_resolution=3,
               first_subdivision=-1, last_subdivision=3)

# Test octave -1, subdivision -1 (where we have largest error)
level = ss[-1, -1]
step = 2.0^(-1)  # 0.5

# Compute Julia Hessian determinant
julia_det = vlfeat_hessian_det(level.data, level.sigma, step)

# Load VLFeat reference
vlfeat_img = load("vlfeat_comparison/vlfeat_hessian_det/hessian_det_o-1_s-1.tif")
vlfeat_det = Float32.(channelview(vlfeat_img))

println("="^70)
println("Comparing octave -1, subdivision -1 pixel values")
println("="^70)
println("Image size: $(size(julia_det))")
println("Sigma: $(level.sigma), Step: $(step)")
println()

# Compare several pixels
test_pixels = [(10, 10), (50, 50), (100, 100), (5, 5), (150, 150)]

for (y, x) in test_pixels
    if y <= size(julia_det, 1) && x <= size(julia_det, 2)
        j_val = julia_det[y, x]
        v_val = vlfeat_det[y, x]
        diff = j_val - v_val
        rel_err = abs(diff) / max(abs(v_val), 1e-10)
        @printf("Pixel (%3d,%3d): Julia=%.6e, VLFeat=%.6e, Diff=%.6e, RelErr=%.2e\n",
                y, x, j_val, v_val, diff, rel_err)
    end
end

println()
println("="^70)
println("Statistics")
println("="^70)
diff = julia_det .- vlfeat_det
println("Mean difference: $(mean(diff))")
println("RMS error: $(sqrt(mean(diff.^2)))")
println("Max abs error: $(maximum(abs.(diff)))")
println("Mean Julia: $(mean(julia_det))")
println("Mean VLFeat: $(mean(vlfeat_det))")
println("Std Julia: $(std(julia_det))")
println("Std VLFeat: $(std(vlfeat_det))")

# Check for systematic bias
println()
println("Correlation: $(cor(vec(julia_det), vec(vlfeat_det)))")
