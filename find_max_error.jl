"""
Find the location of maximum Hessian determinant error.
"""

using VisualGeometryCore
using FileIO
using ImageCore: channelview
using Colors
using Printf

# Load input and create scale space
img = load("vlfeat_comparison/input.tif")
ss = ScaleSpace(img; first_octave=-1, octave_resolution=3,
               first_subdivision=-1, last_subdivision=3)

# Test octave -1, subdivision -1
level = ss[-1, -1]
step = 2.0^(-1)

# Compute Julia Hessian determinant
julia_det = vlfeat_hessian_det(level.data, level.sigma, step)

# Load VLFeat reference
vlfeat_img = load("vlfeat_comparison/vlfeat_hessian_det/hessian_det_o-1_s-1.tif")
vlfeat_det = Float32.(channelview(vlfeat_img))

# Find maximum error location
diff = abs.(julia_det .- vlfeat_det)
max_idx = argmax(diff)
max_error = diff[max_idx]

y, x = Tuple(max_idx)

println("="^70)
println("Maximum error location")
println("="^70)
@printf("Location: (%d, %d)\n", y, x)
@printf("Julia value: %.6e\n", julia_det[y, x])
@printf("VLFeat value: %.6e\n", vlfeat_det[y, x])
@printf("Absolute error: %.6e\n", max_error)
println()

# Check surrounding region (5x5 window around max error)
println("Surrounding 5x5 region:")
println()
println("Julia values:")
y_range = max(1, y-2):min(size(julia_det, 1), y+2)
x_range = max(1, x-2):min(size(julia_det, 2), x+2)
for row in y_range
    for col in x_range
        @printf("%9.4f ", julia_det[row, col])
    end
    println()
end

println()
println("VLFeat values:")
for row in y_range
    for col in x_range
        @printf("%9.4f ", vlfeat_det[row, col])
    end
    println()
end

println()
println("Differences:")
for row in y_range
    for col in x_range
        @printf("%9.6f ", julia_det[row, col] - vlfeat_det[row, col])
    end
    println()
end

# Check if max error is at boundary
h, w = size(julia_det)
if y <= 2 || y >= h-1 || x <= 2 || x >= w-1
    println()
    println("⚠ Maximum error is near image boundary! (within 2 pixels)")
    println("  This suggests a border handling difference.")
end

# Also check the underlying Gaussian image at this location
println()
println("="^70)
println("Underlying Gaussian level at max error location")
println("="^70)
gaussian_data = Float32.(channelview(level.data))
vlfeat_gaussian = Float32.(channelview(load("vlfeat_comparison/vlfeat_gaussian/gaussian_o-1_s-1.tif")))

@printf("Julia Gaussian: %.6e\n", gaussian_data[y, x])
@printf("VLFeat Gaussian: %.6e\n", vlfeat_gaussian[y, x])
@printf("Gaussian diff: %.6e\n", abs(gaussian_data[y, x] - vlfeat_gaussian[y, x]))
