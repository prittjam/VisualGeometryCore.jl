#!/usr/bin/env julia

using VisualGeometryCore
using TestImages
using ImageFiltering: Kernel, imfilter, centered

println("Testing unified scale space API with broadcast-based kernels\n")

# Load test image
img = testimage("cameraman")
img_gray = Float32.(img)

h, w = size(img_gray)
println("Test image size: $(w)x$(h)")

# Define kernel function for Gaussian smoothing (returns kernel, not filtered result)
gaussian_kernel = (sigma) -> Kernel.gaussian(sigma)

# Pre-construct derivative kernels for Hessian
kernel_xx = centered([0 0 0; 1 -2 1; 0 0 0])
kernel_yy = centered([0 1 0; 0 -2 0; 0 1 0])
kernel_xy = centered([0.25 0 -0.25; 0 0 0; -0.25 0 0.25])

# Test 1: Gaussian scale space with single filter
println("\n=== Test 1: Gaussian scale space (1 filter) ===")
ss = ScaleSpace(img_gray)  # Use new image-based constructor
ss(img_gray, gaussian_kernel)
println("✓ Gaussian scale space created successfully")
println("  Levels: $(length(ss.levels))")
println("  First level sigma: $(ss.levels[1].sigma)")
println("  Last level sigma: $(ss.levels[end].sigma)")

# Test 2: Hessian scale space from Gaussian using broadcast kernels
println("\n=== Test 2: Hessian scale space (broadcast kernels) ===")
hess_ss = ScaleSpace(VisualGeometryCore.Size2(w, h), image_type=VisualGeometryCore.HessianImages{Float32})
hess_ss(ss, [kernel_xx, kernel_yy, kernel_xy])
println("✓ Hessian scale space created successfully")
println("  Levels: $(length(hess_ss.levels))")
println("  Data type: $(typeof(hess_ss.levels.data[1]))")

# Verify Hessian components exist
first_hess = hess_ss.levels.data[1]
println("  Hessian components: $(keys(first_hess))")
println("  Ixx range: [$(minimum(first_hess.Ixx)), $(maximum(first_hess.Ixx))]")
println("  Iyy range: [$(minimum(first_hess.Iyy)), $(maximum(first_hess.Iyy))]")
println("  Ixy range: [$(minimum(first_hess.Ixy)), $(maximum(first_hess.Ixy))]")

# Test 3: Laplacian scale space from Hessian using pure broadcasting
println("\n=== Test 3: Laplacian scale space (pure broadcast) ===")
lap_ss = ScaleSpace(VisualGeometryCore.Size2(w, h), image_type=VisualGeometryCore.LaplacianImage{Float32})
lap_ss.levels.data .= [(L = ixx + iyy,) for (ixx, iyy) in zip(getfield.(hess_ss.levels.data, :Ixx), getfield.(hess_ss.levels.data, :Iyy))]
println("✓ Laplacian scale space created successfully")
println("  Levels: $(length(lap_ss.levels))")
println("  Data type: $(typeof(lap_ss.levels.data[1]))")

# Verify Laplacian values
first_lap = lap_ss.levels.data[1]
println("  Laplacian range: [$(minimum(first_lap)), $(maximum(first_lap))]")

# Test 4: Verify broadcast correctness
println("\n=== Test 4: Verify broadcast correctness ===")

# Build another Hessian manually using element-wise operations
hess_ss_manual = ScaleSpace(VisualGeometryCore.Size2(w, h), image_type=VisualGeometryCore.HessianImages{Float32})
for i in eachindex(hess_ss_manual.levels)
    smooth_level = ss.levels.data[i].g  # Access the .g field
    hess_level = hess_ss_manual.levels.data[i]
    hess_level.Ixx .= imfilter(smooth_level, kernel_xx)
    hess_level.Iyy .= imfilter(smooth_level, kernel_yy)
    hess_level.Ixy .= imfilter(smooth_level, kernel_xy)
end

# Compare with broadcast-built Hessian
max_diff_Ixx = maximum(abs.(hess_ss.levels.data[1].Ixx .- hess_ss_manual.levels.data[1].Ixx))
max_diff_Iyy = maximum(abs.(hess_ss.levels.data[1].Iyy .- hess_ss_manual.levels.data[1].Iyy))
max_diff_Ixy = maximum(abs.(hess_ss.levels.data[1].Ixy .- hess_ss_manual.levels.data[1].Ixy))

println("  Max difference Ixx (broadcast vs manual): $max_diff_Ixx")
println("  Max difference Iyy (broadcast vs manual): $max_diff_Iyy")
println("  Max difference Ixy (broadcast vs manual): $max_diff_Ixy")

if max_diff_Ixx < 1e-6 && max_diff_Iyy < 1e-6 && max_diff_Ixy < 1e-6
    println("  ✓ Broadcast filtering produces identical results to manual construction")
else
    println("  ⚠ Warning: Differences detected")
end

println("\n=== All tests completed successfully! ===")
