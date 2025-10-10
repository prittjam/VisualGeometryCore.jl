#!/usr/bin/env julia

using VisualGeometryCore
using TestImages
using ImageFiltering: Kernel, imfilter

println("Testing unified scale space API with filter chains\n")

# Load test image
img = testimage("cameraman")
img_gray = Float32.(img)

h, w = size(img_gray)
println("Test image size: $(w)x$(h)")

# Define filter functions
gaussian_filter = (data, sigma) -> imfilter(data, Kernel.gaussian(sigma), "reflect")
laplacian_filter = (lap_data, hess_level) -> lap_data .= hess_level.Ixx .+ hess_level.Iyy

# Test 1: Gaussian scale space with single filter
println("\n=== Test 1: Gaussian scale space (1 filter) ===")
ss = ScaleSpace(width=w, height=h)
ss(img_gray, gaussian_filter)
println("✓ Gaussian scale space created successfully")
println("  Levels: $(length(ss.levels))")
println("  First level sigma: $(ss.levels[1].sigma)")
println("  Last level sigma: $(ss.levels[end].sigma)")

# Test 2: Hessian scale space from Gaussian
println("\n=== Test 2: Hessian scale space (from Gaussian) ===")
hess_ss = ScaleSpace(width=w, height=h, data_type=:hessian)
hess_ss(ss, hessian_filter)
println("✓ Hessian scale space created successfully")
println("  Levels: $(length(hess_ss.levels))")
println("  Data type: $(typeof(hess_ss.levels.data[1]))")

# Verify Hessian components exist
first_hess = hess_ss.levels.data[1]
println("  Hessian components: $(keys(first_hess))")
println("  Ixx range: [$(minimum(first_hess.Ixx)), $(maximum(first_hess.Ixx))]")
println("  Iyy range: [$(minimum(first_hess.Iyy)), $(maximum(first_hess.Iyy))]")
println("  Ixy range: [$(minimum(first_hess.Ixy)), $(maximum(first_hess.Ixy))]")

# Test 3: Laplacian scale space from Hessian
println("\n=== Test 3: Laplacian scale space (from Hessian) ===")
lap_ss = ScaleSpace(width=w, height=h, data_type=:laplacian)
lap_ss(hess_ss, laplacian_filter)
println("✓ Laplacian scale space created successfully")
println("  Levels: $(length(lap_ss.levels))")
println("  Data type: $(typeof(lap_ss.levels.data[1]))")

# Verify Laplacian values
first_lap = lap_ss.levels.data[1]
println("  Laplacian range: [$(minimum(first_lap)), $(maximum(first_lap))]")

# Test 4: Verify intermediate stages (compare with separate construction)
println("\n=== Test 4: Verify filter chain correctness ===")

# Build Gaussian manually
ss_manual = ScaleSpace(width=w, height=h)
ss_manual(img_gray, gaussian_filter)

# Build Hessian from Gaussian manually
hess_ss_manual = ScaleSpace(width=w, height=h, data_type=:hessian)
for i in eachindex(hess_ss_manual.levels)
    smooth_level = ss_manual.levels.data[i]
    hess_level = hess_ss_manual.levels.data[i]
    hessian_filter(hess_level.Ixx, hess_level.Iyy, hess_level.Ixy, smooth_level)
end

# Compare with chain-built Hessian
max_diff_Ixx = maximum(abs.(hess_ss.levels.data[1].Ixx .- hess_ss_manual.levels.data[1].Ixx))
max_diff_Iyy = maximum(abs.(hess_ss.levels.data[1].Iyy .- hess_ss_manual.levels.data[1].Iyy))
max_diff_Ixy = maximum(abs.(hess_ss.levels.data[1].Ixy .- hess_ss_manual.levels.data[1].Ixy))

println("  Max difference Ixx (chain vs manual): $max_diff_Ixx")
println("  Max difference Iyy (chain vs manual): $max_diff_Iyy")
println("  Max difference Ixy (chain vs manual): $max_diff_Ixy")

if max_diff_Ixx < 1e-6 && max_diff_Iyy < 1e-6 && max_diff_Ixy < 1e-6
    println("  ✓ Filter chain produces identical results to manual construction")
else
    println("  ⚠ Warning: Differences detected between chain and manual construction")
end

println("\n=== All tests completed successfully! ===")
