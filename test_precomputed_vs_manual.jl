#!/usr/bin/env julia
"""
Test that pre-computed derivative refinement gives identical results to manual computation.
"""

using VisualGeometryCore
using FileIO
using Printf

println("="^80)
println("TESTING: Pre-computed vs Manual Derivative Refinement")
println("="^80)
println()

# Load test image
img = load("vlfeat_comparison/input.tif")

# Build scale space
ss = ScaleSpace(img; first_octave=-1, octave_resolution=3,
               first_subdivision=-1, last_subdivision=3)

# Compute Hessian determinant response using VLFeat
hessian_resp = ScaleSpaceResponse(ss, DERIVATIVE_KERNELS.xx)
for level in ss
    step = 2.0^level.octave
    det_data = vlfeat_hessian_det(level.data, level.sigma, step)
    resp_level = hessian_resp[level.octave, level.subdivision]
    resp_level.data .= Gray{Float32}.(det_data)
end

println("✓ Computed Hessian determinant response")
println()

# Find discrete extrema for one octave
octave_num = -1
octave_cube = hessian_resp[octave_num].G
discrete_extrema = find_extrema_3d(octave_cube, 0.003)

println("Found $(length(discrete_extrema)) discrete extrema in octave $octave_num")
println()

# Pre-compute ALL derivatives for this octave
println("Pre-computing 3D derivatives...")
derivatives_resp = (
    ∇x = ScaleSpaceResponse(hessian_resp, DERIVATIVE_KERNELS_3D.dx)(hessian_resp),
    ∇y = ScaleSpaceResponse(hessian_resp, DERIVATIVE_KERNELS_3D.dy)(hessian_resp),
    ∇z = ScaleSpaceResponse(hessian_resp, DERIVATIVE_KERNELS_3D.dz)(hessian_resp),
    ∇²xx = ScaleSpaceResponse(hessian_resp, DERIVATIVE_KERNELS_3D.dxx)(hessian_resp),
    ∇²yy = ScaleSpaceResponse(hessian_resp, DERIVATIVE_KERNELS_3D.dyy)(hessian_resp),
    ∇²zz = ScaleSpaceResponse(hessian_resp, DERIVATIVE_KERNELS_3D.dzz)(hessian_resp),
    ∇²xy = ScaleSpaceResponse(hessian_resp, DERIVATIVE_KERNELS_3D.dxy)(hessian_resp),
    ∇²xz = ScaleSpaceResponse(hessian_resp, DERIVATIVE_KERNELS_3D.dxz)(hessian_resp),
    ∇²yz = ScaleSpaceResponse(hessian_resp, DERIVATIVE_KERNELS_3D.dyz)(hessian_resp)
)

# Extract octave cubes for this octave
derivatives_cubes = (
    response = hessian_resp[octave_num].G,
    ∇x = derivatives_resp.∇x[octave_num].G,
    ∇y = derivatives_resp.∇y[octave_num].G,
    ∇z = derivatives_resp.∇z[octave_num].G,
    ∇²xx = derivatives_resp.∇²xx[octave_num].G,
    ∇²yy = derivatives_resp.∇²yy[octave_num].G,
    ∇²zz = derivatives_resp.∇²zz[octave_num].G,
    ∇²xy = derivatives_resp.∇²xy[octave_num].G,
    ∇²xz = derivatives_resp.∇²xz[octave_num].G,
    ∇²yz = derivatives_resp.∇²yz[octave_num].G
)

println("✓ Pre-computed derivatives")
println()

# Refine using both methods and compare
println("Refining extrema using both methods...")
println()

first_subdivision = -1
octave_resolution = 3
base_scale = 2.015874
step = 2.0^octave_num

results = []

for (x, y, z) in discrete_extrema
    # Refine using pre-computed derivatives
    result_precomp, success_precomp = refine_extremum_3d(
        derivatives_cubes, x, y, z,
        octave_num, first_subdivision, octave_resolution,
        base_scale, step
    )

    # Refine using manual computation
    result_manual, success_manual = refine_extremum_3d_manual(
        octave_cube, x, y, z,
        octave_num, first_subdivision, octave_resolution,
        base_scale, step
    )

    # Compare results
    if success_precomp != success_manual
        println("⚠ SUCCESS MISMATCH at ($x, $y, $z): precomp=$success_precomp, manual=$success_manual")
        continue
    end

    if !success_precomp
        # Both failed, that's fine
        continue
    end

    # Compare refined positions
    pos_error = sqrt((result_precomp.x - result_manual.x)^2 +
                    (result_precomp.y - result_manual.y)^2 +
                    (result_precomp.z - result_manual.z)^2)

    sigma_error = abs(result_precomp.sigma - result_manual.sigma)
    peak_error = abs(result_precomp.peakScore - result_manual.peakScore)
    edge_error = abs(result_precomp.edgeScore - result_manual.edgeScore)

    push!(results, (pos=pos_error, sigma=sigma_error, peak=peak_error, edge=edge_error))

    # Check for exact match (within floating point precision)
    if !(pos_error < 1e-10 && sigma_error < 1e-10 && peak_error < 1e-10 && edge_error < 1e-10)
        println("Extremum at ($x, $y, $z):")
        @printf("  Position error: %.6e\n", pos_error)
        @printf("  Sigma error:    %.6e\n", sigma_error)
        @printf("  Peak error:     %.6e\n", peak_error)
        @printf("  Edge error:     %.6e\n", edge_error)
    end
end

println()
println("="^80)
println("RESULTS")
println("="^80)
println()

num_compared = length(results)
num_matched = count(r -> r.pos < 1e-10 && r.sigma < 1e-10 && r.peak < 1e-10 && r.edge < 1e-10, results)

max_pos_error = isempty(results) ? 0.0 : maximum(r -> r.pos, results)
max_sigma_error = isempty(results) ? 0.0 : maximum(r -> r.sigma, results)
max_peak_error = isempty(results) ? 0.0 : maximum(r -> r.peak, results)
max_edge_error = isempty(results) ? 0.0 : maximum(r -> r.edge, results)

@printf("Extrema compared: %d\n", num_compared)
@printf("Exact matches:    %d (%.1f%%)\n", num_matched, 100.0 * num_matched / max(num_compared, 1))
println()

@printf("Maximum errors:\n")
@printf("  Position: %.6e\n", max_pos_error)
@printf("  Sigma:    %.6e\n", max_sigma_error)
@printf("  Peak:     %.6e\n", max_peak_error)
@printf("  Edge:     %.6e\n", max_edge_error)
println()

if num_matched == num_compared && max_pos_error < 1e-10
    println("✓✓✓ PERFECT MATCH!")
    println("Pre-computed and manual refinement give identical results.")
else
    println("⚠ Some discrepancies detected")
end

println()
println("="^80)
