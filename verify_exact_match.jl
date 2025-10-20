#!/usr/bin/env julia
"""
Verify exact match between Julia and VLFeat features.
Shows detailed comparison for all features.
"""

using VisualGeometryCore
using FileIO
using JSON3
using Printf

println("="^80)
println("DETAILED FEATURE COMPARISON: Julia vs VLFeat")
println("="^80)
println()

# Load VLFeat results
vlfeat_json = JSON3.read(read("vlfeat_detections.json", String))
vlfeat_features = vlfeat_json["features"]

# Run Julia detection
img = load("vlfeat_comparison/input.tif")
ss = ScaleSpace(img; first_octave=-1, octave_resolution=3, first_subdivision=-1, last_subdivision=3)

hessian_resp = ScaleSpaceResponse(ss, DERIVATIVE_KERNELS.xx)
for level in ss
    step = 2.0^level.octave
    det_data = vlfeat_hessian_det(level.data, level.sigma, step)
    resp_level = hessian_resp[level.octave, level.subdivision]
    resp_level.data .= Gray{Float32}.(det_data)
end

extrema = detect_extrema(hessian_resp;
    peak_threshold=0.003,
    edge_threshold=10.0,
    base_scale=2.015874,
    octave_resolution=3)

# Filter to VLFeat's octave range
extrema_in_range = filter(e -> e.octave >= -1 && e.octave <= 2, extrema)

@printf("VLFeat features: %d\n", length(vlfeat_features))
@printf("Julia features:  %d\n", length(extrema_in_range))
println()

if length(vlfeat_features) != length(extrema_in_range)
    println("❌ COUNT MISMATCH!")
    exit(1)
end

# Sort both by position for easier comparison
vf_sorted = sort(collect(vlfeat_features), by=f -> (f["x"], f["y"], f["sigma"]))

julia_converted = [(
    x = (je.x - 1) * je.step,
    y = (je.y - 1) * je.step,
    sigma = je.sigma,
    peak = je.peakScore,
    edge = je.edgeScore,
    octave = je.octave
) for je in extrema_in_range]

jl_sorted = sort(julia_converted, by=f -> (f.x, f.y, f.sigma))

# Compare each pair
println("Detailed comparison (first 10 features):")
println("-"^80)

for i in 1:min(10, length(vf_sorted))
    vf = vf_sorted[i]
    jl = jl_sorted[i]

    pos_err = sqrt((vf["x"] - jl.x)^2 + (vf["y"] - jl.y)^2)
    sigma_err = abs(vf["sigma"] - jl.sigma)
    peak_err = abs(vf["peakScore"] - jl.peak)
    edge_err = abs(vf["edgeScore"] - jl.edge)

    @printf("#%2d:\n", i)
    @printf("  Position: VF=(%.3f, %.3f) JL=(%.3f, %.3f) err=%.6f\n",
            vf["x"], vf["y"], jl.x, jl.y, pos_err)
    @printf("  Sigma:    VF=%.3f JL=%.3f err=%.6f\n",
            vf["sigma"], jl.sigma, sigma_err)
    @printf("  Peak:     VF=%.6e JL=%.6e err=%.6e\n",
            vf["peakScore"], jl.peak, peak_err)
    @printf("  Edge:     VF=%.3f JL=%.3f err=%.3f\n",
            vf["edgeScore"], jl.edge, edge_err)
    println()
end

# Compute errors for all features
all_pos_errs = [sqrt((vf["x"] - jl.x)^2 + (vf["y"] - jl.y)^2)
                for (vf, jl) in zip(vf_sorted, jl_sorted)]
all_sigma_errs = [abs(vf["sigma"] - jl.sigma)
                  for (vf, jl) in zip(vf_sorted, jl_sorted)]
all_peak_errs = [abs(vf["peakScore"] - jl.peak)
                 for (vf, jl) in zip(vf_sorted, jl_sorted)]
all_edge_errs = [abs(vf["edgeScore"] - jl.edge)
                 for (vf, jl) in zip(vf_sorted, jl_sorted)]

println("="^80)
println("ERROR STATISTICS (all 40 features)")
println("="^80)
println()

println("Position errors:")
@printf("  Max:    %.6e\n", maximum(all_pos_errs))
@printf("  Mean:   %.6e\n", sum(all_pos_errs) / length(all_pos_errs))
@printf("  Median: %.6e\n", sort(all_pos_errs)[div(length(all_pos_errs), 2)])
println()

println("Sigma errors:")
@printf("  Max:    %.6e\n", maximum(all_sigma_errs))
@printf("  Mean:   %.6e\n", sum(all_sigma_errs) / length(all_sigma_errs))
@printf("  Median: %.6e\n", sort(all_sigma_errs)[div(length(all_sigma_errs), 2)])
println()

println("Peak score errors:")
@printf("  Max:    %.6e\n", maximum(all_peak_errs))
@printf("  Mean:   %.6e\n", sum(all_peak_errs) / length(all_peak_errs))
@printf("  Median: %.6e\n", sort(all_peak_errs)[div(length(all_peak_errs), 2)])
println()

println("Edge score errors:")
@printf("  Max:    %.6e\n", maximum(all_edge_errs))
@printf("  Mean:   %.6e\n", sum(all_edge_errs) / length(all_edge_errs))
@printf("  Median: %.6e\n", sort(all_edge_errs)[div(length(all_edge_errs), 2)])
println()

println("="^80)
println("VERDICT")
println("="^80)
println()

if maximum(all_pos_errs) < 1e-3 && maximum(all_sigma_errs) < 1e-3
    println("✓✓✓ PERFECT MATCH!")
    println("    All features match VLFeat within sub-pixel accuracy")
    @printf("    Max position error: %.6e\n", maximum(all_pos_errs))
    @printf("    Max sigma error: %.6e\n", maximum(all_sigma_errs))
else
    println("⚠ Good match but not perfect")
    @printf("  Max position error: %.6e\n", maximum(all_pos_errs))
    @printf("  Max sigma error: %.6e\n", maximum(all_sigma_errs))
end

println()
println("="^80)
