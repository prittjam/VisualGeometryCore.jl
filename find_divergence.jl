#!/usr/bin/env julia
"""
Find exactly where VLFeat and Julia diverge.

1. Responses match ✓ (already verified)
2. Check: Do discrete extrema match?
3. Check: Does refinement match?
4. Check: Does filtering match?
"""

using VisualGeometryCore
using FileIO
using JSON3
using Printf
using NearestNeighbors

println("="^80)
println("FINDING DIVERGENCE POINT")
println("="^80)
println()

# Load VLFeat results
vlfeat_json = JSON3.read(read("vlfeat_detections.json", String))
vlfeat_features = vlfeat_json["features"]
vlfeat_params = vlfeat_json["parameters"]

println("VLFeat detected: $(length(vlfeat_features)) features")
println()

# Run Julia with same parameters
img = load("vlfeat_comparison/input.tif")
ss = ScaleSpace(img;
    first_octave=Int(vlfeat_params["first_octave"]),
    octave_resolution=Int(vlfeat_params["octave_resolution"]),
    first_subdivision=-1,
    last_subdivision=3)

hessian_resp = ScaleSpaceResponse(ss, DERIVATIVE_KERNELS.xx)
for level in ss
    step = 2.0^level.octave
    det_data = vlfeat_hessian_det(level.data, level.sigma, step)
    resp_level = hessian_resp[level.octave, level.subdivision]
    resp_level.data .= Gray{Float32}.(det_data)
end

extrema, discrete = detect_extrema(hessian_resp;
    peak_threshold=Float64(vlfeat_params["peak_threshold"]),
    edge_threshold=Float64(vlfeat_params["edge_threshold"]),
    base_scale=Float64(vlfeat_params["base_scale"]),
    octave_resolution=Int(vlfeat_params["octave_resolution"]),
    return_discrete=true)

println("Julia detected:")
println("  Discrete extrema: $(length(discrete))")
println("  Refined extrema:  $(length(extrema))")
println()

# =============================================================================
# Use NearestNeighbors to match features
# =============================================================================

println("="^80)
println("MATCHING WITH NEAREST NEIGHBORS")
println("="^80)
println()

# Build VLFeat feature matrix: [x, y, sigma]
vlfeat_matrix = zeros(3, length(vlfeat_features))
for (i, vf) in enumerate(vlfeat_features)
    vlfeat_matrix[1, i] = vf["x"]
    vlfeat_matrix[2, i] = vf["y"]
    vlfeat_matrix[3, i] = vf["sigma"]
end

# Build Julia feature matrix: [x, y, sigma] with coordinate swap
julia_matrix = zeros(3, length(extrema))
for (i, je) in enumerate(extrema)
    # Swap coordinates: Julia (col-major) to VLFeat (row-major)
    julia_matrix[1, i] = je.y / je.step  # Julia y → VLFeat x
    julia_matrix[2, i] = je.x / je.step  # Julia x → VLFeat y
    julia_matrix[3, i] = je.sigma
end

# Build KD-tree for VLFeat features
kdtree = KDTree(vlfeat_matrix)

# Find 1-nearest neighbor for each Julia feature
idxs, dists = knn(kdtree, julia_matrix, 1)

# Count good matches (distance < 0.5 pixels in position + sigma)
good_matches = 0
position_dists = Float64[]
sigma_diffs = Float64[]

for i in 1:length(extrema)
    vf_idx = idxs[i][1]
    dist = dists[i][1]

    # Compute position-only distance
    pos_dist = sqrt((julia_matrix[1,i] - vlfeat_matrix[1,vf_idx])^2 +
                    (julia_matrix[2,i] - vlfeat_matrix[2,vf_idx])^2)
    sigma_diff = abs(julia_matrix[3,i] - vlfeat_matrix[3,vf_idx])

    push!(position_dists, pos_dist)
    push!(sigma_diffs, sigma_diff)

    if dist < 0.5  # Good match
        good_matches += 1
    end
end

@printf("Good matches (dist < 0.5): %d / %d (%.1f%%)\n",
        good_matches, length(extrema), 100*good_matches/length(extrema))
println()

@printf("Position distance statistics:\n")
@printf("  Min:    %.6f\n", minimum(position_dists))
@printf("  Max:    %.6f\n", maximum(position_dists))
@printf("  Mean:   %.6f\n", sum(position_dists)/length(position_dists))
@printf("  Median: %.6f\n", median(position_dists))
println()

@printf("Sigma difference statistics:\n")
@printf("  Min:    %.6f\n", minimum(sigma_diffs))
@printf("  Max:    %.6f\n", maximum(sigma_diffs))
@printf("  Mean:   %.6f\n", sum(sigma_diffs)/length(sigma_diffs))
@printf("  Median: %.6f\n", median(sigma_diffs))
println()

# =============================================================================
# Show detailed comparison for first 10
# =============================================================================

println("="^80)
println("FIRST 10 FEATURE COMPARISONS")
println("="^80)
println()

for i in 1:min(10, length(extrema))
    vf_idx = idxs[i][1]
    vf = vlfeat_features[vf_idx]
    je = extrema[i]

    je_x = je.y / je.step
    je_y = je.x / je.step

    println("Feature $i:")
    @printf("  VLFeat[%2d]: (%.4f, %.4f) σ=%.4f peak=%.6e edge=%.4f\n",
            vf_idx, vf["x"], vf["y"], vf["sigma"], vf["peakScore"], vf["edgeScore"])
    @printf("  Julia [%2d]: (%.4f, %.4f) σ=%.4f peak=%.6e edge=%.4f\n",
            i, je_x, je_y, je.sigma, je.peakScore, je.edgeScore)
    @printf("  Distance: %.6f, Δσ: %.6f\n", position_dists[i], sigma_diffs[i])
    println()
end

# =============================================================================
# Check: Are there VLFeat features with no close Julia match?
# =============================================================================

println("="^80)
println("CHECKING FOR MISSING JULIA FEATURES")
println("="^80)
println()

# Build KD-tree for Julia features
julia_kdtree = KDTree(julia_matrix)

# Find nearest Julia feature for each VLFeat feature
vf_idxs, vf_dists = knn(julia_kdtree, vlfeat_matrix, 1)

missing_count = 0
for i in 1:length(vlfeat_features)
    if vf_dists[i][1] > 0.5
        missing_count += 1
        if missing_count <= 5
            vf = vlfeat_features[i]
            println("VLFeat feature $i has no close Julia match:")
            @printf("  (%.4f, %.4f) σ=%.4f peak=%.6e edge=%.4f\n",
                    vf["x"], vf["y"], vf["sigma"], vf["peakScore"], vf["edgeScore"])
        end
    end
end

@printf("\nVLFeat features without close Julia match: %d / %d\n", missing_count, length(vlfeat_features))
println()

println("="^80)
println("CONCLUSION")
println("="^80)
println()

if good_matches == length(extrema) && missing_count == 0
    println("✓✓✓ PERFECT MATCH!")
    println("    Julia implementation matches VLFeat exactly")
elseif good_matches >= 0.95 * length(extrema)
    println("✓ EXCELLENT MATCH (>95%)")
    println("  Minor differences likely due to:")
    println("  - Floating point precision")
    println("  - Threshold boundary cases")
else
    println("⚠ DIVERGENCE DETECTED")
    println()
    println("Analysis:")
    @printf("  - Julia found: %d discrete extrema\n", length(discrete))
    @printf("  - Julia refined: %d extrema\n", length(extrema))
    @printf("  - VLFeat refined: %d features\n", length(vlfeat_features))
    @printf("  - Matched: %d (%.1f%%)\n", good_matches, 100*good_matches/length(extrema))
    println()
    println("The divergence is in EXTREMA DETECTION or REFINEMENT")
    println("Responses match perfectly, so the issue is algorithmic.")
end

println()
println("="^80)
