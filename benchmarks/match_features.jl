#!/usr/bin/env julia
"""
Match Julia features with VLFeat features using NearestNeighbors.jl
"""

using VisualGeometryCore
using FileIO
using JSON3
using Printf
using NearestNeighbors
using Statistics

println("="^80)
println("MATCHING JULIA vs VLFEAT FEATURES")
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

discrete = find_extrema_3d(hessian_resp, 0.8 * 0.003)
extrema = refine_extrema(hessian_resp, discrete;
    peak_threshold=0.003,
    edge_threshold=10.0,
    base_scale=2.015874,
    octave_resolution=3)

# Filter to VLFeat's octave range
extrema_in_range = filter(e -> e.octave >= -1 && e.octave <= 2, extrema)

@printf("VLFeat features: %d\n", length(vlfeat_features))
@printf("Julia extrema:   %d\n", length(extrema_in_range))
println()

# Build feature matrices: [x, y, sigma]
# VLFeat coordinates
vlfeat_matrix = zeros(3, length(vlfeat_features))
for (i, vf) in enumerate(vlfeat_features)
    vlfeat_matrix[1, i] = vf["x"]
    vlfeat_matrix[2, i] = vf["y"]
    vlfeat_matrix[3, i] = vf["sigma"]
end

# Julia coordinates with CORRECT conversion:
# vlfeat_coord = (julia_coord - 1) * step
julia_matrix = zeros(3, length(extrema_in_range))
for (i, je) in enumerate(extrema_in_range)
    julia_matrix[1, i] = (je.x - 1) * je.step  # 1-based → 0-based, then scale
    julia_matrix[2, i] = (je.y - 1) * je.step
    julia_matrix[3, i] = je.sigma
end

# Build KD-tree for VLFeat features
kdtree = KDTree(vlfeat_matrix)

# Find 1-nearest neighbor for each Julia feature
idxs, dists = knn(kdtree, julia_matrix, 1)

# Flatten results
julia_to_vlfeat = [idxs[i][1] for i in 1:length(extrema_in_range)]
match_distances = [dists[i][1] for i in 1:length(extrema_in_range)]

println("="^80)
println("MATCH STATISTICS")
println("="^80)
println()

perfect_matches = sum(match_distances .< 1e-6)
good_matches = sum(match_distances .< 0.01)
@printf("Perfect matches (dist < 1e-6): %d / %d (%.1f%%)\n",
        perfect_matches, length(extrema_in_range), 100*perfect_matches/length(extrema_in_range))
@printf("Good matches (dist < 0.01):    %d / %d (%.1f%%)\n",
        good_matches, length(extrema_in_range), 100*good_matches/length(extrema_in_range))
println()

@printf("Distance statistics:\n")
@printf("  Min:    %.6e\n", minimum(match_distances))
@printf("  Max:    %.6e\n", maximum(match_distances))
@printf("  Mean:   %.6e\n", mean(match_distances))
@printf("  Median: %.6e\n", median(match_distances))
println()

# Check for unmatched VLFeat features (reverse matching)
julia_kdtree = KDTree(julia_matrix)
vf_to_julia_idxs, vf_to_julia_dists = knn(julia_kdtree, vlfeat_matrix, 1)
vf_match_dists = [vf_to_julia_dists[i][1] for i in 1:length(vlfeat_features)]

unmatched_vf = findall(vf_match_dists .> 0.01)
@printf("Unmatched VLFeat features: %d / %d\n", length(unmatched_vf), length(vlfeat_features))

if !isempty(unmatched_vf)
    println()
    println("Unmatched VLFeat features:")
    for i in unmatched_vf[1:min(5, length(unmatched_vf))]
        vf = vlfeat_features[i]
        @printf("  #%2d: (%.3f, %.3f) σ=%.3f peak=%.6e edge=%.3f\n",
                i, vf["x"], vf["y"], vf["sigma"], vf["peakScore"], vf["edgeScore"])
    end
end

println()
println("="^80)
println("SUMMARY")
println("="^80)
println()

if perfect_matches == length(extrema_in_range) && length(unmatched_vf) == 0
    println("✓✓✓ PERFECT MATCH!")
    println("    Julia implementation matches VLFeat exactly!")
elseif good_matches == length(extrema_in_range) && length(unmatched_vf) <= 2
    println("✓✓ EXCELLENT MATCH!")
    println("    Julia finds $(length(extrema_in_range)) features, VLFeat finds $(length(vlfeat_features))")
    println("    Difference: $(abs(length(extrema_in_range) - length(vlfeat_features))) features")
    println("    All matched features have dist < 0.01")
else
    println("⚠ Some differences detected")
end

println()
println("Coordinate conversion formula:")
println("  vlfeat_x = (julia_x - 1) * step")
println("  vlfeat_y = (julia_y - 1) * step")
println("  (Julia is 1-indexed, VLFeat is 0-indexed)")
println()
println("="^80)
