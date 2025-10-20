#!/usr/bin/env julia
"""
Systematic comparison of Julia implementation with VLFeat output.

This test script compares at each stage:
1. Discrete extrema (before refinement)
2. Refined positions
3. Peak scores
4. Edge scores
5. Final filtered features

The goal is to match VLFeat's output exactly.
"""

using VisualGeometryCore
using FileIO
using JSON3
using Printf
using Statistics

println("="^80)
println("SYSTEMATIC VLFEAT COMPARISON")
println("="^80)
println()

# =============================================================================
# Step 1: Run VLFeat C program
# =============================================================================

println("Step 1: Running VLFeat C detector")
println("-"^80)

# Run VLFeat
run(`./vlfeat_compare`)

# Load VLFeat results
vlfeat_json = JSON3.read(read("vlfeat_detections.json", String))
vlfeat_features = vlfeat_json["features"]
vlfeat_params = vlfeat_json["parameters"]

println("✓ VLFeat detected $(length(vlfeat_features)) features")
println("  Parameters:")
println("    first_octave: $(vlfeat_params["first_octave"])")
println("    octave_resolution: $(vlfeat_params["octave_resolution"])")
println("    peak_threshold: $(vlfeat_params["peak_threshold"])")
println("    edge_threshold: $(vlfeat_params["edge_threshold"])")
println()

# =============================================================================
# Step 2: Run Julia implementation with SAME parameters
# =============================================================================

println("Step 2: Running Julia implementation with matching parameters")
println("-"^80)

# Load same image
img = load("vlfeat_comparison/input.tif")
println("✓ Loaded test image: $(size(img))")

# Create scale space with EXACT same parameters as VLFeat
ss = ScaleSpace(img;
    first_octave=vlfeat_params["first_octave"],
    octave_resolution=vlfeat_params["octave_resolution"],
    first_subdivision=-1,
    last_subdivision=3)

println("✓ Created ScaleSpace")
println()

# Compute Hessian responses
println("Computing Hessian determinant responses...")
hessian_resp = ScaleSpaceResponse(ss, DERIVATIVE_KERNELS.xx)

for level in ss
    step = 2.0^level.octave
    det_data = vlfeat_hessian_det(level.data, level.sigma, step)
    resp_level = hessian_resp[level.octave, level.subdivision]
    resp_level.data .= Gray{Float32}.(det_data)
end
println("✓ Computed Hessian responses")
println()

# Detect with discrete extrema output
println("Detecting extrema with discrete output...")
extrema, discrete = detect_extrema(hessian_resp;
    peak_threshold=Float64(vlfeat_params["peak_threshold"]),
    edge_threshold=Float64(vlfeat_params["edge_threshold"]),
    base_scale=Float64(vlfeat_params["base_scale"]),
    octave_resolution=Int(vlfeat_params["octave_resolution"]),
    return_discrete=true)

println("✓ Julia detected:")
println("  Discrete extrema: $(length(discrete))")
println("  Refined extrema:  $(length(extrema))")
println()

# =============================================================================
# Step 3: Compare discrete extrema count
# =============================================================================

println("="^80)
println("COMPARISON 1: Discrete Extrema Count (before refinement)")
println("="^80)
println()

@printf("VLFeat features (after refinement): %4d\n", length(vlfeat_features))
@printf("Julia discrete extrema:              %4d\n", length(discrete))
@printf("Julia refined extrema:               %4d\n", length(extrema))
println()

if length(discrete) != length(vlfeat_features)
    println("⚠ Discrete extrema count differs!")
    println("  This suggests a difference in the initial extrema detection")
    println("  or that VLFeat's filtering happens at a different stage.")
    println()
end

# =============================================================================
# Step 4: Compare refined feature positions
# =============================================================================

println("="^80)
println("COMPARISON 2: Refined Feature Positions")
println("="^80)
println()

# Try to match Julia refined features with VLFeat features
matches = []
for (i, vf) in enumerate(vlfeat_features)
    vf_x = vf["x"]
    vf_y = vf["y"]
    vf_sigma = vf["sigma"]

    # Find closest Julia feature
    min_dist = Inf
    best_match = nothing

    for je in extrema
        # Convert to same coordinate system
        # NOTE: Julia uses column-major (y,x) while VLFeat/C uses row-major (x,y)
        # So we need to swap: Julia's x → VLFeat's y, Julia's y → VLFeat's x
        je_x_img = je.y / je.step  # Julia's y becomes VLFeat's x
        je_y_img = je.x / je.step  # Julia's x becomes VLFeat's y
        je_sigma = je.sigma

        dist = sqrt((je_x_img - vf_x)^2 + (je_y_img - vf_y)^2)
        sigma_diff = abs(je_sigma - vf_sigma)

        # Relax sigma matching since VLFeat may compute it differently
        if dist < min_dist && sigma_diff < 0.5
            min_dist = dist
            best_match = (extremum=je, je_x=je_x_img, je_y=je_y_img)
        end
    end

    if best_match !== nothing && min_dist < 2.0
        push!(matches, (vlfeat=vf, julia=best_match, dist=min_dist, idx=i))
    end
end

@printf("Matched features: %d / %d (%.1f%%)\n",
        length(matches), length(vlfeat_features),
        100*length(matches)/length(vlfeat_features))
println()

if !isempty(matches)
    distances = [m.dist for m in matches]
    @printf("Position match statistics:\n")
    @printf("  Min distance:    %.6f pixels\n", minimum(distances))
    @printf("  Max distance:    %.6f pixels\n", maximum(distances))
    @printf("  Mean distance:   %.6f pixels\n", mean(distances))
    @printf("  Median distance: %.6f pixels\n", median(distances))
    println()
end

# =============================================================================
# Step 5: Compare peak scores
# =============================================================================

println("="^80)
println("COMPARISON 3: Peak Scores")
println("="^80)
println()

if !isempty(matches)
    peak_diffs = []
    for m in matches
        vf_peak = m.vlfeat["peakScore"]
        jl_peak = m.julia.peakScore
        push!(peak_diffs, abs(vf_peak - jl_peak))
    end

    @printf("Peak score differences (for matched features):\n")
    @printf("  Min diff:    %.6e\n", minimum(peak_diffs))
    @printf("  Max diff:    %.6e\n", maximum(peak_diffs))
    @printf("  Mean diff:   %.6e\n", mean(peak_diffs))
    @printf("  Median diff: %.6e\n", median(peak_diffs))
    println()
end

# =============================================================================
# Step 6: Compare edge scores
# =============================================================================

println("="^80)
println("COMPARISON 4: Edge Scores")
println("="^80)
println()

if !isempty(matches)
    edge_diffs = []
    for m in matches
        vf_edge = m.vlfeat["edgeScore"]
        jl_edge = m.julia.edgeScore
        push!(edge_diffs, abs(vf_edge - jl_edge))
    end

    @printf("Edge score differences (for matched features):\n")
    @printf("  Min diff:    %.6e\n", minimum(edge_diffs))
    @printf("  Max diff:    %.6e\n", maximum(edge_diffs))
    @printf("  Mean diff:   %.6e\n", mean(edge_diffs))
    @printf("  Median diff: %.6e\n", median(edge_diffs))
    println()
end

# =============================================================================
# Step 7: Show detailed comparison for first 5 matches
# =============================================================================

println("="^80)
println("DETAILED COMPARISON: First 5 Matched Features")
println("="^80)
println()

for (i, m) in enumerate(matches[1:min(5, length(matches))])
    println("Feature $i:")
    println("  VLFeat: (%.2f, %.2f) σ=%.3f peak=%.6e edge=%.3f" %
            (m.vlfeat["x"], m.vlfeat["y"], m.vlfeat["sigma"],
             m.vlfeat["peakScore"], m.vlfeat["edgeScore"]))

    jx = m.julia.x / m.julia.step
    jy = m.julia.y / m.julia.step
    println("  Julia:  (%.2f, %.2f) σ=%.3f peak=%.6e edge=%.3f" %
            (jx, jy, m.julia.sigma, m.julia.peakScore, m.julia.edgeScore))

    println("  Δpos: %.4f pixels, Δpeak: %.6e, Δedge: %.4f" %
            (m.dist,
             abs(m.vlfeat["peakScore"] - m.julia.peakScore),
             abs(m.vlfeat["edgeScore"] - m.julia.edgeScore)))
    println()
end

# =============================================================================
# Summary
# =============================================================================

println("="^80)
println("SUMMARY")
println("="^80)
println()

if length(extrema) == length(vlfeat_features) &&
   length(matches) == length(vlfeat_features) &&
   mean([m.dist for m in matches]) < 0.001
    println("✓✓✓ EXCELLENT: Julia implementation matches VLFeat!")
    println("    All features detected with sub-pixel accuracy")
elseif length(matches) >= 0.9 * length(vlfeat_features)
    println("✓ GOOD: Most features match (>90%)")
    println("  Minor differences in:")
    if length(extrema) != length(vlfeat_features)
        println("  - Feature count")
    end
    if !isempty(matches) && mean([m.dist for m in matches]) > 0.01
        println("  - Position accuracy")
    end
else
    println("⚠ NEEDS INVESTIGATION: Significant differences detected")
    println()
    println("Likely issues:")
    println("  1. Different threshold application")
    println("  2. Refinement algorithm differences")
    println("  3. Edge score calculation differences")
    println()
    println("Next steps:")
    println("  - Check discrete extrema before refinement")
    println("  - Compare refinement step-by-step")
    println("  - Verify threshold values match exactly")
end

println()
println("="^80)
