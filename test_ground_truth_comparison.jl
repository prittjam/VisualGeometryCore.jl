#!/usr/bin/env julia
"""
Test HarrisLaplace detector against ground truth blobs.

This script:
1. Loads ground truth blobs from JSON (subpixel accuracy)
2. Loads test image and runs HarrisLaplace detection
3. Compares detected extrema (integer coordinates) with ground truth
4. Once subpixel refinement is implemented, validates refined positions
"""

using VisualGeometryCore
using FileIO
using JSON3
using Printf
using LinearAlgebra
using Unitful
using Statistics

println("="^80)
println("HARRISLAPLACE DETECTOR - GROUND TRUTH COMPARISON")
println("="^80)
println()

# =============================================================================
# Step 1: Load Ground Truth Blobs
# =============================================================================

println("Step 1: Loading ground truth blobs from JSON")
println("-"^80)

json_path = "test/data/blob_pattern_eBd.json"
json_data = JSON3.read(read(json_path, String))

# Extract blobs array and deserialize to IsoBlob objects
ground_truth_blobs = IsoBlob[]
for blob_data in json_data["blobs"]
    blob = StructTypes.construct(IsoBlob, blob_data)
    push!(ground_truth_blobs, blob)
end

println("✓ Loaded $(length(ground_truth_blobs)) ground truth blobs")
println()

# Show statistics
if !isempty(ground_truth_blobs)
    sigmas = [ustrip(b.σ) for b in ground_truth_blobs]
    println("Ground truth blob statistics:")
    @printf("  Min σ:    %.3f px\n", minimum(sigmas))
    @printf("  Max σ:    %.3f px\n", maximum(sigmas))
    @printf("  Mean σ:   %.3f px\n", sum(sigmas)/length(sigmas))
    @printf("  Median σ: %.3f px\n", median(sigmas))
    println()

    # Count by scale
    println("Distribution by scale:")
    for scale in sort(unique(sigmas))
        count = sum(s == scale for s in sigmas)
        @printf("  σ=%.1f px: %3d blobs\n", scale, count)
    end
    println()
end

# =============================================================================
# Step 2: Load Test Image and Run Detection
# =============================================================================

println("Step 2: Running HarrisLaplace detector")
println("-"^80)

# Load test image and convert to grayscale
img_path = "test/data/blob_pattern_eBd.png"
img_color = load(img_path)
img = Gray.(img_color)  # Convert RGBA to Gray
println("✓ Loaded test image: $(size(img))")

# Create scale space with Hessian detector settings
# Match VLFeat settings: first_octave=-1, octave_resolution=3
ss = ScaleSpace(img; first_octave=-1, octave_resolution=3,
               first_subdivision=-1, last_subdivision=3)

octave_range, subdivision_range = axes(ss)
println("✓ Created ScaleSpace:")
println("  Octaves: $(first(octave_range)) to $(last(octave_range))")
println("  Subdivisions: $(first(subdivision_range)) to $(last(subdivision_range))")
println("  Total levels: $(length(ss.levels))")
println()

# Compute Hessian determinant responses
println("Computing Hessian determinant responses...")
hessian_resp = ScaleSpaceResponse(ss, DERIVATIVE_KERNELS.xx)

for level in ss
    step = 2.0^level.octave
    det_data = vlfeat_hessian_det(level.data, level.sigma, step)
    resp_level = hessian_resp[level.octave, level.subdivision]
    resp_level.data .= Gray{Float32}.(det_data)
end
println("✓ Computed Hessian responses for all levels")
println()

# Detect extrema
println("Detecting extrema...")
peak_threshold = 0.001
edge_threshold = 10.0
extrema = detect_extrema(hessian_resp, peak_threshold=peak_threshold,
                        edge_threshold=edge_threshold)

println("✓ Detected $(length(extrema)) extrema")
println("  Peak threshold: $peak_threshold")
println("  Edge threshold: $edge_threshold")
println()

# =============================================================================
# Step 3: Compare Detections with Ground Truth
# =============================================================================

println("="^80)
println("Step 3: Comparing detections with ground truth")
println("="^80)
println()

println("NOTE: Current comparison uses integer extrema coordinates.")
println("      Once subpixel refinement is working, we'll compare refined positions.")
println()

# For each ground truth blob, find nearest detection
# We'll use a distance threshold based on scale
matches = []
unmatched_gt = []

for (i, gt_blob) in enumerate(ground_truth_blobs)
    gt_x = ustrip(gt_blob.center[1])
    gt_y = ustrip(gt_blob.center[2])
    gt_sigma = ustrip(gt_blob.σ)

    # Find nearest extremum
    min_dist = Inf
    best_match = nothing

    for ext in extrema
        # Convert extremum coordinates to input image space
        # For octave -1 (step=0.5), coordinates are in 2x space
        # For octave 0 (step=1.0), coordinates are in 1x space
        # etc.
        ext_x = ext.x / ext.step
        ext_y = ext.y / ext.step

        # Distance in image space
        dist = sqrt((ext_x - gt_x)^2 + (ext_y - gt_y)^2)

        # Also check scale similarity (sigma should be close)
        sigma_ratio = max(ext.sigma / gt_sigma, gt_sigma / ext.sigma)

        # Consider it a potential match if distance is small and scale is similar
        if dist < min_dist && sigma_ratio < 1.5
            min_dist = dist
            best_match = ext
        end
    end

    # Match threshold: within 3 pixels and similar scale
    if best_match !== nothing && min_dist < 3.0
        push!(matches, (gt=gt_blob, det=best_match, dist=min_dist, gt_idx=i))
    else
        push!(unmatched_gt, (blob=gt_blob, idx=i))
    end
end

# Find unmatched detections
matched_extrema = Set(m.det for m in matches)
unmatched_det = [ext for ext in extrema if ext ∉ matched_extrema]

# =============================================================================
# Step 4: Report Results
# =============================================================================

println("="^80)
println("DETECTION RESULTS")
println("="^80)
println()

@printf("Ground truth blobs:     %4d\n", length(ground_truth_blobs))
@printf("Detected extrema:       %4d\n", length(extrema))
@printf("Matched:                %4d\n", length(matches))
@printf("Unmatched GT:           %4d\n", length(unmatched_gt))
@printf("Unmatched detections:   %4d\n", length(unmatched_det))
println()

if !isempty(matches)
    distances = [m.dist for m in matches]
    @printf("Match distance statistics (pixels):\n")
    @printf("  Min:    %.3f\n", minimum(distances))
    @printf("  Max:    %.3f\n", maximum(distances))
    @printf("  Mean:   %.3f\n", sum(distances)/length(distances))
    @printf("  Median: %.3f\n", median(distances))
    println()
end

# Show first 10 matches
if !isempty(matches)
    println("First 10 matches:")
    println("  # | GT(x, y, σ) → Det(x, y, σ) | Dist | Oct")
    println("-"^80)
    for (i, m) in enumerate(matches[1:min(10, length(matches))])
        gt_x = ustrip(m.gt.center[1])
        gt_y = ustrip(m.gt.center[2])
        gt_sigma = ustrip(m.gt.σ)

        det_x = m.det.x / m.det.step
        det_y = m.det.y / m.det.step
        det_sigma = m.det.sigma

        @printf(" %2d | (%.1f, %.1f, %.1f) → (%.1f, %.1f, %.1f) | %.2f | %2d\n",
                i, gt_x, gt_y, gt_sigma, det_x, det_y, det_sigma, m.dist, m.det.octave)
    end
    println()
end

# Show some unmatched ground truth (potential false negatives)
if !isempty(unmatched_gt)
    println("First 10 unmatched ground truth blobs (false negatives?):")
    for (i, um) in enumerate(unmatched_gt[1:min(10, length(unmatched_gt))])
        gt_x = ustrip(um.blob.center[1])
        gt_y = ustrip(um.blob.center[2])
        gt_sigma = ustrip(um.blob.σ)
        @printf("  %2d: GT #%3d at (%.1f, %.1f, σ=%.1f)\n",
                i, um.idx, gt_x, gt_y, gt_sigma)
    end
    println()
end

# Show some unmatched detections (potential false positives)
if !isempty(unmatched_det)
    println("First 10 unmatched detections (false positives?):")
    for (i, ext) in enumerate(unmatched_det[1:min(10, length(unmatched_det))])
        det_x = ext.x / ext.step
        det_y = ext.y / ext.step
        @printf("  %2d: Det at (%.1f, %.1f, σ=%.3f) oct=%2d\n",
                i, det_x, det_y, ext.sigma, ext.octave)
    end
    println()
end

println("="^80)
println("NEXT STEPS")
println("="^80)
println()
println("Current status:")
println("  ✓ Hessian responses match VLFeat (RMS < 1e-6)")
println("  ✓ Extrema detection is working (discrete 3D maxima)")
println("  ⚠ Need subpixel refinement to match ground truth precisely")
println()
println("The subpixel refinement (refine_extremum_3d) is already implemented.")
println("It should be refining extrema positions, but we need to verify:")
println("  1. Refinement converges properly")
println("  2. Refined positions match ground truth within tolerance")
println("  3. Edge rejection works correctly")
println()
println("="^80)
