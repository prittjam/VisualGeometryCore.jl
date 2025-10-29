#!/usr/bin/env julia
"""
Test Harris-Laplace detection on integer-coordinate blob pattern.

This test creates a pattern with blobs at exact integer coordinates (in Colmap convention),
then runs detection and measures the bias/error.
"""

using BlobBoards
using VisualGeometryCore
using Unitful: ustrip, mm
using FileIO
using Statistics
using LinearAlgebra

println("="^70)
println("Testing Harris-Laplace on Integer-Coordinate Pattern")
println("="^70)

# Generate pattern with integer coordinates (Colmap convention)
println("\n1. Generating pattern with integer coordinates...")
pattern, ground_truth_blobs, config = blob_pattern(400, 300; 
    integer_coordinates=true,
    min_scale=5pd,
    dir="/tmp",
    seed=UInt64(0xABCD1234))

n_gt = length(ground_truth_blobs)
println("   Generated $n_gt blobs with integer centers")

# Verify all coordinates are integers
all_integer = all(ground_truth_blobs) do blob
    x = ustrip(blob.center[1])
    y = ustrip(blob.center[2])
    isinteger(x) && isinteger(y)
end
println("   All coordinates are integers: $all_integer")

# Show first few ground truth positions
println("\n   First 5 ground truth positions (Colmap convention):")
for (i, blob) in enumerate(ground_truth_blobs[1:min(5, n_gt)])
    x = ustrip(blob.center[1])
    y = ustrip(blob.center[2])
    σ = ustrip(blob.σ)
    println("     GT $i: ($(x), $(y)) σ=$σ")
end

# Use the pattern image we already have in memory
img = pattern
println("\n2. Running Harris-Laplace detection on $(size(img))...")

# Run detection (returns in Colmap convention by default)
detected = detect_features(img;
    method=:hessian_laplace,
    peak_threshold=0.003,
    edge_threshold=10.0,
    first_octave=-1,
    octave_resolution=3,
    first_subdivision=-1,
    last_subdivision=3,
    base_scale=2.015874)

n_detected = length(detected)
println("   Detected $n_detected blobs")

# Show first few detections
println("\n   First 5 detected positions (Colmap convention):")
for (i, blob) in enumerate(detected[1:min(5, n_detected)])
    x = ustrip(blob.center[1])
    y = ustrip(blob.center[2])
    σ = ustrip(blob.σ)
    resp = blob.response
    println("     Det $i: ($(x), $(y)) σ=$σ response=$resp")
end

# Match ground truth to detections using nearest neighbor
println("\n3. Matching ground truth to detections...")
using NearestNeighbors

# Build KDTree from detections
det_positions = hcat([[ustrip(b.center[1]), ustrip(b.center[2])] for b in detected]...)
kdtree = KDTree(det_positions)

# Match each ground truth blob to nearest detection
matches = []
for gt_blob in ground_truth_blobs
    gt_x = ustrip(gt_blob.center[1])
    gt_y = ustrip(gt_blob.center[2])
    gt_σ = ustrip(gt_blob.σ)
    
    # Find nearest detection
    idx, dist = knn(kdtree, [gt_x, gt_y], 1)
    det_blob = detected[idx[1]]
    
    det_x = ustrip(det_blob.center[1])
    det_y = ustrip(det_blob.center[2])
    det_σ = ustrip(det_blob.σ)
    
    # Only accept matches within 3 pixels
    if dist[1] < 3.0
        push!(matches, (
            gt_x=gt_x, gt_y=gt_y, gt_σ=gt_σ,
            det_x=det_x, det_y=det_y, det_σ=det_σ,
            dist=dist[1]
        ))
    end
end

n_matched = length(matches)
match_rate = 100.0 * n_matched / n_gt
println("   Matched: $n_matched / $n_gt ($(round(match_rate, digits=1))%)")

# Compute errors
if n_matched > 0
    x_errors = [m.det_x - m.gt_x for m in matches]
    y_errors = [m.det_y - m.gt_y for m in matches]
    pos_errors = [sqrt((m.det_x - m.gt_x)^2 + (m.det_y - m.gt_y)^2) for m in matches]
    σ_errors = [abs(m.det_σ - m.gt_σ) for m in matches]
    
    println("\n4. Error Analysis:")
    println("   Position errors:")
    println("     X bias: $(round(mean(x_errors), digits=4)) px (std: $(round(std(x_errors), digits=4)))")
    println("     Y bias: $(round(mean(y_errors), digits=4)) px (std: $(round(std(y_errors), digits=4)))")
    println("     Distance RMS: $(round(sqrt(mean(pos_errors.^2)), digits=4)) px")
    println("     Distance max: $(round(maximum(pos_errors), digits=4)) px")
    println("     Distance mean: $(round(mean(pos_errors), digits=4)) px")
    
    println("\n   Scale errors:")
    println("     Sigma RMS: $(round(sqrt(mean(σ_errors.^2)), digits=4))")
    println("     Sigma max: $(round(maximum(σ_errors), digits=4))")
    println("     Sigma mean: $(round(mean(σ_errors), digits=4))")
    
    # Analyze directional bias
    angles = [atan(m.det_y - m.gt_y, m.det_x - m.gt_x) * 180/π for m in matches]
    mean_angle = atan(mean(y_errors), mean(x_errors)) * 180/π
    
    println("\n   Directional bias:")
    println("     Mean angle: $(round(mean_angle, digits=1))° (0°=East, 90°=North)")
    println("     Mean offset vector: ($(round(mean(x_errors), digits=4)), $(round(mean(y_errors), digits=4)))")
    
    # Show first few matches in detail
    println("\n5. First 5 matches (GT → Detection):")
    for (i, m) in enumerate(matches[1:min(5, n_matched)])
        dx = m.det_x - m.gt_x
        dy = m.det_y - m.gt_y
        println("     Match $i: ($(m.gt_x), $(m.gt_y)) → ($(round(m.det_x, digits=3)), $(round(m.det_y, digits=3)))")
        println("              Δ=($(round(dx, digits=3)), $(round(dy, digits=3))) dist=$(round(m.dist, digits=3))")
    end
    
    println("\n" * "="^70)
    println("Summary:")
    println("  Detection rate: $(round(match_rate, digits=1))%")
    println("  Position RMS: $(round(sqrt(mean(pos_errors.^2)), digits=4)) px")
    println("  Mean bias: ($(round(mean(x_errors), digits=4)), $(round(mean(y_errors), digits=4))) px")
    println("  Scale RMS: $(round(sqrt(mean(σ_errors.^2)), digits=4))")
    println("="^70)
else
    println("\n❌ No matches found!")
end
