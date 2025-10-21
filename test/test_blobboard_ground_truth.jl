#!/usr/bin/env julia
"""
Test blob detection accuracy against BlobBoard ground truth.

This test measures detection accuracy (RMS error to true blob positions)
rather than implementation correctness (matching VLFeat).
"""

using Test
using VisualGeometryCore
using VisualGeometryCore: BlobFeature, SaddleFeature
using FileIO
using JSON3
using Printf
using Colors
using Unitful

@testset "BlobBoard Ground Truth Validation" begin
    # Load the BlobBoard image and ground truth
    img_path = joinpath(@__DIR__, "data", "blob_pattern_eBd.png")
    gt_path = joinpath(@__DIR__, "data", "blob_pattern_eBd.json")

    if !isfile(img_path) || !isfile(gt_path)
        @warn "BlobBoard test data not found, skipping ground truth validation"
        return
    end

    println("\nLoading BlobBoard ground truth...")
    img_raw = load(img_path)
    # Convert to grayscale if needed
    img = if eltype(img_raw) <: Gray
        Gray{Float32}.(img_raw)
    else
        # Extract luminance from color/RGBA
        Gray{Float32}.(map(c -> Gray(c), img_raw))
    end

    gt_json = JSON3.read(read(gt_path, String))

    # Extract ground truth blob positions and scales
    gt_blobs = [(
        x = blob["center"][1]["value"],
        y = blob["center"][2]["value"],
        σ = blob["σ"]["value"]
    ) for blob in gt_json["blobs"]]

    num_gt_blobs = length(gt_blobs)
    println("  Ground truth blobs: $num_gt_blobs")
    println("  Image size: $(size(img))")

    # Build scale space and detect blobs
    println("\nDetecting blobs...")
    ss = ScaleSpace(img, first_octave=-1, octave_resolution=3,
                   first_subdivision=-1, last_subdivision=3)

    # Compute Hessian determinant response using VLFeat
    hessian_resp = ScaleSpaceResponse(ss, DERIVATIVE_KERNELS.xx)
    laplacian_resp = ScaleSpaceResponse(ss, DERIVATIVE_KERNELS.xx)
    for level in ss
        step = 2.0^level.octave
        det_data = vlfeat_hessian_det(level.data, level.sigma, step)
        lap_data = vlfeat_laplacian(level.data, level.sigma, step)

        resp_level = hessian_resp[level.octave, level.subdivision]
        resp_level.data .= Gray{Float32}.(det_data)

        lap_level = laplacian_resp[level.octave, level.subdivision]
        lap_level.data .= Gray{Float32}.(lap_data)
    end

    # Detect extrema with Laplacian for blob polarity
    extrema = detect_extrema(hessian_resp;
        peak_threshold=0.003,
        edge_threshold=10.0,
        base_scale=2.015874,
        octave_resolution=3,
        laplacian_resp=laplacian_resp)

    num_detected = length(extrema)
    println("  Detected blobs: $num_detected")

    # Convert extrema to IsoBlobDetection
    blob_detections = [IsoBlobDetection(ext) for ext in extrema]

    # Count polarities for reporting
    num_blobs = count(b -> b.polarity == BlobFeature, blob_detections)
    num_saddles = length(blob_detections) - num_blobs
    num_bright = count(b -> b.laplacian_sign == -1, blob_detections)
    num_dark = count(b -> b.laplacian_sign == 1, blob_detections)
    num_neutral = count(b -> b.laplacian_sign == 0, blob_detections)

    println("  Blob-like (positive det(H)): $num_blobs")
    println("  Saddle/edge-like (negative det(H)): $num_saddles")
    println("  Bright blobs (Laplacian < 0): $num_bright")
    println("  Dark blobs (Laplacian > 0): $num_dark")
    println("  Neutral (Laplacian ≈ 0): $num_neutral")

    # Convert to simple format for matching (use ALL detections)
    detected_blobs = [(
        x = ustrip(b.center[1]),
        y = ustrip(b.center[2]),
        σ = ustrip(b.σ),
        polarity = b.polarity,
        laplacian_sign = b.laplacian_sign,
        response = b.response
    ) for b in blob_detections]

    # Match detected blobs to ground truth (nearest neighbor matching)
    println("\nMatching detections to ground truth...")

    matches = []
    matched_gt_indices = Set{Int}()

    for det in detected_blobs
        # Find nearest ground truth blob
        best_dist = Inf
        best_gt_idx = 0

        for (i, gt) in enumerate(gt_blobs)
            if i in matched_gt_indices
                continue  # Already matched
            end

            # Distance in (x, y, σ) space
            # Weight position and scale appropriately
            pos_dist = sqrt((det.x - gt.x)^2 + (det.y - gt.y)^2)
            scale_dist = abs(det.σ - gt.σ)

            # Combined distance (position is more important)
            dist = pos_dist + 0.1 * scale_dist

            if dist < best_dist
                best_dist = dist
                best_gt_idx = i
            end
        end

        # Accept match if close enough (within 3 pixels position, any scale)
        if best_dist < 5.0 && best_gt_idx > 0
            gt = gt_blobs[best_gt_idx]
            pos_error = sqrt((det.x - gt.x)^2 + (det.y - gt.y)^2)
            scale_error = abs(det.σ - gt.σ)

            push!(matches, (
                det = det,
                gt = gt,
                pos_error = pos_error,
                scale_error = scale_error,
                scale_ratio = det.σ / gt.σ
            ))

            push!(matched_gt_indices, best_gt_idx)
        end
    end

    num_matched = length(matches)
    recall = num_matched / num_gt_blobs
    precision = num_matched / num_detected

    # Count polarity of matched blobs
    num_blob_matched = count(m -> m.det.polarity == BlobFeature, matches)
    num_saddle_matched = num_matched - num_blob_matched
    num_bright_matched = count(m -> m.det.laplacian_sign == -1, matches)
    num_dark_matched = count(m -> m.det.laplacian_sign == 1, matches)

    println("  Matched: $num_matched / $num_gt_blobs ground truth blobs")
    @printf("  Recall: %.1f%% (of ground truth)\n", 100 * recall)
    @printf("  Precision: %.1f%% (of all detections)\n", 100 * precision)
    println("  Matched types: $num_blob_matched blob-like, $num_saddle_matched saddle-like")
    println("  Matched intensity: $num_bright_matched bright, $num_dark_matched dark")

    # Compute RMS errors for matched blobs
    if !isempty(matches)
        pos_errors = [m.pos_error for m in matches]
        scale_errors = [m.scale_error for m in matches]
        scale_ratios = [m.scale_ratio for m in matches]

        rms_position = sqrt(sum(pos_errors.^2) / length(pos_errors))
        rms_scale = sqrt(sum(scale_errors.^2) / length(scale_errors))
        mean_scale_ratio = sum(scale_ratios) / length(scale_ratios)

        # Separate statistics for dark blobs only
        dark_matches = filter(m -> m.det.laplacian_sign == 1, matches)
        if !isempty(dark_matches)
            dark_pos_errors = [m.pos_error for m in dark_matches]
            dark_scale_errors = [m.scale_error for m in dark_matches]
            dark_scale_ratios = [m.scale_ratio for m in dark_matches]

            dark_rms_position = sqrt(sum(dark_pos_errors.^2) / length(dark_pos_errors))
            dark_rms_scale = sqrt(sum(dark_scale_errors.^2) / length(dark_scale_errors))
            dark_mean_scale_ratio = sum(dark_scale_ratios) / length(dark_scale_ratios)
        end

        println("\n" * "="^80)
        println("RMS ERRORS vs GROUND TRUTH (ALL MATCHES)")
        println("="^80)
        @printf("  Position (x,y): %.4f px\n", rms_position)
        @printf("  Scale (σ):      %.4f px\n", rms_scale)
        @printf("  Mean scale ratio: %.4f\n", mean_scale_ratio)
        println()
        @printf("  Max position error: %.4f px\n", maximum(pos_errors))
        @printf("  Max scale error:    %.4f px\n", maximum(scale_errors))
        @printf("  Median position error: %.4f px\n", sort(pos_errors)[div(length(pos_errors), 2)])
        @printf("  Median scale error:    %.4f px\n", sort(scale_errors)[div(length(scale_errors), 2)])
        println("="^80)

        if !isempty(dark_matches)
            println("\n" * "="^80)
            println("RMS ERRORS vs GROUND TRUTH (DARK BLOBS ONLY - $(length(dark_matches)) matches)")
            println("="^80)
            @printf("  Position (x,y): %.4f px\n", dark_rms_position)
            @printf("  Scale (σ):      %.4f px\n", dark_rms_scale)
            @printf("  Mean scale ratio: %.4f\n", dark_mean_scale_ratio)
            println()
            @printf("  Max position error: %.4f px\n", maximum(dark_pos_errors))
            @printf("  Max scale error:    %.4f px\n", maximum(dark_scale_errors))
            @printf("  Median position error: %.4f px\n", sort(dark_pos_errors)[div(length(dark_pos_errors), 2)])
            @printf("  Median scale error:    %.4f px\n", sort(dark_scale_errors)[div(length(dark_scale_errors), 2)])

            # Check for systematic offset
            x_offsets = [m.det.x - m.gt.x for m in dark_matches]
            y_offsets = [m.det.y - m.gt.y for m in dark_matches]
            mean_x_offset = sum(x_offsets) / length(x_offsets)
            mean_y_offset = sum(y_offsets) / length(y_offsets)

            println()
            println("  Systematic offset (mean):")
            @printf("    X offset: %.4f px\n", mean_x_offset)
            @printf("    Y offset: %.4f px\n", mean_y_offset)
            @printf("    Magnitude: %.4f px\n", sqrt(mean_x_offset^2 + mean_y_offset^2))
            println("="^80)
        end

        # Tests
        @testset "Detection Accuracy" begin
            @test recall > 0.95  # Should detect at least 95% of blobs
            @test rms_position < 1.5  # RMS position error < 1.5 pixels
            @test rms_scale < 0.2  # RMS scale error < 0.2 pixels
            @test 0.95 < mean_scale_ratio < 1.05  # Scale ratio within 5%
        end

        @testset "RMS Error Bounds" begin
            @test rms_position < 2.0  # RMS position error bound
            @test maximum(pos_errors) < 5.0  # Max position error (outliers)
            @test rms_scale < 0.3  # RMS scale error bound
        end
    else
        @warn "No matches found! Detection may have failed."
        @test num_matched > 0
    end

    # Print some examples
    if num_matched > 0
        println("\nExample matches (first 5):")
        for (i, m) in enumerate(matches[1:min(5, num_matched)])
            @printf("  #%d: GT=(%.2f, %.2f, σ=%.2f) → Det=(%.2f, %.2f, σ=%.2f) | err: pos=%.3f, σ=%.3f\n",
                    i, m.gt.x, m.gt.y, m.gt.σ, m.det.x, m.det.y, m.det.σ,
                    m.pos_error, m.scale_error)
        end
    end
end

println("\n✓ BlobBoard ground truth validation complete!")
