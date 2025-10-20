#!/usr/bin/env julia
"""
Investigate why VLFeat features #14 and #33 are missing from Julia detections.

Possible reasons:
1. Not found as discrete extrema
2. Refinement failed to converge
3. Failed peak or edge threshold
4. Different octave/subdivision
"""

using VisualGeometryCore
using FileIO
using JSON3
using Printf
using ImageCore: channelview

println("="^80)
println("INVESTIGATING MISSING VLFEAT FEATURES")
println("="^80)
println()

# Load VLFeat results
vlfeat_json = JSON3.read(read("vlfeat_detections.json", String))

# The two missing features
missing_ids = [14, 33]

for missing_id in missing_ids
    vf = vlfeat_json["features"][missing_id]

    println("="^80)
    @printf("VLFeat Feature #%d\n", missing_id)
    println("="^80)
    @printf("  Position: (%.3f, %.3f)\n", vf["x"], vf["y"])
    @printf("  Sigma:    %.3f\n", vf["sigma"])
    @printf("  Peak:     %.6e\n", vf["peakScore"])
    @printf("  Edge:     %.3f\n", vf["edgeScore"])
    println()
end

# Run Julia detection with discrete extrema output
img = load("vlfeat_comparison/input.tif")
ss = ScaleSpace(img; first_octave=-1, octave_resolution=3, first_subdivision=-1, last_subdivision=3)

hessian_resp = ScaleSpaceResponse(ss, DERIVATIVE_KERNELS.xx)
for level in ss
    step = 2.0^level.octave
    det_data = vlfeat_hessian_det(level.data, level.sigma, step)
    resp_level = hessian_resp[level.octave, level.subdivision]
    resp_level.data .= Gray{Float32}.(det_data)
end

println("Running Julia detection with discrete extrema tracking...")
extrema, discrete = detect_extrema(hessian_resp;
    peak_threshold=0.003,
    edge_threshold=10.0,
    base_scale=2.015874,
    octave_resolution=3,
    return_discrete=true)

@printf("Julia found %d discrete extrema\n", length(discrete))
@printf("Julia found %d refined extrema\n", length(extrema))
println()

# For each missing feature, check if we found a discrete extremum nearby
for missing_id in missing_ids
    vf = vlfeat_json["features"][missing_id]
    vf_x = vf["x"]
    vf_y = vf["y"]
    vf_sigma = vf["sigma"]

    println("="^80)
    @printf("Investigating VLFeat #%d at (%.3f, %.3f) σ=%.3f\n", missing_id, vf_x, vf_y, vf_sigma)
    println("="^80)
    println()

    # Convert VLFeat coords to Julia octave coords for each octave
    # vlfeat_coord = (julia_coord - 1) * step
    # => julia_coord = vlfeat_coord / step + 1

    println("Checking discrete extrema in each octave:")
    for octave in -1:2
        step = 2.0^octave

        # Convert to Julia coordinates in this octave
        julia_x = vf_x / step + 1
        julia_y = vf_y / step + 1

        # Find discrete extrema in this octave within a small radius
        discrete_in_octave = filter(d -> d[1] == octave, discrete)

        if !isempty(discrete_in_octave)
            nearby = []
            for (o, x, y, z) in discrete_in_octave
                dist = sqrt((x - julia_x)^2 + (y - julia_y)^2)
                if dist < 5.0  # within 5 pixels in octave space
                    push!(nearby, (x=x, y=y, z=z, dist=dist))
                end
            end

            if !isempty(nearby)
                sort!(nearby, by=n -> n.dist)
                @printf("  Octave %2d (step=%.1f): Found %d nearby discrete extrema\n",
                        octave, step, length(nearby))
                for (i, n) in enumerate(nearby[1:min(3, length(nearby))])
                    @printf("    %d. (%.1f, %.1f, z=%.1f) dist=%.2f in octave space\n",
                            i, n.x, n.y, n.z, n.dist)

                    # Check if this was refined
                    was_refined = false
                    for je in extrema
                        if je.octave == octave &&
                           abs(je.x - n.x) < 0.5 &&
                           abs(je.y - n.y) < 0.5 &&
                           abs(je.z - n.z) < 0.5
                            was_refined = true
                            @printf("       → REFINED: x=%.3f y=%.3f σ=%.3f peak=%.6e edge=%.3f\n",
                                    je.x, je.y, je.sigma, je.peakScore, je.edgeScore)

                            # Convert to VLFeat coords
                            je_vf_x = (je.x - 1) * je.step
                            je_vf_y = (je.y - 1) * je.step
                            @printf("       → VLFeat coords: (%.3f, %.3f)\n", je_vf_x, je_vf_y)
                            break
                        end
                    end

                    if !was_refined
                        println("       → NOT REFINED (refinement failed or filtered)")

                        # Try to manually check the response value
                        octave_idx = findfirst(oc -> oc.octave == octave, hessian_resp.octaves)
                        if octave_idx !== nothing
                            octave_data = hessian_resp.octaves[octave_idx]

                            ix = round(Int, n.x)
                            iy = round(Int, n.y)
                            iz = round(Int, n.z)

                            # Get subdivision from z
                            first_sub = first(octave_data.subdivisions)
                            subdivision = first_sub + iz - 1

                            # Check bounds
                            h, w, d = size(octave_data.G)
                            if 1 <= ix <= w && 1 <= iy <= h && 1 <= iz <= d
                                peak_val = Float32(octave_data.G[iy, ix, iz])
                                @printf("       → Response value: %.6e\n", peak_val)

                                if abs(peak_val) < 0.003
                                    println("       → REASON: Peak score below threshold (0.003)")
                                end
                            end
                        end
                    end
                end
            else
                @printf("  Octave %2d: No discrete extrema within 5 pixels\n", octave)
            end
        else
            @printf("  Octave %2d: No discrete extrema in this octave\n", octave)
        end
    end

    println()
end

println("="^80)
println("SUMMARY")
println("="^80)
println()
println("Next steps:")
println("  1. If discrete extrema were found but not refined:")
println("     → Check refinement convergence criteria")
println("  2. If discrete extrema not found at all:")
println("     → Check discrete extrema detection threshold")
println("  3. If refined but filtered:")
println("     → Check peak/edge threshold values")
println()
println("="^80)
