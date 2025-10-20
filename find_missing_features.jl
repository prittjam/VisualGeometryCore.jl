#!/usr/bin/env julia

using VisualGeometryCore
using FileIO
using JSON3
using Printf

# Load and detect
vlfeat_json = JSON3.read(read("vlfeat_detections.json", String))
img = load("vlfeat_comparison/input.tif")
ss = ScaleSpace(img; first_octave=-1, octave_resolution=3, first_subdivision=-1, last_subdivision=3)

hessian_resp = ScaleSpaceResponse(ss, DERIVATIVE_KERNELS.xx)
for level in ss
    step = 2.0^level.octave
    det_data = vlfeat_hessian_det(level.data, level.sigma, step)
    resp_level = hessian_resp[level.octave, level.subdivision]
    resp_level.data .= Gray{Float32}.(det_data)
end

extrema = detect_extrema(hessian_resp; peak_threshold=0.003, edge_threshold=10.0, base_scale=2.015874, octave_resolution=3)
extrema_in_range = filter(e -> e.octave >= -1 && e.octave <= 2, extrema)

# Find matched VLFeat features
matched = zeros(Bool, length(vlfeat_json["features"]))
for je in extrema_in_range
    je_x = (je.x - 1) * je.step
    je_y = (je.y - 1) * je.step

    for (j, vf) in enumerate(vlfeat_json["features"])
        dist = sqrt((je_x - vf["x"])^2 + (je_y - vf["y"])^2)
        if dist < 0.01
            matched[j] = true
            break
        end
    end
end

@printf("Matched: %d / %d\n", sum(matched), length(vlfeat_json["features"]))
println()

if sum(matched) < length(vlfeat_json["features"])
    println("Unmatched VLFeat features:")
    for (i, vf) in enumerate(vlfeat_json["features"])
        if !matched[i]
            @printf("  #%2d: (%.3f, %.3f) σ=%.3f peak=%.6e edge=%.3f\n",
                    i, vf["x"], vf["y"], vf["sigma"], vf["peakScore"], vf["edgeScore"])
        end
    end
else
    println("✓✓✓ ALL VLFeat features matched!")
end
