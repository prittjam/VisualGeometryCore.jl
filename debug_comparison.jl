#!/usr/bin/env julia
"""
Debug comparison - show raw values from both implementations
"""

using VisualGeometryCore
using FileIO
using JSON3
using Printf

# Load VLFeat results
vlfeat_json = JSON3.read(read("vlfeat_detections.json", String))
vlfeat_features = vlfeat_json["features"]
vlfeat_params = vlfeat_json["parameters"]

# Run Julia
img = load("vlfeat_comparison/input.tif")
ss = ScaleSpace(img;
    first_octave=vlfeat_params["first_octave"],
    octave_resolution=vlfeat_params["octave_resolution"],
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

println("="^80)
println("DEBUG COMPARISON")
println("="^80)
println()

println("First 5 VLFeat features:")
for (i, vf) in enumerate(vlfeat_features[1:min(5, length(vlfeat_features))])
    @printf("  %d: x=%.4f y=%.4f σ=%.4f peak=%.6e edge=%.4f\n",
            i, vf["x"], vf["y"], vf["sigma"], vf["peakScore"], vf["edgeScore"])
end
println()

println("First 5 Julia extrema:")
for (i, je) in enumerate(extrema[1:min(5, length(extrema))])
    @printf("  %d: x=%.4f y=%.4f z=%.2f σ=%.4f step=%.2f (img: x=%.4f y=%.4f) peak=%.6e edge=%.4f\n",
            i, je.x, je.y, je.z, je.sigma, je.step,
            je.x/je.step, je.y/je.step, je.peakScore, je.edgeScore)
end
println()

println("Julia sigma values for comparison:")
for je in extrema[1:min(5, length(extrema))]
    println("  σ=$(je.sigma)")
end
