"""
Check what sigma values VLFeat is actually using for its levels.
"""

using FileIO
using ImageCore: channelview
using Colors
using Statistics

println("VLFeat Gaussian levels found:")
println("="^60)

for file in sort(readdir("vlfeat_comparison/vlfeat_gaussian", join=true))
    if endswith(file, ".tif")
        img = load(file)
        data = Float32.(channelview(img))

        # Extract octave and subdivision from filename
        # Format: gaussian_o{octave}_s{subdivision}.tif
        m = match(r"gaussian_o(-?\d+)_s(-?\d+)\.tif", basename(file))
        if m !== nothing
            octave = parse(Int, m.captures[1])
            subdivision = parse(Int, m.captures[2])

            # Calculate expected sigma
            # VLFeat formula: sigma = baseScale * 2^(o + s/octaveResolution)
            base_scale = 2.015874
            octave_resolution = 3
            expected_sigma = base_scale * 2.0^(octave + subdivision / octave_resolution)

            sz = size(data)
            println("o=$octave, s=$subdivision: size=$(sz[2])x$(sz[1]), σ=$(round(expected_sigma, digits=4)), mean=$(round(mean(data), digits=5))")
        end
    end
end

println()
println("Julia levels for comparison:")
println("="^60)

using VisualGeometryCore

img = load("vlfeat_comparison/input.tif")
ss = ScaleSpace(img)

for level in ss
    sz = level_size(level)
    data_mean = mean(channelview(level.data))
    println("o=$(level.octave), s=$(level.subdivision): size=$(sz.width)x$(sz.height), σ=$(round(level.sigma, digits=4)), mean=$(round(data_mean, digits=5))")
end
