"""
Check how VLFeat upsamples compared to Julia's imresize.

VLFeat uses bilinear interpolation for upsampling.
"""

using FileIO
using ImageCore: channelview
using Colors
using ImageTransformations: imresize
using Statistics

# Load original input
input_img = load("vlfeat_comparison/input.tif")
input_data = Float32.(channelview(input_img))

println("Original input: $(size(input_data))")
println()

# Julia upsampling (what we currently do)
h, w = size(input_img)
new_size = (h * 2, w * 2)
julia_upsampled = imresize(input_img, new_size)
julia_up_data = Float32.(channelview(julia_upsampled))

println("Julia upsampled: $(size(julia_up_data))")
println("Mean: $(mean(julia_up_data))")
println()

# Load VLFeat octave -1, subdivision -1 (the first level after upsampling and initial blur)
vlfeat_img = load("vlfeat_comparison/vlfeat_gaussian/gaussian_o-1_s-1.tif")
vlfeat_data = Float32.(channelview(vlfeat_img))

println("VLFeat o=-1, s=-1: $(size(vlfeat_data))")
println("Mean: $(mean(vlfeat_data))")
println()

# Load Julia o=-1, s=-1
using VisualGeometryCore
ss = ScaleSpace(input_img; first_octave=-1, first_subdivision=-1, last_subdivision=3)
julia_level = ss[-1, -1]
julia_data = Float32.(channelview(julia_level.data))

println("Julia o=-1, s=-1 (after ScaleSpace): $(size(julia_data))")
println("Mean: $(mean(julia_data))")
println()

# Compare just the upsampled image (before any blur)
println("="^60)
println("Checking upsampling method:")
println("="^60)

# VLFeat's copy_and_upsample creates 2x2 blocks from each pixel using:
# destination[0] = v00
# destination[1] = 0.5 * (v00 + v10)  # horizontal neighbor
# destination[width*2] = 0.5 * (v00 + v01)  # vertical neighbor
# destination[width*2+1] = 0.25 * (v00 + v01 + v10 + v11)  # diagonal

println("Checking if imresize matches VLFeat bilinear upsampling...")
println("First few pixels of upsampled comparison:")
println()
println("Input [1:3, 1:3]:")
display(input_data[1:3, 1:3])
println()
println("Julia upsampled [1:6, 1:6]:")
display(julia_up_data[1:6, 1:6])
