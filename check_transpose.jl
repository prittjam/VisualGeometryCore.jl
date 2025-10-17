"""
Check if there's a transpose or flip issue between Julia and VLFeat outputs.
"""

using FileIO
using ImageCore: channelview
using Colors

# Load first level from both implementations
julia_img = load("vlfeat_comparison/julia_gaussian/gaussian_o0_s0.tif")
vlfeat_img = load("vlfeat_comparison/vlfeat_gaussian/gaussian_o0_s0.tif")

julia_data = Float32.(channelview(julia_img))
vlfeat_data = Float32.(channelview(vlfeat_img))

println("Image sizes:")
println("  Julia:  $(size(julia_data))")
println("  VLFeat: $(size(vlfeat_data))")
println()

# Check corners
println("Corner values:")
println("  Top-left:")
println("    Julia:  $(julia_data[1,1])")
println("    VLFeat: $(vlfeat_data[1,1])")
println("  Top-right:")
println("    Julia:  $(julia_data[1,end])")
println("    VLFeat: $(vlfeat_data[1,end])")
println("  Bottom-left:")
println("    Julia:  $(julia_data[end,1])")
println("    VLFeat: $(vlfeat_data[end,1])")
println("  Bottom-right:")
println("    Julia:  $(julia_data[end,end])")
println("    VLFeat: $(vlfeat_data[end,end])")
println()

# Check if transpose matches
transposed_vlfeat = vlfeat_data'
println("Testing transpose:")
println("  RMS with original:   $(sqrt(sum((julia_data .- vlfeat_data).^2) / length(julia_data)))")
println("  RMS with transpose:  $(sqrt(sum((julia_data .- transposed_vlfeat).^2) / length(julia_data)))")
println()

# Check if flips match
flipped_ud = reverse(vlfeat_data, dims=1)
flipped_lr = reverse(vlfeat_data, dims=2)
flipped_both = reverse(reverse(vlfeat_data, dims=1), dims=2)

println("Testing flips:")
println("  RMS with up-down flip:    $(sqrt(sum((julia_data .- flipped_ud).^2) / length(julia_data)))")
println("  RMS with left-right flip: $(sqrt(sum((julia_data .- flipped_lr).^2) / length(julia_data)))")
println("  RMS with both flips:      $(sqrt(sum((julia_data .- flipped_both).^2) / length(julia_data)))")
