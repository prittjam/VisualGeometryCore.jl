"""
Check that Julia and VLFeat are reading the same input image.
"""

using FileIO
using ImageCore: channelview
using Colors
using Statistics

# Load input image
input_img = load("vlfeat_comparison/input.tif")
input_data = Float32.(channelview(input_img))

println("Input image:")
println("  Size: $(size(input_data))")
println("  Mean: $(mean(input_data))")
println("  Min:  $(minimum(input_data))")
println("  Max:  $(maximum(input_data))")
println()

# Show corners
println("Input corners:")
println("  Top-left:     $(input_data[1,1])")
println("  Top-right:    $(input_data[1,end])")
println("  Bottom-left:  $(input_data[end,1])")
println("  Bottom-right: $(input_data[end,end])")
println()

# Show first 5x5
println("Input first 5x5:")
display(input_data[1:5, 1:5])
println()

# Now check what sigma values are being used
using VisualGeometryCore

img = load("vlfeat_comparison/input.tif")
ss = ScaleSpace(img)

println("\nJulia ScaleSpace parameters:")
println("  Base sigma: $(ss.base_sigma)")
println("  Camera PSF: $(ss.camera_psf)")
println("  Octave resolution: $(length(unique(ss.levels.subdivision)))")
println("  First octave: $(minimum(ss.levels.octave))")
println()

# Show sigma values for first few levels
println("First few levels:")
for (i, level) in enumerate(ss)
    if i <= 5
        println("  Level $i: octave=$(level.octave), subdivision=$(level.subdivision), sigma=$(round(level.sigma, digits=4))")
    end
end
