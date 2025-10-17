"""
Diagnostic script to investigate differences between Julia and VLFeat implementations.
"""

using VisualGeometryCore
using Colors, FixedPointNumbers
using FileIO
using Statistics
using ImageCore: channelview

# Load the input image
img = load("vlfeat_comparison/input.tif")
println("Input image stats:")
println("  Size: $(size(img))")
println("  Type: $(eltype(img))")
println("  Min: $(minimum(img)), Max: $(maximum(img))")
println("  Mean: $(mean(img))")
println()

# Create Julia scale space
println("Creating Julia scale space...")
ss = ScaleSpace(img)

# Compare first level (octave 0, subdivision 0) in detail
o, s = 0, 0
julia_file = "vlfeat_comparison/julia_gaussian/gaussian_o$(o)_s$(s).tif"
vlfeat_file = "vlfeat_comparison/vlfeat_gaussian/gaussian_o$(o)_s$(s).tif"

julia_img = load(julia_file)
vlfeat_img = load(vlfeat_file)

julia_data = Float32.(channelview(julia_img))
vlfeat_data = Float32.(channelview(vlfeat_img))

println("Comparing octave=$o, subdivision=$s:")
println()

println("Julia stats:")
println("  Min: $(minimum(julia_data))")
println("  Max: $(maximum(julia_data))")
println("  Mean: $(mean(julia_data))")
println("  Std: $(std(julia_data))")

println()
println("VLFeat stats:")
println("  Min: $(minimum(vlfeat_data))")
println("  Max: $(maximum(vlfeat_data))")
println("  Mean: $(mean(vlfeat_data))")
println("  Std: $(std(vlfeat_data))")

println()
diff = julia_data .- vlfeat_data
println("Difference stats:")
println("  RMS: $(sqrt(mean(diff.^2)))")
println("  Max abs: $(maximum(abs.(diff)))")
println("  Mean: $(mean(diff))")
println("  Std: $(std(diff))")

println()
println("Sample pixel comparisons (first 5x5):")
println("Julia:")
display(julia_data[1:5, 1:5])
println("\nVLFeat:")
display(vlfeat_data[1:5, 1:5])
println("\nDifference:")
display(diff[1:5, 1:5])

# Check kernel parameters
println("\n" * "="^60)
println("Kernel Analysis")
println("="^60)
level = ss[0, 0]
sigma = level.sigma
println("Sigma for o=$o, s=$s: $sigma")

# Generate VLFeat kernel
width = ceil(Int, sigma * 3.0)
kernel_size = 2 * width + 1
println("VLFeat kernel size: $(kernel_size)x$(kernel_size)")
println("VLFeat kernel width: $width")

using ImageFiltering: Kernel
kern = Kernel.gaussian((sigma, sigma), (kernel_size, kernel_size))
println("Kernel sum: $(sum(kern))")
println("Kernel center value: $(kern[0, 0])")
