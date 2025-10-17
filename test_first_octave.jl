"""
Test if setting first_octave=-1 matches VLFeat better.
"""

using VisualGeometryCore
using FileIO
using ImageCore: channelview
using Colors
using Statistics

# Load input image
img = load("vlfeat_comparison/input.tif")

# Create scale space with first_octave=-1 (like VLFeat)
println("Creating Julia scale space with first_octave=-1...")
ss = ScaleSpace(img; first_octave=-1)

println("\nJulia levels (first_octave=-1):")
println("="^60)
for level in ss
    if level.octave <= 2  # Only show first few octaves
        sz = level_size(level)
        data_mean = mean(channelview(level.data))
        println("o=$(level.octave), s=$(level.subdivision): size=$(sz.width)x$(sz.height), σ=$(round(level.sigma, digits=4)), mean=$(round(data_mean, digits=6))")
    end
end

# Compare octave 0, subdivision 0 with VLFeat
println("\n" * "="^60)
println("Comparing octave=0, subdivision=0:")
println("="^60)

julia_level = ss[0, 0]
julia_data = Float32.(channelview(julia_level.data))

vlfeat_img = load("vlfeat_comparison/vlfeat_gaussian/gaussian_o0_s0.tif")
vlfeat_data = Float32.(channelview(vlfeat_img))

diff = julia_data .- vlfeat_data
rms = sqrt(mean(diff.^2))
max_err = maximum(abs.(diff))

println("Julia mean:  $(mean(julia_data))")
println("VLFeat mean: $(mean(vlfeat_data))")
println("RMS error:   $(rms)")
println("Max error:   $(max_err)")

if rms < 1e-6
    println("\n✓✓✓ SUCCESS: RMS error < 1e-6!")
elseif rms < 1e-5
    println("\n✓ CLOSE: RMS error < 1e-5")
else
    println("\n✗ Still large error")
end
