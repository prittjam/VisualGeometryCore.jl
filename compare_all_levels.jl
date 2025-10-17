"""
Compare all overlapping levels between Julia and VLFeat.
"""

using VisualGeometryCore
using FileIO
using ImageCore: channelview
using Colors
using Statistics

# Create Julia scale space with first_octave=-1
img = load("vlfeat_comparison/input.tif")
ss = ScaleSpace(img; first_octave=-1)

println("Comparing overlapping levels:")
println("="^70)

# VLFeat has octaves -1 to 2, subdivisions 0 to 2 (ignoring -1 and 3 for now)
# Julia has octaves -1 to 2, subdivisions 0 to 2

for o in -1:2
    for s in 0:2
        # Load VLFeat level
        vlfeat_file = "vlfeat_comparison/vlfeat_gaussian/gaussian_o$(o)_s$(s).tif"
        if !isfile(vlfeat_file)
            continue
        end

        vlfeat_img = load(vlfeat_file)
        vlfeat_data = Float32.(channelview(vlfeat_img))

        # Get Julia level
        julia_level = ss[o, s]
        julia_data = Float32.(channelview(julia_level.data))

        # Compare
        diff = julia_data .- vlfeat_data
        rms = sqrt(mean(diff.^2))
        max_err = maximum(abs.(diff))

        status = rms < 1e-6 ? "✓✓✓" : (rms < 1e-5 ? "✓✓" : (rms < 1e-4 ? "✓" : (rms < 0.01 ? "~" : "✗")))
        println("$status o=$o, s=$s: RMS=$(round(rms, sigdigits=4)), Max=$(round(max_err, sigdigits=4))")
    end
end
