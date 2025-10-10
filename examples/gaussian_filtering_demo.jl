"""
Gaussian Filtering Demo

This example demonstrates populating the scale space with actual Gaussian-filtered images.
"""

using VisualGeometryCore
using VisualGeometryCore: ScaleSpace, default_geometry, populate_scale_space!
using VisualGeometryCore: get_level, get_sigma, summary
using TestImages
using Statistics

println("=== Gaussian Filtering Demo ===")

# Load a test image
println("Loading test image...")
img = testimage("cameraman")
img_gray = Float32.(img)  # Convert to grayscale Float32

img_height, img_width = size(img_gray)
println("Image size: $(img_width) × $(img_height)")

# Create scale space geometry
geom = default_geometry(img_width, img_height; octave_resolution=3, base_sigma=1.6)
println("\nCreating scale space...")
ss = ScaleSpace(geom, img_width, img_height, Float32)

# Show summary before population
summary(ss)

# Populate the scale space with Gaussian-filtered images
println("\n=== Populating Scale Space with Gaussian Filtering ===")
println("Applying Gaussian filters at computed σ values...")

populate_scale_space!(ss, img_gray)

println("✓ Scale space populated with Gaussian-filtered images!")

# Analyze the filtered results
println("\n=== Analysis of Filtered Images ===")

# Show statistics for a few levels
sample_levels = [
    (0, -1, "Base octave, finest scale"),
    (0, 0, "Base octave, base scale"), 
    (0, 2, "Base octave, coarse scale"),
    (1, 0, "Half resolution, base scale"),
    (2, 0, "Quarter resolution, base scale")
]

for (octave, scale, description) in sample_levels
    if octave <= geom.last_octave
        level = get_level(ss, octave, scale)
        sigma = get_sigma(ss, octave, scale)
        
        # Compute statistics
        level_mean = mean(level)
        level_std = std(level)
        level_min, level_max = extrema(level)
        
        println("$description:")
        println("  (o=$octave, s=$scale): σ=$(round(sigma, digits=3))")
        println("  Size: $(size(level))")
        println("  Stats: μ=$(round(level_mean, digits=3)), σ=$(round(level_std, digits=3))")
        println("  Range: [$(round(level_min, digits=3)), $(round(level_max, digits=3))]")
        println()
    end
end

# Compare smoothing effects
println("=== Smoothing Effect Analysis ===")
base_level = get_level(ss, 0, 0)  # Base scale
smooth_level = get_level(ss, 0, 2)  # Smoother scale

# Compute local variance as a measure of detail
function local_variance(img::Matrix, window_size::Int=3)
    h, w = size(img)
    variance_map = similar(img)
    half_window = window_size ÷ 2
    
    for j in 1:w
        for i in 1:h
            # Define window bounds
            i_start = max(1, i - half_window)
            i_end = min(h, i + half_window)
            j_start = max(1, j - half_window)
            j_end = min(w, j + half_window)
            
            # Compute local variance
            window = img[i_start:i_end, j_start:j_end]
            variance_map[i, j] = var(window)
        end
    end
    
    return variance_map
end

base_variance = local_variance(base_level)
smooth_variance = local_variance(smooth_level)

base_detail = mean(base_variance)
smooth_detail = mean(smooth_variance)
detail_reduction = (base_detail - smooth_detail) / base_detail * 100

println("Local detail analysis (3×3 windows):")
println("  Base scale (σ=$(round(get_sigma(ss, 0, 0), digits=3))): avg variance = $(round(base_detail, digits=6))")
println("  Smooth scale (σ=$(round(get_sigma(ss, 0, 2), digits=3))): avg variance = $(round(smooth_detail, digits=6))")
println("  Detail reduction: $(round(detail_reduction, digits=1))%")

# Show scale space is ready for further processing
println("\n=== Scale Space Ready for Processing ===")
println("The scale space now contains actual Gaussian-filtered images at:")

level_count = 0
for octave in geom.first_octave:min(geom.first_octave+2, geom.last_octave)
    println("  Octave $octave:")
    for scale in geom.octave_first_subdivision:geom.octave_last_subdivision
        global level_count
        level_count += 1
        sigma = get_sigma(ss, octave, scale)
        level = get_level(ss, octave, scale)
        println("    Scale $scale: σ=$(round(sigma, digits=3)), size=$(size(level))")
    end
end

println("\nReady for:")
println("  • Difference of Gaussians (DoG) computation")
println("  • Blob detection across scales")
println("  • Feature extraction at multiple resolutions")
println("  • Scale-invariant analysis")