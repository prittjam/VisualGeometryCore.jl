"""
Scale Space Utilities Demo

This example demonstrates the utility functions for working with preallocated scale spaces.
"""

using VisualGeometryCore
using VisualGeometryCore: ScaleSpace, default_geometry, summary
using VisualGeometryCore: iterate_levels, scale_coordinates, octave_to_input_coordinates

# Create a scale space for a typical image
img_width, img_height = 800, 600
geom = default_geometry(img_width, img_height; octave_resolution=3, base_sigma=1.6)
ss = ScaleSpace(geom, img_width, img_height, Float32)

println("=== Scale Space Utilities Demo ===\n")

# Show summary
summary(ss)

println("\n=== Iterating Over Levels ===")
level_count = 0
for (octave, scale, level) in iterate_levels(ss)
    global level_count
    sigma = get_sigma(ss, octave, scale)
    level_count += 1
    if level_count <= 5  # Show first 5 levels
        println("Level $level_count: (o=$octave, s=$scale), σ=$(round(sigma, digits=3)), size=$(size(level))")
    end
end
println("... (showing first 5 of $level_count total levels)")

println("\n=== Coordinate Conversion Examples ===")

# Example: blob detected at octave 2, scale 1
octave, scale = 2, 1
blob_x_octave, blob_y_octave = 50.0, 40.0  # Coordinates in octave 2

# Convert to input image coordinates
blob_x_input, blob_y_input = octave_to_input_coordinates(ss, blob_x_octave, blob_y_octave, octave)

println("Blob detected in octave $octave:")
println("  Octave coordinates: ($blob_x_octave, $blob_y_octave)")
println("  Input coordinates: ($(round(blob_x_input, digits=1)), $(round(blob_y_input, digits=1)))")

# Show scale factor
octave_geom = get_octave_geometry(ss, octave)
println("  Scale factor: $(octave_geom.step)x")

# Convert between different octaves
blob_x_oct0, blob_y_oct0 = scale_coordinates(blob_x_octave, blob_y_octave, octave, 0)
blob_x_oct1, blob_y_oct1 = scale_coordinates(blob_x_octave, blob_y_octave, octave, 1)

println("\nSame blob in different octaves:")
println("  Octave 0: ($(round(blob_x_oct0, digits=1)), $(round(blob_y_oct0, digits=1)))")
println("  Octave 1: ($(round(blob_x_oct1, digits=1)), $(round(blob_y_oct1, digits=1)))")
println("  Octave 2: ($blob_x_octave, $blob_y_octave)")

println("\n=== Scale Space Properties ===")

# Show sigma progression within an octave
println("Sigma progression in octave 0:")
for s in valid_scale_range(ss)
    sigma = get_sigma(ss, 0, s)
    println("  Scale $s: σ = $(round(sigma, digits=3))")
end

# Show sigma progression across octaves at same scale
println("\nSigma progression at scale 0 across octaves:")
for o in valid_octave_range(ss)
    if o <= 3  # Show first few octaves
        sigma = get_sigma(ss, o, 0)
        octave_geom = get_octave_geometry(ss, o)
        println("  Octave $o: σ = $(round(sigma, digits=3)), size = $(octave_geom.width)×$(octave_geom.height)")
    end
end

println("\n=== Memory and Performance ===")
total_pixels, memory_mb = memory_usage(ss)
println("Total preallocated memory: $(round(memory_mb, digits=2)) MB")
println("Average pixels per level: $(round(total_pixels / length(ss.levels), digits=0))")

# Show memory distribution across octaves
println("\nMemory per octave:")
for octave in valid_octave_range(ss)
    octave_geom = get_octave_geometry(ss, octave)
    pixels_per_level = octave_geom.width * octave_geom.height
    levels_per_octave = length(valid_scale_range(ss))
    octave_pixels = pixels_per_level * levels_per_octave
    octave_mb = octave_pixels * sizeof(Float32) / (1024^2)
    println("  Octave $octave: $(round(octave_mb, digits=2)) MB ($(octave_geom.width)×$(octave_geom.height) × $levels_per_octave levels)")
end

println("\n=== Ready for Scale Space Processing ===")
println("This preallocated structure is now ready for:")
println("  • Gaussian filtering at each computed σ")
println("  • Difference of Gaussians (DoG) computation")
println("  • Multi-scale blob detection")
println("  • Efficient coordinate transformations between scales")