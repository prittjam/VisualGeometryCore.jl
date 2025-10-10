"""
Scale Space Construction Demo

This example demonstrates how to create and use a preallocated Gaussian scale space
similar to VLFeat's approach.
"""

using VisualGeometryCore
using VisualGeometryCore: ScaleSpace, ScaleSpaceGeometry, default_geometry
using VisualGeometryCore: get_level, get_octave_geometry, get_sigma

# Example image dimensions
img_width, img_height = 640, 480

println("=== Scale Space Construction Demo ===")
println("Input image size: $(img_width) × $(img_height)")

# Create default geometry
geom = default_geometry(img_width, img_height; octave_resolution=3, base_sigma=1.6)

println("\nScale Space Geometry:")
println("  Octaves: $(geom.first_octave) to $(geom.last_octave)")
println("  Octave resolution: $(geom.octave_resolution)")
println("  Scale subdivisions: $(geom.octave_first_subdivision) to $(geom.octave_last_subdivision)")
println("  Base sigma: $(geom.base_sigma)")

# Create preallocated scale space
ss = ScaleSpace(geom, img_width, img_height, Float32)

println("\nPreallocated Scale Space:")
println("  Total octaves: $(length(ss.octave_geometries))")
println("  Total levels: $(length(ss.levels))")

# Show octave geometries
println("\nOctave Geometries:")
for (i, octave) in enumerate(geom.first_octave:geom.last_octave)
    octave_geom = get_octave_geometry(ss, octave)
    println("  Octave $octave: $(octave_geom.width)×$(octave_geom.height), step=$(octave_geom.step)")
end

# Show some scale levels and their sigmas
println("\nSample Scale Levels:")
for octave in geom.first_octave:min(geom.first_octave+2, geom.last_octave)
    for scale in geom.octave_first_subdivision:geom.octave_last_subdivision
        sigma = get_sigma(ss, octave, scale)
        level = get_level(ss, octave, scale)
        println("  (o=$octave, s=$scale): σ=$(round(sigma, digits=3)), size=$(size(level))")
    end
    println()
end

# Demonstrate accessing specific levels
println("=== Accessing Scale Levels ===")

# Get level at octave 0, scale 0 (original resolution, base scale)
level_0_0 = get_level(ss, 0, 0)
sigma_0_0 = get_sigma(ss, 0, 0)
println("Level (0,0): σ=$(sigma_0_0), size=$(size(level_0_0))")

# Get level at octave 1, scale 1 (half resolution, higher scale)
if geom.last_octave >= 1
    level_1_1 = get_level(ss, 1, 1)
    sigma_1_1 = get_sigma(ss, 1, 1)
    println("Level (1,1): σ=$(round(sigma_1_1, digits=3)), size=$(size(level_1_1))")
end

# Show memory usage
total_pixels = sum(prod(size(level)) for level in values(ss.levels))
memory_mb = total_pixels * sizeof(Float32) / (1024^2)
println("\nMemory Usage:")
println("  Total pixels: $(total_pixels)")
println("  Memory: $(round(memory_mb, digits=2)) MB")

println("\n=== Scale Space Ready for Gaussian Filtering ===")
println("All image levels are preallocated and ready for:")
println("  1. Gaussian filtering at computed σ values")
println("  2. Difference of Gaussians (DoG) computation")
println("  3. Blob detection across scales")