"""
Test extrema detection in scale space.
"""

using VisualGeometryCore
using FileIO
using Printf

println("="^70)
println("Test Extrema Detection")
println("="^70)
println()

# Load test image
println("Loading test image...")
img = load("vlfeat_comparison/input.tif")
println("✓ Image loaded: $(size(img))")
println()

# Create scale space with Hessian detector settings
println("Creating scale space...")
ss = ScaleSpace(img; first_octave=-1, octave_resolution=3,
               first_subdivision=-1, last_subdivision=3)

octave_range, subdivision_range = axes(ss)
println("✓ ScaleSpace created:")
println("  Octaves: $(first(octave_range)) to $(last(octave_range))")
println("  Subdivisions: $(first(subdivision_range)) to $(last(subdivision_range))")
println("  Total levels: $(length(ss))")
println()

# Compute Hessian determinant responses
println("Computing Hessian determinant responses...")
println()

# We need to create a transform function that wraps vlfeat_hessian_det
# The transform receives (dst, src) where both are 2D Gray arrays (views)
function hessian_det_transform!(dst::AbstractMatrix{Gray{Float32}},
                                src::AbstractMatrix{Gray{Float32}},
                                sigma::Float64, step::Float64)
    # Extract Float32 data
    det_data = vlfeat_hessian_det(src, sigma, step)
    # Copy to destination
    dst .= Gray{Float32}.(det_data)
    return dst
end

# Create response structure
# Since transform needs sigma and step, we'll compute per-level manually
# Actually, let me just iterate through levels and compute directly

println("Creating response storage...")
# Create template response structure
hessian_resp = ScaleSpaceResponse(ss, DERIVATIVE_KERNELS.xx)  # Just for structure

println("Computing Hessian determinants for each level...")
for level in ss
    step = 2.0^level.octave
    det_data = vlfeat_hessian_det(level.data, level.sigma, step)

    # Find corresponding response level and fill it
    resp_level = hessian_resp[level.octave, level.subdivision]
    resp_level.data .= Gray{Float32}.(det_data)

    @printf("  o=%2d, s=%2d: σ=%.3f, step=%.1f\n",
            level.octave, level.subdivision, level.sigma, step)
end

println()
println("✓ All Hessian determinants computed")
println()

# Detect extrema
println("Detecting extrema...")
println("  Peak threshold: 0.001")
println("  Edge threshold: 10.0")
println()

extrema = detect_extrema(hessian_resp, peak_threshold=0.001, edge_threshold=10.0)

println("✓ Detected $(length(extrema)) extrema")
println()

# Show statistics
if !isempty(extrema)
    println("="^70)
    println("Extrema Statistics")
    println("="^70)

    peak_scores = [abs(e.peakScore) for e in extrema]
    edge_scores = [e.edgeScore for e in extrema]

    @printf("Peak scores:  min=%.6e, max=%.6e, mean=%.6e\n",
            minimum(peak_scores), maximum(peak_scores), sum(peak_scores)/length(peak_scores))
    @printf("Edge scores:  min=%.6e, max=%.6e, mean=%.6e\n",
            minimum(edge_scores), maximum(edge_scores), sum(edge_scores)/length(edge_scores))
    println()

    # Count by octave
    println("Extrema per octave:")
    for o in octave_range
        count = sum(e.octave == o for e in extrema)
        if count > 0
            @printf("  Octave %2d: %4d extrema\n", o, count)
        end
    end

    println()
    println("First 10 extrema:")
    for (i, e) in enumerate(extrema[1:min(10, length(extrema))])
        @printf("  %2d: oct=%2d, (x,y,z)=(%6.2f, %6.2f, %5.2f), σ=%.3f, peak=%9.6f, edge=%6.3f\n",
                i, e.octave, e.x, e.y, e.z, e.sigma, e.peakScore, e.edgeScore)
    end
end

println()
println("="^70)
println("Test complete!")
println("="^70)
