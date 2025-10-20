#!/usr/bin/env julia
"""
Test refinement on the specific discrete extrema that VLFeat refined but Julia didn't.
"""

using VisualGeometryCore
using FileIO
using Printf

# Load and setup
img = load("vlfeat_comparison/input.tif")
ss = ScaleSpace(img; first_octave=-1, octave_resolution=3, first_subdivision=-1, last_subdivision=3)

hessian_resp = ScaleSpaceResponse(ss, DERIVATIVE_KERNELS.xx)
for level in ss
    step = 2.0^level.octave
    det_data = vlfeat_hessian_det(level.data, level.sigma, step)
    resp_level = hessian_resp[level.octave, level.subdivision]
    resp_level.data .= Gray{Float32}.(det_data)
end

# Test cases: (octave, x, y, z) from investigation
test_cases = [
    (octave=-1, x=105, y=111, z=2, vf_id=14),
    (octave=-1, x=66, y=69, z=4, vf_id=33)
]

for tc in test_cases
    println("="^80)
    @printf("Testing refinement for VLFeat #%d\n", tc.vf_id)
    @printf("Discrete extremum: octave=%d, (x=%d, y=%d, z=%d)\n", tc.octave, tc.x, tc.y, tc.z)
    println("="^80)
    println()

    # Get the octave data
    octave_idx = findfirst(oc -> oc.octave == tc.octave, hessian_resp.octaves)
    if octave_idx === nothing
        println("ERROR: Octave not found!")
        continue
    end

    octave_data = hessian_resp.octaves[octave_idx]
    octave_3d = octave_data.G
    step = octave_data.step
    first_subdivision = first(octave_data.subdivisions)

    println("Octave info:")
    @printf("  Size: %s\n", size(octave_3d))
    @printf("  Step: %.2f\n", step)
    @printf("  First subdivision: %d\n", first_subdivision)
    println()

    # Try refinement
    println("Calling refine_extremum_3d...")
    result, converged = refine_extremum_3d(octave_3d, tc.x, tc.y, tc.z,
                                          tc.octave, first_subdivision, 3, 2.015874, step)

    if converged
        println("✓ CONVERGED!")
        @printf("  Refined: (%.3f, %.3f, %.3f)\n", result.x, result.y, result.z)
        @printf("  Sigma: %.3f\n", result.sigma)
        @printf("  Peak score: %.6e\n", result.peakScore)
        @printf("  Edge score: %.3f\n", result.edgeScore)
        println()

        # Check thresholds
        @printf("Threshold checks:\n")
        @printf("  |Peak| > 0.003? %s (%.6e)\n",
                abs(result.peakScore) > 0.003 ? "✓" : "✗", abs(result.peakScore))
        @printf("  Edge < 10.0?    %s (%.3f)\n",
                result.edgeScore < 10.0 ? "✓" : "✗", result.edgeScore)

        if abs(result.peakScore) < 0.003
            println("  → FILTERED: Peak score too low")
        elseif result.edgeScore >= 10.0
            println("  → FILTERED: Edge score too high")
        else
            println("  → PASSED ALL FILTERS!")

            # Convert to VLFeat coords
            vf_x = (result.x - 1) * step
            vf_y = (result.y - 1) * step
            @printf("\nVLFeat coords: (%.3f, %.3f)\n", vf_x, vf_y)
        end
    else
        println("✗ DID NOT CONVERGE")
        if result === nothing
            println("  Reason: Refinement returned nothing")
            println("  Possible causes:")
            println("    - Went out of bounds")
            println("    - Offset > 1.5")
            println("    - Singular Hessian")
        end
    end

    println()
end

println("="^80)
