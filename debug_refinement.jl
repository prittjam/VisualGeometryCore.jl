#!/usr/bin/env julia
"""
Debug refinement with detailed output
"""

using VisualGeometryCore
using FileIO
using Printf
using LinearAlgebra
using StaticArrays
using ImageCore: channelview

# Manually inline the refinement with debug output
function debug_refine_extremum_3d(octave_3d, x::Int, y::Int, z::Int,
                                  octave_num::Int, first_subdivision::Int, octave_resolution::Int,
                                  base_scale::Float64, step::Float64)
    height, width, depth = size(octave_3d)

    @printf("Starting refinement:\n")
    @printf("  Initial: x=%d, y=%d, z=%d\n", x, y, z)
    @printf("  Octave bounds: [1, %d] × [1, %d] × [1, %d]\n", width, height, depth)

    dx, dy, dz = 0, 0, 0
    max_iterations = 5

    for iter in 1:max_iterations
        @printf("\nIteration %d:\n", iter)

        # Update position
        x += dx
        y += dy
        z += dz

        @printf("  Position: x=%d, y=%d, z=%d\n", x, y, z)

        # Bounds check
        if x < 2 || x > width-1 || y < 2 || y > height-1 || z < 2 || z > depth-1
            println("  → OUT OF BOUNDS! Needs [2, width-1] × [2, height-1] × [2, depth-1]")
            return (nothing, false)
        end

        # Helper to access octave data
        at(dx, dy, dz) = Float64(octave_3d[y+dy, x+dx, z+dz].val)

        # Compute gradient
        Dx = 0.5 * (at(1, 0, 0) - at(-1, 0, 0))
        Dy = 0.5 * (at(0, 1, 0) - at(0, -1, 0))
        Dz = 0.5 * (at(0, 0, 1) - at(0, 0, -1))

        # Compute Hessian
        Dxx = at(1, 0, 0) + at(-1, 0, 0) - 2.0 * at(0, 0, 0)
        Dyy = at(0, 1, 0) + at(0, -1, 0) - 2.0 * at(0, 0, 0)
        Dzz = at(0, 0, 1) + at(0, 0, -1) - 2.0 * at(0, 0, 0)

        Dxy = 0.25 * (at(1, 1, 0) + at(-1, -1, 0) - at(-1, 1, 0) - at(1, -1, 0))
        Dxz = 0.25 * (at(1, 0, 1) + at(-1, 0, -1) - at(-1, 0, 1) - at(1, 0, -1))
        Dyz = 0.25 * (at(0, 1, 1) + at(0, -1, -1) - at(0, -1, 1) - at(0, 1, -1))

        H = @SMatrix [Dxx Dxy Dxz; Dxy Dyy Dyz; Dxz Dyz Dzz]
        g = @SVector [Dx, Dy, Dz]

        det_H = det(H)
        @printf("  det(H) = %.6e\n", det_H)

        if abs(det_H) < 1e-10
            println("  → SINGULAR HESSIAN!")
            return (nothing, false)
        end

        b = H \ (-g)
        @printf("  Offset b = [%.3f, %.3f, %.3f]\n", b[1], b[2], b[3])

        # Check displacement - VLFeat only moves in x,y!
        dx = (b[1] > 0.6 && x < width-1 ? 1 : 0) + (b[1] < -0.6 && x > 2 ? -1 : 0)
        dy = (b[2] > 0.6 && y < height-1 ? 1 : 0) + (b[2] < -0.6 && y > 2 ? -1 : 0)
        dz = 0  # VLFeat does NOT move in z

        @printf("  Move: dx=%d, dy=%d, dz=%d (z fixed per VLFeat)\n", dx, dy, dz)

        if dx == 0 && dy == 0
            println("  → CONVERGED!")

            # Check offset validity
            if abs(b[1]) >= 1.5 || abs(b[2]) >= 1.5 || abs(b[3]) >= 1.5
                @printf("  → OFFSET TOO LARGE: |b| = [%.3f, %.3f, %.3f]\n", abs(b[1]), abs(b[2]), abs(b[3]))
                return (nothing, false)
            end

            # Compute peak score
            peakScore = at(0, 0, 0) + 0.5 * (Dx * b[1] + Dy * b[2] + Dz * b[3])
            @printf("  Peak score: %.6e\n", peakScore)

            # Compute edge score
            trace_H = Dxx + Dyy
            det_H_2d = Dxx * Dyy - Dxy * Dxy

            edgeScore = if det_H_2d < 0
                Inf
            else
                alpha = (trace_H * trace_H) / det_H_2d
                0.5 * alpha - 1.0 + sqrt(max(0.25 * alpha - 1.0, 0.0) * alpha)
            end
            @printf("  Edge score: %.3f\n", edgeScore)

            # Compute refined position
            refined_x = x + b[1]
            refined_y = y + b[2]
            refined_z = z + b[3]

            @printf("  Refined: (%.3f, %.3f, %.3f)\n", refined_x, refined_y, refined_z)

            # Final bounds check
            if refined_x < 1 || refined_x > width || refined_y < 1 || refined_y > height ||
               refined_z < 1 || refined_z > depth
                println("  → REFINED POSITION OUT OF BOUNDS!")
                return (nothing, false)
            end

            subdivision = first_subdivision + (z - 1)
            sigma = base_scale * 2.0^(octave_num + (refined_z + first_subdivision - 1) / octave_resolution)

            extremum = Extremum3D(octave_num, x, y, z,
                                 refined_x, refined_y, refined_z,
                                 sigma, step, peakScore, edgeScore)
            return (extremum, true)
        end
    end

    println("  → DID NOT CONVERGE AFTER 5 ITERATIONS!")
    return (nothing, false)
end

# Test
img = load("vlfeat_comparison/input.tif")
ss = ScaleSpace(img; first_octave=-1, octave_resolution=3, first_subdivision=-1, last_subdivision=3)

hessian_resp = ScaleSpaceResponse(ss, DERIVATIVE_KERNELS.xx)
for level in ss
    step = 2.0^level.octave
    det_data = vlfeat_hessian_det(level.data, level.sigma, step)
    resp_level = hessian_resp[level.octave, level.subdivision]
    resp_level.data .= Gray{Float32}.(det_data)
end

octave_idx = findfirst(oc -> oc.octave == -1, hessian_resp.octaves)
octave_data = hessian_resp.octaves[octave_idx]
octave_3d = octave_data.G

println("="^80)
println("VLFeat #14: (105, 111, z=2)")
println("="^80)
result, converged = debug_refine_extremum_3d(octave_3d, 105, 111, 2,
                                              -1, -1, 3, 2.015874, 0.5)
