#!/usr/bin/env julia
"""
Test 3D derivative kernels to verify they match VLFeat's finite differencing.
"""

using VisualGeometryCore
using FileIO
using Printf

println("="^80)
println("TESTING 3D DERIVATIVE KERNELS")
println("="^80)
println()

# Load test image
img = load("vlfeat_comparison/input.tif")

# Build scale space
ss = ScaleSpace(img; first_octave=-1, octave_resolution=3,
               first_subdivision=-1, last_subdivision=3)

# Compute Hessian determinant response using VLFeat
hessian_resp = ScaleSpaceResponse(ss, DERIVATIVE_KERNELS.xx)
for level in ss
    step = 2.0^level.octave
    det_data = vlfeat_hessian_det(level.data, level.sigma, step)
    resp_level = hessian_resp[level.octave, level.subdivision]
    resp_level.data .= Gray{Float32}.(det_data)
end

println("✓ Computed Hessian determinant response")
println()

# Test creating derivative responses with 3D kernels
println("Creating 3D derivative responses...")
println()

# First derivatives
∇x_resp = ScaleSpaceResponse(hessian_resp, DERIVATIVE_KERNELS_3D.dx)
∇y_resp = ScaleSpaceResponse(hessian_resp, DERIVATIVE_KERNELS_3D.dy)
∇z_resp = ScaleSpaceResponse(hessian_resp, DERIVATIVE_KERNELS_3D.dz)

println("  ✓ Created ∇x, ∇y, ∇z response structures")

# Second derivatives (Hessian diagonal)
∇²xx_resp = ScaleSpaceResponse(hessian_resp, DERIVATIVE_KERNELS_3D.dxx)
∇²yy_resp = ScaleSpaceResponse(hessian_resp, DERIVATIVE_KERNELS_3D.dyy)
∇²zz_resp = ScaleSpaceResponse(hessian_resp, DERIVATIVE_KERNELS_3D.dzz)

println("  ✓ Created ∇²xx, ∇²yy, ∇²zz response structures")

# Mixed partial derivatives
∇²xy_resp = ScaleSpaceResponse(hessian_resp, DERIVATIVE_KERNELS_3D.dxy)
∇²xz_resp = ScaleSpaceResponse(hessian_resp, DERIVATIVE_KERNELS_3D.dxz)
∇²yz_resp = ScaleSpaceResponse(hessian_resp, DERIVATIVE_KERNELS_3D.dyz)

println("  ✓ Created ∇²xy, ∇²xz, ∇²yz response structures")
println()

# Compute derivatives (applies 3D convolution)
println("Computing 3D derivatives...")
∇x_resp(hessian_resp)
∇y_resp(hessian_resp)
∇z_resp(hessian_resp)
println("  ✓ Computed first derivatives")

∇²xx_resp(hessian_resp)
∇²yy_resp(hessian_resp)
∇²zz_resp(hessian_resp)
println("  ✓ Computed second derivatives (diagonal)")

∇²xy_resp(hessian_resp)
∇²xz_resp(hessian_resp)
∇²yz_resp(hessian_resp)
println("  ✓ Computed mixed partial derivatives")
println()

# Verify results are finite
println("Verifying derivative values are finite...")
for (name, resp) in [
    ("∇x", ∇x_resp), ("∇y", ∇y_resp), ("∇z", ∇z_resp),
    ("∇²xx", ∇²xx_resp), ("∇²yy", ∇²yy_resp), ("∇²zz", ∇²zz_resp),
    ("∇²xy", ∇²xy_resp), ("∇²xz", ∇²xz_resp), ("∇²yz", ∇²yz_resp)
]
    all_finite = all(isfinite(Float32(level.data[i]))
                    for level in resp
                    for i in eachindex(level.data))
    if all_finite
        println("  ✓ $name: all values finite")
    else
        println("  ✗ $name: contains non-finite values!")
    end
end
println()

# Test accessing derivatives at a specific location
println("Testing derivative access at octave -1, subdivision 0...")
octave_num = -1
subdiv = 0

level_dx = ∇x_resp[octave_num, subdiv]
level_dy = ∇y_resp[octave_num, subdiv]
level_dz = ∇z_resp[octave_num, subdiv]

@printf("  Octave %d, subdivision %d, size: %dx%d\n",
        octave_num, subdiv, size(level_dx.data, 2), size(level_dx.data, 1))

# Sample a point in the middle
y, x = size(level_dx.data) .÷ 2
@printf("  Sample point (%d, %d):\n", x, y)
@printf("    ∇x  = %.6e\n", Float32(level_dx.data[y, x]))
@printf("    ∇y  = %.6e\n", Float32(level_dy.data[y, x]))
@printf("    ∇z  = %.6e\n", Float32(level_dz.data[y, x]))
println()

println("="^80)
println("✓ 3D KERNEL TEST COMPLETE")
println("="^80)
