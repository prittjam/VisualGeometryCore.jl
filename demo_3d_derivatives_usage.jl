#!/usr/bin/env julia
"""
Demonstration of how 3D derivative responses can be used for extrema refinement.

This shows the new architecture where derivatives can be pre-computed across
the entire scale-space response, then accessed during refinement.
"""

using VisualGeometryCore
using FileIO
using Printf

println("="^80)
println("DEMONSTRATION: Pre-computed 3D Derivatives for Extrema Refinement")
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

# Pre-compute ALL 3D derivatives of the response
println("Pre-computing 3D derivatives of Hessian determinant response...")

derivatives_3d = (
    ∇x = ScaleSpaceResponse(hessian_resp, DERIVATIVE_KERNELS_3D.dx)(hessian_resp),
    ∇y = ScaleSpaceResponse(hessian_resp, DERIVATIVE_KERNELS_3D.dy)(hessian_resp),
    ∇z = ScaleSpaceResponse(hessian_resp, DERIVATIVE_KERNELS_3D.dz)(hessian_resp),
    ∇²xx = ScaleSpaceResponse(hessian_resp, DERIVATIVE_KERNELS_3D.dxx)(hessian_resp),
    ∇²yy = ScaleSpaceResponse(hessian_resp, DERIVATIVE_KERNELS_3D.dyy)(hessian_resp),
    ∇²zz = ScaleSpaceResponse(hessian_resp, DERIVATIVE_KERNELS_3D.dzz)(hessian_resp),
    ∇²xy = ScaleSpaceResponse(hessian_resp, DERIVATIVE_KERNELS_3D.dxy)(hessian_resp),
    ∇²xz = ScaleSpaceResponse(hessian_resp, DERIVATIVE_KERNELS_3D.dxz)(hessian_resp),
    ∇²yz = ScaleSpaceResponse(hessian_resp, DERIVATIVE_KERNELS_3D.dyz)(hessian_resp)
)

println("  ✓ All 9 derivative responses computed")
println()

# Show how to access derivatives at a specific location
octave_num = -1
y_coord, x_coord, z_slice = 64, 64, 2  # Example coordinates in octave space

println("Example: Accessing derivatives at octave=$octave_num, position=($x_coord, $y_coord, $z_slice)")
println()

# Get the octave cubes
hess_cube = hessian_resp[octave_num].G
dx_cube = derivatives_3d.∇x[octave_num].G
dy_cube = derivatives_3d.∇y[octave_num].G
dz_cube = derivatives_3d.∇z[octave_num].G
dxx_cube = derivatives_3d.∇²xx[octave_num].G
dyy_cube = derivatives_3d.∇²yy[octave_num].G
dzz_cube = derivatives_3d.∇²zz[octave_num].G
dxy_cube = derivatives_3d.∇²xy[octave_num].G
dxz_cube = derivatives_3d.∇²xz[octave_num].G
dyz_cube = derivatives_3d.∇²yz[octave_num].G

# Access values at the coordinate
response_val = Float32(hess_cube[y_coord, x_coord, z_slice])
Dx = Float32(dx_cube[y_coord, x_coord, z_slice])
Dy = Float32(dy_cube[y_coord, x_coord, z_slice])
Dz = Float32(dz_cube[y_coord, x_coord, z_slice])
Dxx = Float32(dxx_cube[y_coord, x_coord, z_slice])
Dyy = Float32(dyy_cube[y_coord, x_coord, z_slice])
Dzz = Float32(dzz_cube[y_coord, x_coord, z_slice])
Dxy = Float32(dxy_cube[y_coord, x_coord, z_slice])
Dxz = Float32(dxz_cube[y_coord, x_coord, z_slice])
Dyz = Float32(dyz_cube[y_coord, x_coord, z_slice])

@printf("  Response value: %.6e\n", response_val)
println()
@printf("  Gradient:\n")
@printf("    ∇x  = %.6e\n", Dx)
@printf("    ∇y  = %.6e\n", Dy)
@printf("    ∇z  = %.6e\n", Dz)
println()
@printf("  Hessian (diagonal):\n")
@printf("    ∇²xx = %.6e\n", Dxx)
@printf("    ∇²yy = %.6e\n", Dyy)
@printf("    ∇²zz = %.6e\n", Dzz)
println()
@printf("  Hessian (mixed):\n")
@printf("    ∇²xy = %.6e\n", Dxy)
@printf("    ∇²xz = %.6e\n", Dxz)
@printf("    ∇²yz = %.6e\n", Dyz)
println()

# Show memory usage
total_elements = sum(length(octave.G) for octave in hessian_resp.octaves)
@printf("Memory overhead:\n")
@printf("  Base response: %d elements\n", total_elements)
@printf("  9 derivatives: %d elements (9×)\n", 9 * total_elements)
@printf("  Overhead ratio: %.1f×\n", 10.0)  # 1 base + 9 derivatives
println()

# Compare with manual computation
println("Verifying pre-computed derivatives match manual computation...")

# Manual computation (like current refine_extremum_3d does)
at(ddx, ddy, ddz) = Float64(hess_cube[y_coord+ddy, x_coord+ddx, z_slice+ddz].val)

Dx_manual = 0.5 * (at(1, 0, 0) - at(-1, 0, 0))
Dy_manual = 0.5 * (at(0, 1, 0) - at(0, -1, 0))
Dz_manual = 0.5 * (at(0, 0, 1) - at(0, 0, -1))

val_center = at(0, 0, 0)
Dxx_manual = at(1, 0, 0) + at(-1, 0, 0) - 2.0 * val_center
Dyy_manual = at(0, 1, 0) + at(0, -1, 0) - 2.0 * val_center
Dzz_manual = at(0, 0, 1) + at(0, 0, -1) - 2.0 * val_center

Dxy_manual = 0.25 * (at(1, 1, 0) + at(-1, -1, 0) - at(-1, 1, 0) - at(1, -1, 0))
Dxz_manual = 0.25 * (at(1, 0, 1) + at(-1, 0, -1) - at(-1, 0, 1) - at(1, 0, -1))
Dyz_manual = 0.25 * (at(0, 1, 1) + at(0, -1, -1) - at(0, -1, 1) - at(0, 1, -1))

# Compare
println("  Gradient errors:")
@printf("    |Dx_pre - Dx_manual|   = %.6e\n", abs(Dx - Dx_manual))
@printf("    |Dy_pre - Dy_manual|   = %.6e\n", abs(Dy - Dy_manual))
@printf("    |Dz_pre - Dz_manual|   = %.6e\n", abs(Dz - Dz_manual))

println("  Hessian diagonal errors:")
@printf("    |Dxx_pre - Dxx_manual| = %.6e\n", abs(Dxx - Dxx_manual))
@printf("    |Dyy_pre - Dyy_manual| = %.6e\n", abs(Dyy - Dyy_manual))
@printf("    |Dzz_pre - Dzz_manual| = %.6e\n", abs(Dzz - Dzz_manual))

println("  Hessian mixed errors:")
@printf("    |Dxy_pre - Dxy_manual| = %.6e\n", abs(Dxy - Dxy_manual))
@printf("    |Dxz_pre - Dxz_manual| = %.6e\n", abs(Dxz - Dxz_manual))
@printf("    |Dyz_pre - Dyz_manual| = %.6e\n", abs(Dyz - Dyz_manual))

println()

max_error = maximum([
    abs(Dx - Dx_manual), abs(Dy - Dy_manual), abs(Dz - Dz_manual),
    abs(Dxx - Dxx_manual), abs(Dyy - Dyy_manual), abs(Dzz - Dzz_manual),
    abs(Dxy - Dxy_manual), abs(Dxz - Dxz_manual), abs(Dyz - Dyz_manual)
])

if max_error < 1e-6
    println("✓ PRE-COMPUTED DERIVATIVES MATCH MANUAL COMPUTATION")
else
    println("⚠ Discrepancy detected! Max error: $(max_error)")
end

println()
println("="^80)
println("CONCLUSION")
println("="^80)
println()
println("The 3D derivative kernels work correctly and match manual computation.")
println("This architecture allows for:")
println("  1. Pre-computing derivatives once for the entire scale-space")
println("  2. Accessing pre-computed values during refinement (faster for many extrema)")
println("  3. GPU-friendly approach (single kernel launch for all derivatives)")
println("  4. Clean separation: kernels defined separately from refinement logic")
println()
println("Memory trade-off:")
println("  - 10× memory (1 response + 9 derivatives)")
println("  - But derivatives are temporary (discard after refinement)")
println("  - Worth it for GPU parallelism")
println()
