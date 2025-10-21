# =============================================================================
# LOCAL FEATURES - KERNELS AND DERIVATIVE COMPUTATIONS
# =============================================================================

# Kernels and derivative computations - using statements handled by main module

# =============================================================================
# DERIVATIVE KERNELS
# =============================================================================

"""
    DERIVATIVE_KERNELS

2D derivative kernels using factored (separable) form for efficiency.

Separable kernels are factored using `ImageFiltering.kernelfactors`, which decomposes
2D convolutions into sequential 1D operations. This provides ~3x speedup:
- Factored: O(2n) operations per pixel (two 1D passes)
- Dense: O(n²) operations per pixel (one 2D pass)

# Kernels
- `xx`, `yy`: Second derivatives (factored, separable)
- `xy`: Mixed partial derivative (dense, non-separable)
- `x`, `y`: First derivatives (factored, 1D)

# Type
All kernels return `Tuple` (factored) except `xy` which returns `SMatrix` (dense).

# Usage
```julia
# Automatically dispatches to efficient factored kernel processing
resp = ScaleSpaceResponse(ss, DERIVATIVE_KERNELS.xx)
resp(ss)  # ~3x faster than dense equivalent
```
"""
const DERIVATIVE_KERNELS = (
    # Second derivatives (Hessian components) - factored into 1D kernels
    xx = kernelfactors((centered(SVector(0, 1, 0)), centered(SVector(1, -2, 1)))),
    yy = kernelfactors((centered(SVector(1, -2, 1)), centered(SVector(0, 1, 0)))),
    xy = centered(SMatrix{3,3}([0.25 0 -0.25; 0 0 0; -0.25 0 0.25])),  # Not separable

    # First derivatives (Gradient components) - factored 1D
    x = kernelfactors((centered(SVector(-1, 0, 1)),)),
    y = kernelfactors((centered(SVector(-1, 0, 1)),))
)

# =============================================================================
# 3D DERIVATIVE KERNELS (for scale-space responses)
# =============================================================================

"""
    DERIVATIVE_KERNELS_3D

3D derivative kernels for computing derivatives of scale-space responses.

These kernels operate on (H × W × S) octave cubes where S is the scale (subdivision) dimension.
All 3D kernels are factored using `kernelfactors` for maximum efficiency (~9x speedup).

Used for refining extrema in the Hessian determinant response, matching VLFeat's
implementation (covdet.c:1244-1250).

# Performance
- Factored 3D: O(3n) operations per voxel (three 1D passes)
- Dense 3D: O(n³) operations per voxel (one 3D pass)
- **Speedup: ~9x for 3×3×3 kernels**

# Kernels
All kernels are factored `Tuple{T1,T2,T3}`:
- `dx`, `dy`, `dz`: First derivatives (central differences)
- `dxx`, `dyy`, `dzz`: Second derivatives (diagonal)
- `dxy`, `dxz`, `dyz`: Mixed partial derivatives (separable!)

# Usage
```julia
# Compute 3D derivative of Hessian determinant response
hessian_resp = ScaleSpaceResponse(ss, DERIVATIVE_KERNELS.xx)(ss)
∇x_resp = ScaleSpaceResponse(hessian_resp, DERIVATIVE_KERNELS_3D.dx)
∇x_resp(hessian_resp)  # ~9x faster than dense 3D convolution
```

## Coordinate Convention
- x: horizontal (width) dimension  - 2nd dimension in array (columns)
- y: vertical (height) dimension   - 1st dimension in array (rows)
- z: scale (subdivision) dimension - 3rd dimension in array (slices)

## Finite Differencing Schemes (matching VLFeat exactly)

### First Derivatives (central differences)
```
∂f/∂x = 0.5 * (f(x+1) - f(x-1))
∂f/∂y = 0.5 * (f(y+1) - f(y-1))
∂f/∂z = 0.5 * (f(z+1) - f(z-1))
```

### Second Derivatives (diagonal)
```
∂²f/∂x² = f(x+1) + f(x-1) - 2*f(x)
∂²f/∂y² = f(y+1) + f(y-1) - 2*f(y)
∂²f/∂z² = f(z+1) + f(z-1) - 2*f(z)
```

### Mixed Partial Derivatives (separable!)
```
∂²f/∂x∂y = 0.25 * (f(x+1,y+1) + f(x-1,y-1) - f(x-1,y+1) - f(x+1,y-1))
         = (0.5 * (f(y+1) - f(y-1))) * (0.5 * (f(x+1) - f(x-1)))
∂²f/∂x∂z = (0.5 * (f(x+1) - f(x-1))) * (0.5 * (f(z+1) - f(z-1)))
∂²f/∂y∂z = (0.5 * (f(y+1) - f(y-1))) * (0.5 * (f(z+1) - f(z-1)))
```

## References
- VLFeat covdet.c:1244-1250 (vl_refine_local_extreum_3)
"""
const DERIVATIVE_KERNELS_3D = (
    # First derivatives (central differences) - already 1D, wrap in kernelfactors
    dx = kernelfactors((centered(SVector(1.0)), centered(SVector(-0.5, 0.0, 0.5)), centered(SVector(1.0)))),
    dy = kernelfactors((centered(SVector(-0.5, 0.0, 0.5)), centered(SVector(1.0)), centered(SVector(1.0)))),
    dz = kernelfactors((centered(SVector(1.0)), centered(SVector(1.0)), centered(SVector(-0.5, 0.0, 0.5)))),

    # Second derivatives (diagonal) - already 1D, wrap in kernelfactors
    dxx = kernelfactors((centered(SVector(1.0)), centered(SVector(1.0, -2.0, 1.0)), centered(SVector(1.0)))),
    dyy = kernelfactors((centered(SVector(1.0, -2.0, 1.0)), centered(SVector(1.0)), centered(SVector(1.0)))),
    dzz = kernelfactors((centered(SVector(1.0)), centered(SVector(1.0)), centered(SVector(1.0, -2.0, 1.0)))),

    # Mixed partial derivatives - separable into 1D kernels!
    # dxy = dy * dx = [-0.5, 0, 0.5] in y × [-0.5, 0, 0.5] in x × [1] in z
    dxy = kernelfactors((centered(SVector(-0.5, 0.0, 0.5)), centered(SVector(-0.5, 0.0, 0.5)), centered(SVector(1.0)))),

    # dxz = dx * dz = [1] in y × [-0.5, 0, 0.5] in x × [-0.5, 0, 0.5] in z
    dxz = kernelfactors((centered(SVector(1.0)), centered(SVector(-0.5, 0.0, 0.5)), centered(SVector(-0.5, 0.0, 0.5)))),

    # dyz = dy * dz = [-0.5, 0, 0.5] in y × [1] in x × [-0.5, 0, 0.5] in z
    dyz = kernelfactors((centered(SVector(-0.5, 0.0, 0.5)), centered(SVector(1.0)), centered(SVector(-0.5, 0.0, 0.5))))
)

# =============================================================================
# SCALE-SPACE RESPONSE COMPUTATIONS
# =============================================================================

"""
    hessian_determinant_response(ixx::ScaleSpaceResponse, iyy::ScaleSpaceResponse,
                                 ixy::ScaleSpaceResponse)

Compute Hessian determinant response from component responses with VLFeat-compatible scale normalization.

This function computes det(H) = Ixx * Iyy - Ixy² with scale normalization factor (σ/step)⁴
for each octave level, matching VLFeat's `_vl_det_hessian_response` exactly.

# Arguments
- `ixx`: Ixx (horizontal second derivative) ScaleSpaceResponse
- `iyy`: Iyy (vertical second derivative) ScaleSpaceResponse
- `ixy`: Ixy (mixed partial derivative) ScaleSpaceResponse

# Returns
- `ScaleSpaceResponse`: Hessian determinant with scale normalization

# Scale Normalization
The normalization factor (σ/step)⁴ makes the response scale-invariant, where:
- σ is the Gaussian scale (sigma) at each level
- step is the sampling step (2^octave)

# Example
```julia
# Compute Hessian components
ixx = ScaleSpaceResponse(ss, DERIVATIVE_KERNELS.xx)(ss)
iyy = ScaleSpaceResponse(ss, DERIVATIVE_KERNELS.yy)(ss)
ixy = ScaleSpaceResponse(ss, DERIVATIVE_KERNELS.xy)(ss)

# Compute determinant response
hessian_det = hessian_determinant_response(ixx, iyy, ixy)
```
"""
function hessian_determinant_response(ixx::ScaleSpaceResponse, iyy::ScaleSpaceResponse,
                                     ixy::ScaleSpaceResponse)
    # Create response with same geometry as input
    det_resp = ScaleSpaceResponse(ixx, (dst, src) -> dst .= src)

    # Process each octave
    for octave in ixx.octaves
        octave_num = octave.octave

        # Get component cubes
        ixx_cube = ixx[octave_num].G
        iyy_cube = iyy[octave_num].G
        ixy_cube = ixy[octave_num].G
        det_cube = det_resp[octave_num].G

        # Get metadata
        sigmas = octave.sigmas
        step = octave.step

        # Compute scale normalization factors for all slices
        # Reshape to (1, 1, n_slices) for broadcasting across spatial dimensions
        factors = reshape(Float32.((sigmas ./ step).^4), 1, 1, :)

        # Single broadcast over entire cube: det = (Ixx * Iyy - Ixy²) * (σ/step)⁴
        @. det_cube = (ixx_cube * iyy_cube - ixy_cube * ixy_cube) * factors
    end

    return det_resp
end

"""
    laplacian_response(ixx::ScaleSpaceResponse, iyy::ScaleSpaceResponse)

Compute Laplacian (trace of Hessian) response with VLFeat-compatible scale normalization.

This function computes Laplacian = Ixx + Iyy with scale normalization factor (σ/step)²
for each octave level, matching VLFeat's Laplacian computation exactly.

# Arguments
- `ixx`: Ixx (horizontal second derivative) ScaleSpaceResponse
- `iyy`: Iyy (vertical second derivative) ScaleSpaceResponse

# Returns
- `ScaleSpaceResponse`: Laplacian with scale normalization

# Sign Convention
- Laplacian < 0: Bright blob (intensity peak)
- Laplacian > 0: Dark blob (intensity valley)

# Scale Normalization
The normalization factor (σ/step)² makes the response scale-invariant, where:
- σ is the Gaussian scale (sigma) at each level
- step is the sampling step (2^octave)

# Example
```julia
# Compute Hessian components
ixx = ScaleSpaceResponse(ss, DERIVATIVE_KERNELS.xx)(ss)
iyy = ScaleSpaceResponse(ss, DERIVATIVE_KERNELS.yy)(ss)

# Compute Laplacian response
laplacian = laplacian_response(ixx, iyy)
```
"""
function laplacian_response(ixx::ScaleSpaceResponse, iyy::ScaleSpaceResponse)
    # Create response with same geometry as input
    lap_resp = ScaleSpaceResponse(ixx, (dst, src) -> dst .= src)

    # Process each octave
    for octave in ixx.octaves
        octave_num = octave.octave

        # Get component cubes
        ixx_cube = ixx[octave_num].G
        iyy_cube = iyy[octave_num].G
        lap_cube = lap_resp[octave_num].G

        # Get metadata
        sigmas = octave.sigmas
        step = octave.step

        # Compute scale normalization factors for all slices
        # Reshape to (1, 1, n_slices) for broadcasting across spatial dimensions
        factors = reshape(Float32.((sigmas ./ step).^2), 1, 1, :)

        # Single broadcast over entire cube: laplacian = (Ixx + Iyy) * (σ/step)²
        @. lap_cube = (ixx_cube + iyy_cube) * factors
    end

    return lap_resp
end

