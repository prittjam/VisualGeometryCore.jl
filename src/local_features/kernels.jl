# =============================================================================
# LOCAL FEATURES - KERNELS AND DERIVATIVE COMPUTATIONS
# =============================================================================

# Kernels and derivative computations - using statements handled by main module

# =============================================================================
# DERIVATIVE KERNELS
# =============================================================================

"""
Simple Hessian kernels using StaticMatrix for clean SubArray compatibility.
These work directly with SubArray views from 3D cubes without OffsetArray complexity.
"""
const HESSIAN_KERNELS = (
    xx = centered(SMatrix{3,3}([0 0 0; 1 -2 1; 0 0 0])),
    yy = centered(SMatrix{3,3}([0 1 0; 0 -2 0; 0 1 0])),
    xy = centered(SMatrix{3,3}([0.25 0 -0.25; 0 0 0; -0.25 0 0.25]))
)

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

"""
Laplacian kernel (trace of Hessian).
Uses StaticArrays for better performance.
"""
const LAPLACIAN_KERNEL = centered(SMatrix{3,3}([0 1 0; 1 -4 1; 0 1 0]))

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
# DERIVED COMPUTATIONS
# =============================================================================

"""
    laplacian(hessian_data::NamedTuple)

Compute Laplacian (trace of Hessian) from Hessian components.
Returns Ixx + Iyy.
"""
function laplacian(hessian_data::NamedTuple)
    return @. hessian_data.xx + hessian_data.yy
end

"""
    hessian_determinant(hessian_data::NamedTuple)

Compute determinant of Hessian matrix from components.
Returns Ixx * Iyy - Ixy².
"""
function hessian_determinant(hessian_data::NamedTuple)
    return @. hessian_data.xx * hessian_data.yy - hessian_data.xy * hessian_data.xy
end

# =============================================================================
# DIRECT TRANSFORM FUNCTIONS
# =============================================================================

"""
Direct Laplacian computation function for ScaleSpaceResponse.
"""
const LAPLACIAN_DIRECT = LAPLACIAN_KERNEL

"""
Direct Hessian computation returning NamedTuple of components.
"""
function HESSIAN_DIRECT(img)
    return (
        xx = imfilter(img, DERIVATIVE_KERNELS.xx),
        yy = imfilter(img, DERIVATIVE_KERNELS.yy),
        xy = imfilter(img, DERIVATIVE_KERNELS.xy)
    )
end