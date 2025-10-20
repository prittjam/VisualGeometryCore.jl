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
Factored (separable) derivative kernels for efficient computation.
Uses ImageFiltering.kernelfactors to separate 2D kernels into 1D operations.
This provides ~3x speedup for separable kernels (O(2n) vs O(n²) per pixel).
"""
const DERIVATIVE_KERNELS = (
    # Second derivatives (Hessian components) - factored into 1D kernels
    xx = kernelfactors((centered(SVector(0, 1, 0)), centered(SVector(1, -2, 1)))),
    yy = kernelfactors((centered(SVector(1, -2, 1)), centered(SVector(0, 1, 0)))),
    xy = centered(SMatrix{3,3}([0.25 0 -0.25; 0 0 0; -0.25 0 0.25])),  # Not separable

    # First derivatives (Gradient components) - already 1D
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
3D derivative kernels for computing derivatives of scale-space responses.
These operate on (H × W × S) octave cubes where S is the scale (subdivision) dimension.

Uses factored (separable) kernels for efficient computation. All 3D kernels are separable
into 1D operations along each axis, providing significant speedup.

Used for refining extrema in the Hessian determinant response, matching VLFeat's
implementation (covdet.c:1244-1250).

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