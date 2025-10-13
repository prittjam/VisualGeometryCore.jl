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
Direct (non-separable) derivative kernels for reference and compatibility.
Uses StaticArrays for better performance on small fixed-size kernels.
"""
const DERIVATIVE_KERNELS = (
    # Second derivatives (Hessian components)
    xx = centered(SMatrix{3,3}([0 0 0; 1 -2 1; 0 0 0])),
    yy = centered(SMatrix{3,3}([0 1 0; 0 -2 0; 0 1 0])),
    xy = centered(SMatrix{3,3}([0.25 0 -0.25; 0 0 0; -0.25 0 0.25])),
    
    # First derivatives (Gradient components)
    x = centered(SMatrix{1,3}([-1 0 1])),
    y = centered(SMatrix{3,1}([-1; 0; 1]))
)

"""
Laplacian kernel (trace of Hessian).
Uses StaticArrays for better performance.
"""
const LAPLACIAN_KERNEL = centered(SMatrix{3,3}([0 1 0; 1 -4 1; 0 1 0]))

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