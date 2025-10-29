# ==============================================================================
# Projection Models - Different types of camera projection (pinhole, fisheye, etc.)
# ==============================================================================

"""
    AbstractProjectionModel

Base type for all projection models (pinhole, fisheye, orthographic, etc.).
"""
abstract type AbstractProjectionModel end

"""
    PinholeProjection <: AbstractProjectionModel

Standard pinhole projection model: perspective projection with division by Z coordinate.

The projection is: u = K * [X/Z, Y/Z, 1]ᵀ where K is the calibration matrix.
"""
struct PinholeProjection <: AbstractProjectionModel end

"""
    FisheyeProjection <: AbstractProjectionModel

Fisheye (wide-angle) projection model with radial distortion.
To be implemented in the future.
"""
struct FisheyeProjection <: AbstractProjectionModel end

"""
    OrthographicProjection <: AbstractProjectionModel

Orthographic (parallel) projection model with no perspective effects.
To be implemented in the future.
"""
struct OrthographicProjection <: AbstractProjectionModel end

# ============================================================================
# Transform Builders - Dispatch on Projection Model
# ============================================================================

"""
    build_projection_transform(intrinsics::AbstractIntrinsics, projection::AbstractProjectionModel)

Build composed transform for camera-space to pixel-space projection.

Dispatches on projection model type to build the appropriate composition of
CoordinateTransformations. Returns a callable transform object.

# Returns
Composed transform: `Intrinsics ∘ Projection`
- Input: 3D point in camera coordinates (with units)
- Output: 2D point in pixel coordinates (unitless, will have px added)
"""
function build_projection_transform end

"""
    build_projection_transform(intrinsics, ::PinholeProjection)

Build pinhole projection transform: `AffineMap ∘ PerspectiveMap`

The composition is:
1. PerspectiveMap: [X, Y, Z] → [X/Z, Y/Z]  (perspective division)
2. AffineMap: Apply K[1:2, 1:2] and K[1:2, 3] (calibration)
"""
function build_projection_transform(intrinsics::AbstractIntrinsics, ::PinholeProjection)
    K = intrinsics.K
    # Extract 2×2 linear part and 2D translation - K already has pixel units
    A = SMatrix{2,2}(K[1,1], K[2,1], K[1,2], K[2,2])
    t = SVector{2}(K[1,3], K[2,3])

    # Compose: Calibration ∘ Perspective
    # Units flow naturally:
    #   - PerspectiveMap: [X, Y, Z] with length units → [X/Z, Y/Z] dimensionless
    #   - AffineMap: A (px) * x_norm (dimensionless) + t (px) → result (px)
    return AffineMap(A, t) ∘ PerspectiveMap()
end

# Future: Add builders for other projection models
# build_projection_transform(intrinsics, ::FisheyeProjection) = ...
# build_projection_transform(intrinsics, ::OrthographicProjection) = ...
