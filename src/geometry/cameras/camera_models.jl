# ==============================================================================
# CameraModel - Composable camera model combining intrinsics and projection
# ==============================================================================

"""
    CameraModel{I, P, T}

Camera model combining intrinsics and projection with cached composed transform.

The transform field contains the composed projection: `Intrinsics ∘ Projection`,
built at construction time for efficient repeated projection operations.

# Fields
- `intrinsics::I`: Camera intrinsics (LogicalIntrinsics or PhysicalIntrinsics)
- `projection::P`: Projection model (PinholeProjection, FisheyeProjection, etc.)
- `transform::T`: Cached composed transform (Camera coords → Pixel coords)

# Examples
```julia
# Logical camera with pinhole projection
K = CameraCalibrationMatrix(800.0px, [320.0px, 240.0px])
logical = LogicalIntrinsics(K)
camera = CameraModel(logical, PinholeProjection())

# Physical camera with pinhole projection
f = 16.0mm
pitch = Size2(width=5.86μm/px, height=5.86μm/px)
pp = [320.0px, 240.0px]
physical = PhysicalIntrinsics(f, pitch, pp)
camera = CameraModel(physical, PinholeProjection())
```
"""
struct CameraModel{I<:AbstractIntrinsics, P<:AbstractProjectionModel, T}
    intrinsics::I
    projection::P
    transform::T  # Composed: Intrinsics ∘ Projection transform
end

# ============================================================================
# CameraModel Constructors
# ============================================================================

"""
    CameraModel(intrinsics::AbstractIntrinsics, projection::AbstractProjectionModel)

Construct a CameraModel from intrinsics and projection model.

Builds and caches the composed projection transform at construction time.

# Examples
```julia
# Logical camera with pinhole projection
K = CameraCalibrationMatrix(800.0px, [320.0px, 240.0px])
logical = LogicalIntrinsics(K)
camera = CameraModel(logical, PinholeProjection())

# Physical camera with pinhole projection
f = 16.0mm
pitch = Size2(width=5.86μm/px, height=5.86μm/px)
pp = [320.0px, 240.0px]
physical = PhysicalIntrinsics(f, pitch, pp)
camera = CameraModel(physical, PinholeProjection())

# Using keyword argument with default pinhole projection
camera = CameraModel(logical)  # Defaults to PinholeProjection()
```
"""
function CameraModel(intrinsics::AbstractIntrinsics, projection::AbstractProjectionModel)
    transform = build_projection_transform(intrinsics, projection)
    return CameraModel(intrinsics, projection, transform)
end

# Convenience constructor with default pinhole projection
CameraModel(intrinsics::AbstractIntrinsics; projection=PinholeProjection()) =
    CameraModel(intrinsics, projection)

"""
    CameraModel(f::Len, pitch::Size2{<:LogicalPitch}, pp::AbstractVector{<:PixelWidth}; projection=PinholeProjection()) -> CameraModel

Convenience constructor to build a complete physical camera model from parameters.

# Example
```julia
camera = CameraModel(16.0mm, Size2(width=5.86μm/px, height=5.86μm/px), [320.0px, 240.0px])
```
"""
CameraModel(f::Len, pitch::Size2{<:LogicalPitch}, pp::AbstractVector{<:PixelWidth};
            projection=PinholeProjection()) =
    CameraModel(PhysicalIntrinsics(f, pitch, pp), projection)

"""
    CameraModel(f::PixelWidth, pp::AbstractVector{<:PixelWidth}; projection=PinholeProjection()) -> CameraModel

Convenience constructor to build a logical camera model from focal length and principal point (both in pixels).

# Example
```julia
camera = CameraModel(800.0px, [320.0px, 240.0px])
```
"""
CameraModel(f::PixelWidth, pp::AbstractVector{<:PixelWidth};
            projection=PinholeProjection()) =
    CameraModel(LogicalIntrinsics(f, pp), projection)

"""
    CameraModel(f::Tuple{T,T}, pp::AbstractVector{<:PixelWidth}; projection=PinholeProjection()) where T<:PixelWidth -> CameraModel

Convenience constructor to build a logical camera model with anisotropic focal lengths (fx ≠ fy).

# Example
```julia
camera = CameraModel((800.0px, 805.0px), [320.0px, 240.0px])
```
"""
CameraModel(f::Tuple{T,T}, pp::AbstractVector{<:PixelWidth};
            projection=PinholeProjection()) where T<:PixelWidth =
    CameraModel(LogicalIntrinsics(f, pp), projection)

# ============================================================================
# Projection and Backprojection Methods
# ============================================================================

"""
    project(model::CameraModel, X::AbstractVector) -> Point2

Project 3D point X in camera coordinates to 2D pixel coordinates.

Uses the cached composed transform for efficient projection.

# Arguments
- `model::CameraModel`: Camera model with cached projection transform
- `X::AbstractVector`: 3D point in camera coordinates [X, Y, Z] (with units)

# Returns
- Point2 with pixel coordinates (including px units)

# Example
```julia
camera = CameraModel(16.0mm, Size2(width=5.86μm/px, height=5.86μm/px), [320.0px, 240.0px])
X = SVector(10.0mm, 5.0mm, 100.0mm)
u = project(camera, X)  # Returns Point2 in pixels
```
"""
function project(model::CameraModel, X::AbstractVector)
    # Apply cached composed transform with natural unit flow:
    # - PerspectiveMap strips units via division: [X,Y,Z] in mm → [X/Z, Y/Z] dimensionless
    # - AffineMap: unitless matrix * dimensionless + px translation → px result
    return model.transform(X)
end

"""
    backproject(model::CameraModel, u::StaticVector{2,Float64}) -> Vec3

Backproject 2D pixel coordinates to 3D ray direction.

Inputs and outputs are unitless Float64 (assumed units: px for input, unitless direction for output).

The return type and semantics depend on the intrinsics type:
- LogicalIntrinsics: Returns normalized unitless direction
- PhysicalIntrinsics: Returns normalized direction (computed with metric information)

# Arguments
- `model::CameraModel`: Camera model with intrinsics and projection
- `u::StaticVector{2,Float64}`: 2D pixel coordinates [u, v] (unitless, in px)

# Returns
- `Vec3`: Normalized ray direction (unitless)

# Example
```julia
camera = CameraModel(16.0mm, Size2(width=5.86μm/px, height=5.86μm/px), [320.0px, 240.0px])
u = SVector(400.0, 300.0)  # px (unitless)
ray = backproject(camera, u)  # Returns normalized Vec3 direction
```
"""
backproject(model::CameraModel, u::StaticVector{2,Float64}) =
    backproject(model.intrinsics, model.projection, u)

# Batched version for multiple points
backproject(model::CameraModel, points::AbstractVector{<:StaticVector{2,Float64}}) =
    backproject(model.intrinsics, model.projection, points)

"""
    backproject(intrinsics::LogicalIntrinsics, ::PinholeProjection, u::StaticVector{2}) -> Vec3

Backproject for LogicalIntrinsics: Returns normalized unitless ray direction.

Since no physical parameters are available, the ray has no metric scale information.

# Arguments
- `u`: 2D pixel coordinates (accepts both unitful and plain Float types)

# Returns
- `Vec3`: Normalized ray direction
"""
function backproject(intrinsics::LogicalIntrinsics, ::PinholeProjection,
                     u::StaticVector{2,Float64})
    # K and u are both unitless Float64
    h = to_affine(u)

    # Use precomputed K_inv for efficiency
    # Both K_inv and h are unitless, result is unitless
    ray = intrinsics.K_inv * h

    # Return normalized ray directly
    return normalize(ray)
end

"""
    backproject(intrinsics::PhysicalIntrinsics, ::PinholeProjection, u::StaticVector{2}) -> Vec3

Backproject for PhysicalIntrinsics: Returns normalized direction using metric computation.

The focal length is used in the computation, though the result is still normalized.

# Arguments
- `u`: 2D pixel coordinates (accepts both unitful and plain Float types)

# Returns
- `Vec3`: Normalized ray direction
"""
function backproject(intrinsics::PhysicalIntrinsics, ::PinholeProjection,
                     u::StaticVector{2,Float64})
    # K and u are both unitless Float64
    h = to_affine(u)

    # Use precomputed K_inv for efficiency
    # Both K_inv and h are unitless, result is unitless
    ray_unnormalized = intrinsics.K_inv * h

    # ray_unnormalized is dimensionless
    # Scale by focal length (stored unitless in mm) for metric interpretation
    ray_metric = SVector{3}(
        ray_unnormalized[1] * intrinsics.f,
        ray_unnormalized[2] * intrinsics.f,
        intrinsics.f
    )
    # Return normalized ray directly (ray_metric is unitless)
    return normalize(ray_metric)
end

"""
    unproject(model::CameraModel{<:PhysicalIntrinsics}, u::AbstractVector{Float64}, depth::Float64) -> SVector{3,Float64}

Unproject 2D pixel coordinates to 3D point given depth.

All inputs and outputs are unitless Float64 (assumed units: px for u, mm for depth and result).

Only available for PhysicalIntrinsics since metric scale information is required.

# Arguments
- `model::CameraModel{<:PhysicalIntrinsics}`: Camera model with physical intrinsics
- `u::AbstractVector{Float64}`: 2D pixel coordinates [u, v] (unitless, in px)
- `depth::Float64`: Depth (Z coordinate) (unitless, in mm)

# Returns
- SVector{3,Float64} with 3D point coordinates [X, Y, Z] (unitless, in mm)

# Example
```julia
camera = CameraModel(16.0mm, Size2(width=5.86μm/px, height=5.86μm/px), [320.0px, 240.0px])
u = [400.0, 300.0]  # px (unitless)
depth = 100.0       # mm (unitless)
X = unproject(camera, u, depth)  # Returns 3D point (unitless mm)
```
"""
unproject(model::CameraModel{<:PhysicalIntrinsics}, u::AbstractVector{Float64}, depth::Float64) =
    unproject(model.intrinsics, model.projection, u, depth)

"""
    unproject(intrinsics::PhysicalIntrinsics, ::PinholeProjection, u::AbstractVector{Float64}, depth::Float64) -> SVector{3,Float64}

Unproject for pinhole model with PhysicalIntrinsics.
"""
function unproject(intrinsics::PhysicalIntrinsics, ::PinholeProjection,
                   u::AbstractVector{Float64}, depth::Float64)
    # Use precomputed K_inv for efficiency
    h = to_affine(u)
    ray_normalized = intrinsics.K_inv * h

    # Compute 3D point: [X, Y, Z] where Z = depth
    Z = depth
    X = ray_normalized[1] * Z
    Y = ray_normalized[2] * Z

    return SVector{3}(X, Y, Z)
end
