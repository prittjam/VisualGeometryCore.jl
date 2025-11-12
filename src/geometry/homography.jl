# ==============================================================================
# Homography - 2D projective transformations for planar scenes
# ==============================================================================

# Note: PlanarHomographyMat is defined in geometry/transforms/homogeneous.jl as part of the transform hierarchy

# Ensure inv returns PlanarHomographyMat
Base.inv(H::PlanarHomographyMat{T}) where T = PlanarHomographyMat{T}(inv(SMatrix{3,3,T}(H)))

"""
    PlanarHomographyMat(camera::Camera{<:CameraModel{<:Any, PinholeProjection}}) -> PlanarHomographyMat{Float64}

Compute the homography H that maps from a planar board (z=0 plane) to the image.

Only valid for pinhole cameras. For points on the z=0 plane, the homography is:

    H = K [r1 r2 t]

where K is intrinsics, r1 and r2 are the first two columns of the rotation matrix R,
and t is the translation from camera extrinsics.

# Arguments
- `camera::Camera{<:CameraModel{<:Any, PinholeProjection}}`: Camera with pinhole projection

# Returns
- `PlanarHomographyMat{Float64}`: Homography matrix (board coords → image coords)

# Example
```julia
camera = Camera(model, extrinsics)
H = PlanarHomographyMat(camera)
H_inv = inv(H)  # For warping: image → board
```
"""
function PlanarHomographyMat(camera::Camera{<:CameraModel{<:Any, PinholeProjection}})
    # Extract camera parameters (all already unitless Float64)
    K = camera.model.intrinsics.K
    R = camera.extrinsics.R
    t = camera.extrinsics.t

    # For z=0 plane: H = K [r1 r2 t]
    r1 = R[:, 1]
    r2 = R[:, 2]

    # Build homography
    H = K * hcat(r1, r2, t)

    return PlanarHomographyMat{Float64}(H)
end

"""
    ProjectiveMap{T} <: Transformation

Pure 2D projective homography transformation: R^2 → P^2

Applies homography H to 2D Euclidean points, producing 3D homogeneous coordinates (no perspective division).
Follows CoordinateTransformations.jl naming convention (LinearMap, AffineMap, EuclideanMap).

This is a **mathematical transformation** that:
- Takes Euclidean 2D input (Point2 or SVector{2})
- Lifts to homogeneous coordinates (adds w=1)
- Applies 3×3 homography matrix
- Returns homogeneous 3D output (SVector{3})

For image warping with perspective division, use `ImageWarp`.

# Constructors
- `ProjectiveMap(H::PlanarHomographyMat)` - from planar homography matrix

# Example
```julia
# Create projective transformation
H = PlanarHomographyMat(camera)
proj = ProjectiveMap(H)

# Apply to point (returns homogeneous coordinates)
p_homogeneous = proj(Point2(100.0, 200.0))  # SVector{3}

# Compose with PerspectiveMap for Euclidean output
using CoordinateTransformations: PerspectiveMap
transform = PerspectiveMap() ∘ proj  # R^2 → P^2 → R^2
p_euclidean = transform(Point2(100.0, 200.0))  # SVector{2}
```
"""
struct ProjectiveMap{T<:Real} <: CoordinateTransformations.Transformation
    H::PlanarHomographyMat{T}
end

# Pure homogeneous transformation: R^2 → P^2
(h::ProjectiveMap)(p::StaticVector{2}) = h.H * to_affine(p)  # Returns SVector{3}

# Composition in homogeneous space
function Base.:∘(A::ProjectiveMap, B::ProjectiveMap)
    # Matrix composition: (A ∘ B).H = A.H * B.H
    H_composed = A.H.H * B.H.H
    return ProjectiveMap(PlanarHomographyMat{Float64}(H_composed))
end

Base.inv(h::ProjectiveMap{T}) where T = ProjectiveMap(PlanarHomographyMat{T}(inv(h.H)))

# Trait method for ProjectiveMap
trait(::ProjectiveMap) = ProjectiveTrait()
trait(::Type{<:ProjectiveMap}) = ProjectiveTrait()

"""
    ImageWarp{T} <: Transformation

Wrapper for image warping that handles coordinate convention for `ImageTransformations.warp`.

Composes `PerspectiveMap ∘ ProjectiveMap` and handles:
- Input: (row, col) from warp → swap to (x, y)
- Through: PerspectiveMap ∘ ProjectiveMap (full R^2 → P^2 → R^2 pipeline)
- Output: (x, y) → swap back to (row, col)

This is the transformation to use with `ImageTransformations.warp`.

# Constructor
- `ImageWarp(camera::Camera)` - for warping board image to camera view

# Example
```julia
camera = Camera(model, extrinsics)
H_transform = ImageWarp(camera)
camera_view = warp(board_image, H_transform, output_axes)
```
"""
struct ImageWarp{T<:Real} <: CoordinateTransformations.Transformation
    transform::CoordinateTransformations.Transformation
end

"""
    ImageWarp(camera::Camera) -> ImageWarp

Convenience constructor for warping a board image as seen from camera.

Computes the forward homography (board → image) and stores its **inverse** (image → board),
since ImageTransformations.warp needs the inverse mapping to sample from the source image.

Internally composes PerspectiveMap ∘ ProjectiveMap for the full R^2 → P^2 → R^2 pipeline.

# Example
```julia
# Warp board to camera view
H_transform = ImageWarp(camera)
camera_view = warp(board_image, H_transform, output_axes)
```
"""
function ImageWarp(camera::Camera{<:CameraModel{<:Any, PinholeProjection}})
    # Create inverse ProjectiveMap for warping (image → board)
    proj = ProjectiveMap(inv(PlanarHomographyMat(camera)))

    # Compose PerspectiveMap ∘ ProjectiveMap: R^2 → P^2 → R^2
    transform = CoordinateTransformations.PerspectiveMap() ∘ proj

    return ImageWarp{Float64}(transform)
end

function (w::ImageWarp)(rc::SVector{2})
    # Swap (row, col) → (x, y)
    xy = SVector(rc[2], rc[1])

    # Apply composed transformation: PerspectiveMap ∘ ProjectiveMap
    result = w.transform(xy)  # Returns SVector{2}

    # Swap back (x, y) → (row, col)
    return SVector(result[2], result[1])
end

