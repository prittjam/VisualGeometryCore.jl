# ==============================================================================
# Homography - 2D projective transformations for planar scenes
# ==============================================================================

# Define PlanarHomography as a 3×3 static matrix type
@smatrix_wrapper PlanarHomography 3 3

"""
    PlanarHomography(camera::Camera{<:CameraModel{<:Any, PinholeProjection}}) -> PlanarHomography{Float64}

Construct a planar homography directly from a camera for z=0 plane mapping.

# Example
```julia
camera = Camera(model, extrinsics)
H = PlanarHomography(camera)
```
"""
PlanarHomography(camera::Camera{<:CameraModel{<:Any, PinholeProjection}}) =
    planar_homography(camera)

"""
    HomographyTransform{T}

A 2D projective homography transformation for use with ImageTransformations.warp.

Applies the homography H to 2D points in homogeneous coordinates.
"""
struct HomographyTransform{T<:Real} <: CoordinateTransformations.Transformation
    H::PlanarHomography{T}
end

# Convenience constructors
HomographyTransform(H::SMatrix{3,3,T}) where T = HomographyTransform(PlanarHomography{T}(Tuple(H)))

"""
    HomographyTransform(camera::Camera{<:CameraModel{<:Any, PinholeProjection}}) -> HomographyTransform

Construct a HomographyTransform directly from a camera for z=0 plane warping.

# Example
```julia
camera = Camera(model, extrinsics)
H_transform = HomographyTransform(camera)
warped = warp(board_image, H_transform, axes)
```
"""
HomographyTransform(camera::Camera{<:CameraModel{<:Any, PinholeProjection}}) =
    HomographyTransform(planar_homography(camera))

(h::HomographyTransform)(uv) = begin
    # warp passes (row, col) as SVector{2,Int64}, but homography expects (x, y)
    # Swap indices: row=y, col=x → (x, y) = (uv[2], uv[1])
    p = h.H * to_affine(SVector(uv[2], uv[1]))
    # Convert from homogeneous and return in (row, col) order: (y, x)
    result = to_euclidean(p)
    SVector(result[2], result[1])
end

Base.inv(h::HomographyTransform{T}) where T = HomographyTransform(PlanarHomography{T}(inv(h.H)))

"""
    planar_homography(camera::Camera{<:CameraModel{<:Any, PinholeProjection}}) -> PlanarHomography{Float64}

Compute the homography H that maps from a planar board (z=0 plane) to the image.

Only valid for pinhole cameras. For points on the z=0 plane, the homography is:

    H = K [r1 r2 t]

where K is intrinsics, r1 and r2 are the first two columns of the rotation matrix R,
and t is the translation from camera extrinsics.

# Arguments
- `camera::Camera{<:CameraModel{<:Any, PinholeProjection}}`: Camera with pinhole projection

# Returns
- `PlanarHomography{Float64}`: Homography matrix (board coords → image coords)

# Example
```julia
camera = Camera(model, extrinsics)
H = planar_homography(camera)
H_inv = inv(H)  # For warping: image → board
```
"""
function planar_homography(camera::Camera{<:CameraModel{<:Any, PinholeProjection}})
    # Extract camera parameters (all already unitless Float64)
    K = camera.model.intrinsics.K
    R = camera.extrinsics.R
    t = camera.extrinsics.t

    # For z=0 plane: H = K [r1 r2 t]
    r1 = R[:, 1]
    r2 = R[:, 2]

    # Build homography
    H = K * hcat(r1, r2, t)

    return PlanarHomography{Float64}(H)
end

