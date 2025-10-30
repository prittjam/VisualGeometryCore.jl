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
    # warp passes (row, col) but homography expects (x, y)
    # So swap: row=y, col=x → (x, y) = (uv[2], uv[1])
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
    K = SMatrix{3,3,Float64}(camera.model.intrinsics.K)
    R = SMatrix{3,3}(camera.extrinsics.R)
    t = camera.extrinsics.t

    # For z=0 plane: H = K [r1 r2 t]
    r1 = R[:, 1]
    r2 = R[:, 2]

    # Build homography
    H = K * hcat(r1, r2, t)

    return PlanarHomography{Float64}(Tuple(H))
end

"""
    warp(board_image::AbstractArray, camera::Camera{<:CameraModel{<:Any, PinholeProjection}};
         output_size=nothing,
         fillvalue=zero(eltype(board_image)),
         method=Interpolations.Linear())

Warp a planar board (z=0 plane) as seen from the camera using homography-based warping.

Only valid for pinhole cameras. Computes the homography H mapping board → image,
inverts it to get image → board mapping, and warps the board image accordingly.

# Arguments
- `board_image::AbstractArray`: Input image of the board (Matrix{<:Colorant})
- `camera::Camera{<:CameraModel{<:Any, PinholeProjection}}`: Camera with pinhole projection
- `output_size`: Output image size as (height, width). If nothing, uses input image size
- `fillvalue`: Value for out-of-bounds pixels (default: zero(eltype(board_image)))
- `method`: Interpolation method from Interpolations.jl (default: Linear())

# Returns
- Warped image as seen from the camera

# Example
```julia
using VisualGeometryCore
using ImageTransformations
using Interpolations

# Load board pattern
board_image = load("pattern.png")

# Setup camera
camera = Camera(model, extrinsics)

# Render view
camera_view = warp(board_image, camera)
```
"""
function warp(board_image::AbstractArray, camera::Camera{<:CameraModel{<:Any, PinholeProjection}};
              output_size=nothing,
              fillvalue=zero(eltype(board_image)),
              method=Interpolations.Linear())
    # Convert output_size to tuple
    output_tuple = if output_size isa Size2
        Tuple(ceil.(Int, ustrip.((output_size.height, output_size.width))))
    elseif output_size isa Tuple{Int,Int}
        output_size
    else  # nothing
        size(board_image)
    end

    # Create homography transform and warp using native ImageTransformations
    output_axes = (1:output_tuple[1], 1:output_tuple[2])
    return warp(board_image, HomographyTransform(camera), output_axes, fillvalue, method)
end
