# ==============================================================================
# Homography - 2D projective transformations for planar scenes
# ==============================================================================

"""
    HomographyTransform{T}

A 2D projective homography transformation for use with ImageTransformations.warp.

Applies the homography H to 2D points in homogeneous coordinates.
"""
struct HomographyTransform{T<:Real} <: CoordinateTransformations.Transformation
    H::SMatrix{3,3,T}
end

(h::HomographyTransform)(uv) = begin
    # warp passes (row, col) but homography expects (x, y)
    # So swap: row=y, col=x → (x, y) = (uv[2], uv[1])
    p = h.H * SVector(uv[2], uv[1], 1.0)
    # Return in (row, col) order: (y, x) = (p[2], p[1])
    SVector(p[2]/p[3], p[1]/p[3])
end

Base.inv(h::HomographyTransform) = HomographyTransform(inv(h.H))

"""
    planar_homography(camera::Camera{<:CameraModel{<:Any, PinholeProjection}})

Compute the homography H that maps from a planar board (z=0 plane) to the image.

Only valid for pinhole cameras. For points on the z=0 plane, the homography is:

    H = K [r1 r2 t]

where K is intrinsics, r1 and r2 are the first two columns of the rotation matrix R,
and t is the translation from camera extrinsics.

# Arguments
- `camera::Camera{<:CameraModel{<:Any, PinholeProjection}}`: Camera with pinhole projection

# Returns
- `H::SMatrix{3,3}`: Homography matrix (board coords → image coords)

# Example
```julia
camera = Camera(model, extrinsics)
H = planar_homography(camera)
H_inv = inv(H)  # For warping: image → board
```
"""
function planar_homography(camera::Camera{<:CameraModel{<:Any, PinholeProjection}})
    # Extract camera parameters
    K = ustrip(camera.model.intrinsics.K)  # Intrinsics (unitless)
    R = SMatrix{3,3}(camera.extrinsics.R)  # Rotation
    t = camera.extrinsics.t                 # Translation

    # For z=0 plane: H = K [r1 r2 t]
    r1 = R[:, 1]
    r2 = R[:, 2]

    # Build homography
    H = K * hcat(r1, r2, t)

    return SMatrix{3,3}(H)
end

"""
    render_board(board_image, camera::Camera{<:CameraModel{<:Any, PinholeProjection}};
                 output_size=nothing,
                 fillvalue=zero(eltype(board_image)),
                 method=Interpolations.Linear())

Render a planar board (z=0 plane) as seen from the camera using homography-based warping.

Only valid for pinhole cameras. Computes the homography H mapping board → image,
inverts it to get image → board mapping, and warps the board image accordingly.

# Arguments
- `board_image`: Input image of the board (Matrix{<:Colorant})
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
camera_view = render_board(board_image, camera)
```
"""
function render_board(board_image, camera::Camera{<:CameraModel{<:Any, PinholeProjection}};
                     output_size=nothing,
                     fillvalue=zero(eltype(board_image)),
                     method=Interpolations.Linear())

    # Compute homography for z=0 plane
    H = planar_homography(camera)

    # Create transformation: board → image (forward)
    H_transform = HomographyTransform(H)

    # Determine output size
    if output_size === nothing
        # Default to same size as input board image
        output_size = size(board_image)
    end

    # Warp the board image
    # warp automatically inverts the transform (board → image becomes image → board)
    output_axes = (1:output_size[1], 1:output_size[2])
    return warp(board_image, H_transform, output_axes, fillvalue, method)
end
