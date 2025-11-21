# ==============================================================================
# Camera Utilities - Helper functions for focal length and camera positioning
# ==============================================================================

"""
    focal_length(θ::Rad, resolution::Size2; dimension=:horizontal)

Calculate focal length in pixels from field of view angle (in radians) and resolution.

# Arguments
- `θ::Rad`: Field of view angle in radians
- `resolution::Size2`: Image resolution (width, height) in pixels
- `dimension::Symbol`: Which dimension the FOV applies to (`:horizontal` or `:vertical`)

# Returns
- Focal length in pixels (same logical unit as resolution)

# Examples
```julia
focal_length(0.698rad, Size2(width=1280px, height=960px))
```
"""
function focal_length(θ::Rad, resolution::Size2; dimension=:horizontal)
    if dimension === :horizontal
        f = 0.5*resolution.width
    elseif dimension === :vertical
        f = 0.5*resolution.height
    else
        throw(ArgumentError("dimension must be :horizontal or :vertical"))
    end

    return f / tan(ustrip(θ)/2)
end

"""
    focal_length(θ::Deg, resolution::Size2; dimension=:horizontal)

Calculate focal length in pixels from field of view angle (in degrees) and resolution.
Converts to radians and calls the radian version.

# Examples
```julia
focal_length(40.0°, Size2(width=1280px, height=960px))
```
"""
focal_length(θ::Deg, resolution::Size2; dimension=:horizontal) =
    focal_length(uconvert(rad, θ), resolution; dimension=dimension)

"""
    focal_length(θ, sensor; dimension=:horizontal)

Calculate physical focal length from field of view angle and sensor specifications.

# Arguments
- `θ`: Field of view angle (degrees or radians)
- `sensor`: Sensor with `resolution` and `pitch` fields
- `dimension::Symbol`: Which dimension the FOV applies to (`:horizontal` or `:vertical`)

# Returns
- Physical focal length in millimeters

# Examples
```julia
sensor = CMOS_SENSORS["Sony"]["IMX174"]
f_mm = focal_length(40.0°, sensor)
```
"""
function focal_length(θ, sensor; dimension=:horizontal)
    # Get focal length in pixels
    f_px = focal_length(θ, sensor.resolution; dimension=dimension)

    # Convert to physical length using pixel pitch
    # Use width pitch for horizontal FOV, height pitch for vertical FOV
    pitch = (dimension === :horizontal) ? sensor.pitch.width : sensor.pitch.height

    # f_mm = f_px × pitch (LogicalWidth × LogicalPitch = Length)
    # pitch is now properly typed as LogicalPitch (𝐋/𝐍), so multiplication works naturally
    return uconvert(mm, f_px * pitch)
end

"""
    lookat(eye::StaticVector{3, <:Unitful.Length}, target::StaticVector{3, <:Unitful.Length}, up::StaticVector{3, <:Real})

Create a camera extrinsics transform that looks from `eye` position toward `target`.

Units are enforced at construction but stored internally as unitless Float64 in mm.

Uses **Computer Vision convention**:
- +Z axis points FORWARD (toward target)
- +X axis points RIGHT
- +Y axis points DOWN
- Camera -Y aligns with world up direction

# Arguments
- `eye::StaticVector{3, <:Unitful.Length}`: Camera position in world coordinates (with units)
- `target::StaticVector{3, <:Unitful.Length}`: Point to look at in world coordinates (with units)
- `up::StaticVector{3, <:Real}`: World up direction, unitless (typically [0, -1, 0] for CV convention where -Y is up)

# Returns
- `EuclideanMap`: World-to-camera transformation

# Example
```julia
# Camera at (0, 0, 0) looking toward (0, 0, 1000mm), with world -Y pointing up
extrinsics = lookat(SVector(0.0mm, 0.0mm, 0.0mm), SVector(0.0mm, 0.0mm, 1000.0mm), SVector(0.0, -1.0, 0.0))
```
"""
function lookat(eye::StaticVector{3, <:Unitful.Length}, target::StaticVector{3, <:Unitful.Length}, up::StaticVector{3, <:Real})
    # Convert to mm and strip units
    eye_mm = ustrip.(Float64, mm, eye)
    target_mm = ustrip.(Float64, mm, target)

    # Call the unitless version
    return lookat(SVector{3,Float64}(eye_mm), SVector{3,Float64}(target_mm), up)
end

"""
    lookat(eye::StaticVector{3, <:Real}, target::StaticVector{3, <:Real}, up::StaticVector{3, <:Real})

Create a camera extrinsics transform that looks from `eye` position toward `target`.

Unitless version for internal use and backwards compatibility.

Uses **Computer Vision convention**:
- +Z axis points FORWARD (toward target)
- +X axis points RIGHT
- +Y axis points DOWN
- Camera -Y aligns with world up direction

# Arguments
- `eye::StaticVector{3}`: Camera position in world coordinates (unitless, assumed mm)
- `target::StaticVector{3}`: Point to look at in world coordinates (unitless, assumed mm)
- `up::StaticVector{3}`: World up direction (typically [0, -1, 0] for CV convention where -Y is up)

# Returns
- `EuclideanMap`: World-to-camera transformation

# Example
```julia
# Camera at (0, 0, 0) looking toward (0, 0, 1000), with world -Y pointing up
extrinsics = lookat(SVector(0.0, 0.0, 0.0), SVector(0.0, 0.0, 1000.0), SVector(0.0, -1.0, 0.0))
```
"""
function lookat(eye::StaticVector{3, <:Real}, target::StaticVector{3, <:Real}, up::StaticVector{3, <:Real})
    # Computer vision convention: camera looks down +Z axis
    # Right-handed system: +Z forward, +X right, +Y down
    # The "up" parameter is the world up direction (typically [0, -1, 0])
    # Camera -Y aligns with world up (since camera +Y is down)

    # Forward direction: camera looks along +Z
    zaxis = normalize(target - eye)           # forward (+Z)
    xaxis = normalize(cross(up, zaxis))       # right (+X)
    yaxis = -normalize(cross(zaxis, xaxis))   # down (+Y), note: -Y is up

    # Rotation part (camera basis vectors as columns)
    rotation = hcat(xaxis, yaxis, zaxis)

    # Translation: move the world so that the eye is at the origin
    eye_vec = SVector{3}(eye[1], eye[2], eye[3])
    translation = -rotation' * eye_vec

    return EuclideanMap(Rotations.RotMatrix{3}(rotation'), translation)
end

# =============================================================================
# P3P Pose Sampling - Moved to random.jl
# =============================================================================
# Note: sample_p3p functions have been moved to cameras/random.jl for better organization

# =============================================================================
# P3P Solver with Correspondences
# =============================================================================

"""
    p3p(model::CameraModel, csponds)

Solve the Perspective-3-Point (P3P) problem using correspondences.

This is a thin wrapper around the core `p3p(rays, X)` solver that accepts
correspondences directly. It extracts the 3D world points and 2D image points
from the correspondences, backprojects the image points to rays, and calls
the core solver.

# Arguments
- `model::CameraModel`: Camera model for backprojecting image points to rays
- `csponds`: Correspondences containing exactly 3 point pairs (3D world → 2D image)
  Can be a StructArray or any iterable that supports `.source` and `.target` access

# Returns
- `(Rs, ts)`: Tuple containing:
  - `Rs::Vector{SMatrix{3,3,Float64}}`: Rotation matrices (world-to-camera)
  - `ts::Vector{SVector{3,Float64}}`: Translation vectors (world-to-camera)

Returns empty vectors if no valid solution exists.

# Examples
```julia
# Setup camera
sensor = CMOS_SENSORS["Sony"]["IMX174"]
sensor_bounds = Rect(sensor; image_origin=:julia)
f = focal_length(40.0°, sensor; dimension=:horizontal)
pp = center(sensor_bounds)
model = CameraModel(f, sensor.pitch, pp)

# Create correspondences (3D world points → 2D image points)
X3 = [Point3(100.0, 100.0, 0.0), Point3(200.0, 150.0, 0.0), Point3(150.0, 200.0, 0.0)]
u3 = [Point2(640.0, 480.0), Point2(700.0, 500.0), Point2(680.0, 550.0)]
csponds = StructArray{Pt3ToPt2{eltype(X3), eltype(u3)}}((X3, u3))

# Solve P3P
Rs, ts = p3p(model, csponds)

# Create camera extrinsics from solutions
extrinsics = [EuclideanMap(RotMatrix{3}(Rs[i]), ts[i]) for i in 1:length(Rs)]
cameras = Camera.(Ref(model), extrinsics)
```

# See Also
- `p3p(rays, X)`: Core P3P solver that works with rays directly
- `sample_p3p`: Sample valid camera poses by randomly sampling correspondences
"""
function p3p(model::CameraModel, csponds)
    length(csponds) == 3 || error("P3P requires exactly 3 correspondences, got $(length(csponds))")

    # Extract 3D world points and 2D image points from correspondences
    X = csponds.source
    u = csponds.target

    # Backproject 2D points to normalized rays
    rays = backproject.(Ref(model), u)

    # Call original P3P solver
    return p3p(rays, X)
end
