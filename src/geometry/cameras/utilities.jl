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
    lookat(eye::StaticVector{3, <:Real}, target::StaticVector{3, <:Real}, up::StaticVector{3, <:Real})

Create a camera extrinsics transform that looks from `eye` position toward `target`.

Uses **Computer Vision convention**:
- +Z axis points FORWARD (toward target)
- +X axis points RIGHT
- +Y axis points DOWN
- Camera -Y aligns with world up direction

# Arguments
- `eye::StaticVector{3}`: Camera position in world coordinates
- `target::StaticVector{3}`: Point to look at in world coordinates
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
