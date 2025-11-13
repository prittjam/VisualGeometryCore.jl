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
# P3P Pose Sampling
# =============================================================================

"""
    sample_p3p([rng], model::CameraModel, X::Vector{<:StaticVector{3,Float64}}, sensor_bounds::Rect;
               max_retries=1000)

Sample a valid camera pose by randomly sampling image point correspondences and solving P3P.

Randomly samples 3D point indices and their corresponding 2D image projections within
the sensor bounds, then attempts to solve the P3P problem. Repeats until a valid
solution is found or max_retries is reached.

Follows Julia conventions: `rng` is an optional first positional argument (like `rand` and `sample`).

# Arguments
- `rng=Random.default_rng()`: Random number generator (optional first positional argument)
- `model::CameraModel`: Camera model for backprojection
- `X::Vector{<:StaticVector{3,Float64}}`: 3D point correspondences (at least 3 points)
- `sensor_bounds::Rect`: Sensor bounds for sampling image points (in pixels)

# Keyword Arguments
- `max_retries=1000`: Maximum number of sampling attempts

# Returns
- `(Rs, ts, u_sampled, X_sampled)`: Tuple containing:
  - `Rs::Vector{SMatrix{3,3,Float64}}`: Rotation matrices for valid solutions
  - `ts::Vector{Vec{3,Float64}}`: Translation vectors for valid solutions
  - `u_sampled::Vector{Point2{Float64}}`: Sampled image points used
  - `X_sampled::Vector{Point3{Float64}}`: Corresponding 3D points used

Throws an error if no valid configuration is found after max_retries attempts.

# Examples
```julia
# Setup camera
sensor = CMOS_SENSORS["Sony"]["IMX174"]
sensor_bounds = Rect(sensor; image_origin=:julia)
f = focal_length(40.0°, sensor; dimension=:horizontal)
pp = center(sensor_bounds)
model = CameraModel(f, sensor.pitch, pp)

# 3D points (planar points at z=0)
X = [Point3(100.0, 100.0, 0.0), Point3(200.0, 150.0, 0.0), ...]

# With default RNG
Rs, ts, u, X_used = sample_p3p(model, X, ustrip(sensor_bounds))

# With explicit RNG for reproducibility
rng = Random.MersenneTwister(12345)
Rs, ts, u, X_used = sample_p3p(rng, model, X, ustrip(sensor_bounds))

# Use best solution
cameras = Camera.(Ref(model), EuclideanMap.(RotMatrix{3,Float64}.(Rs), ts))
```
"""
# Method with default RNG
function sample_p3p(model::CameraModel,
                    X::Vector{<:StaticVector{3,Float64}},
                    sensor_bounds::Rect;
                    max_retries::Int=1000)
    return sample_p3p(Random.default_rng(), model, X, sensor_bounds; max_retries=max_retries)
end

# Method with explicit RNG (follows Julia convention: rng as first positional argument)
function sample_p3p(rng::Random.AbstractRNG,
                    model::CameraModel,
                    X::Vector{<:StaticVector{3,Float64}},
                    sensor_bounds::Rect;
                    max_retries::Int=1000)

    length(X) >= 3 || error("Need at least 3 point correspondences for P3P")

    for attempt in 1:max_retries
        # Sample 3 random 3D points
        sampled_idx = randperm(rng, length(X))[1:3]
        X3 = X[sampled_idx]

        # Sample 3 random image points in sensor bounds
        u3 = rand(rng, sensor_bounds, 3)

        # Backproject to rays
        rays = backproject.(Ref(model), u3)

        # Try P3P
        try
            Rs, ts = p3p(rays, X3)

            if length(Rs) > 0
                # Found valid solution
                return (Rs, ts, u3, X3)
            end
        catch e
            # P3P can fail with DomainError for invalid configurations
            continue
        end
    end

    error("Failed to find valid P3P configuration after $max_retries attempts")
end
