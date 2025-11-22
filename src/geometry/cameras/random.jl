# =============================================================================
# Random Sampling Functions for Camera Geometry
# =============================================================================

"""
    sample_p3p(model, X, sensor_bounds; max_retries=1000)
    sample_p3p(rng, model, X, sensor_bounds; max_retries=1000)

Sample a random P3P configuration by selecting 3 random 3D points and 3 random 2D image points.

Tries to find a valid P3P solution by randomly sampling correspondences until a
geometrically valid configuration is found (or max_retries is reached).

# Arguments
- `rng=Random.default_rng()`: Random number generator (optional first positional argument)
- `model::CameraModel`: Camera model for backprojection
- `X::Vector{<:StaticVector{3,Float64}}`: 3D point correspondences (at least 3 points)
- `sensor_bounds::Rect`: Sensor bounds for sampling image points (in pixels)

# Keyword Arguments
- `max_retries=1000`: Maximum number of sampling attempts

# Returns
- `(extrinsics, csponds)`: Tuple containing:
  - `extrinsics::Vector{EuclideanMap}`: Camera extrinsics (world-to-camera transforms) for valid P3P solutions
  - `csponds::StructArray{Cspond}`: 3D-to-2D point correspondences (3D world point → 2D image point)

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
extrinsics, csponds = sample_p3p(model, X, ustrip(sensor_bounds))

# With explicit RNG for reproducibility
rng = Random.MersenneTwister(12345)
extrinsics, csponds = sample_p3p(rng, model, X, ustrip(sensor_bounds))

# Access correspondence data (StructArray interface)
X_used = csponds.source  # All 3D points
u_used = csponds.target  # All 2D image points
# Or index individual correspondences: csponds[1].source, csponds[1].target

# Use best extrinsics (e.g., based on reprojection error)
cameras = Camera.(Ref(model), extrinsics)
best_camera = cameras[1]  # Or select based on reprojection error
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

        # Create correspondences as StructArray directly from field arrays
        # Use Pt3ToPt2 type alias for type stability (3D world points → 2D image points)
        csponds = StructArrays.StructArray{Pt3ToPt2{eltype(X3), eltype(u3)}}((X3, u3))

        # Try P3P using the correspondence-based wrapper
        try
            Rs, ts = p3p(model, csponds)

            if length(Rs) > 0
                # Found valid solution - create EuclideanMaps (camera extrinsics) from rotation matrices and translations
                extrinsics = [EuclideanMap(Rotations.RotMatrix{3,Float64}(Rs[i]), ts[i]) for i in 1:length(Rs)]
                return (extrinsics, csponds)
            end
        catch e
            # P3P can fail with DomainError for invalid configurations
            continue
        end
    end

    error("Failed to find valid P3P configuration after $max_retries attempts")
end
