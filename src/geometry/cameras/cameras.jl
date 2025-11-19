# ==============================================================================
# Camera System
# ==============================================================================
# Modular camera model system for VisualGeometryCore
#
# This file orchestrates the camera system by including specialized submodules:
# - sensors.jl:         Physical sensor specifications (CMOS_SENSORS database)
# - intrinsics.jl:      Camera calibration matrices and intrinsics models
# - projections.jl:     Projection models (pinhole, fisheye, orthographic)
# - camera_models.jl:   Composable camera model (intrinsics + projection)
# - stereo.jl:          Stereo camera rigs and epipolar geometry
# - utilities.jl:       Helper functions (focal_length, lookat, etc.)
# ==============================================================================

# Include sensor specifications
include("sensors.jl")

# Include modular camera components
include("intrinsics.jl")
include("projections.jl")
include("camera_models.jl")

# ==============================================================================
# Camera - Complete camera in 3D space (model + pose)
# ==============================================================================

"""
    Camera{M <: CameraModel, Rt <: EuclideanMap, T}

A camera in 3D space, consisting of a camera model (intrinsics + projection) and extrinsics (pose).

The transform field contains the composed projection: `model.transform ∘ extrinsics`,
built at construction time for efficient repeated projection operations.

# Fields
- `model::CameraModel`: Camera model with cached intrinsics and projection transform
- `extrinsics::EuclideanMap`: Camera extrinsics (world-to-camera transform)
- `transform::T`: Cached composed transform (World → Pixels)

# Examples
```julia
# Create camera with logical intrinsics
model = CameraModel(800.0px, [320.0px, 240.0px])
extrinsics = EuclideanMap(RotMatrix(I), [0.0, 0.0, 0.0])
camera = Camera(model, extrinsics)

# Create camera with physical intrinsics
model = CameraModel(16.0mm, Size2(width=5.86μm/px, height=5.86μm/px), [320.0px, 240.0px])
camera = Camera(model, extrinsics)
```
"""
struct Camera{M <: CameraModel, Rt <: EuclideanMap, T}
    model::M
    extrinsics::Rt
    transform::T  # Composed: model.transform ∘ extrinsics
end

"""
    Camera(model::CameraModel, extrinsics::EuclideanMap)

Construct a Camera from model and extrinsics.

Composes and caches the projection transform at construction time.
"""
function Camera(model::M, extrinsics::Rt) where {M<:CameraModel, Rt<:EuclideanMap}
    # Compose: model.transform ∘ extrinsics
    # (extrinsics is applied first, then model.transform)
    transform = model.transform ∘ extrinsics
    return Camera(model, extrinsics, transform)
end

# For non-EuclideanMap extrinsics, convert first
Camera(model, extrinsics) = Camera(model, EuclideanMap(extrinsics))

# Method-based accessors
pose(c::Camera) = inv(c.extrinsics)

"""
    focal_length(camera::Camera)

Get focal length from camera's intrinsics.

Returns Float64 (in mm) for PhysicalIntrinsics, or (fx, fy) tuple (in pixels) for LogicalIntrinsics.

# Examples
```julia
# Physical camera
model = CameraModel(16.0mm, Size2(width=5.86μm/px, height=5.86μm/px), [320.0px, 240.0px])
camera = Camera(model, extrinsics)
f = focal_length(camera)  # Returns 16.0 (Float64)

# Logical camera
K = CameraCalibrationMatrix(800.0px, [320.0px, 240.0px])
model = CameraModel(LogicalIntrinsics(K), PinholeProjection())
camera = Camera(model, extrinsics)
fx, fy = focal_length(camera)  # Returns (800.0, 800.0)
```
"""
focal_length(camera::Camera) = focal_length(camera.model.intrinsics)

# ============================================================================
# Camera Projection (World → Pixel)
# ============================================================================

"""
    project(camera::Camera, X_world::AbstractVector) -> Point2

Project 3D point from world coordinates to pixel coordinates.

Uses the cached composed transform for efficient projection.

# Arguments
- `camera::Camera`: Camera with cached composed transform
- `X_world::AbstractVector`: 3D point in world coordinates (with units)

# Returns
- Point2 with pixel coordinates (px units)

# Example
```julia
model = CameraModel(16.0mm, Size2(width=5.86μm/px, height=5.86μm/px), [320.0px, 240.0px])
extrinsics = lookat(SVector(0.0, 0.0, 10.0), SVector(0.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0))
camera = Camera(model, extrinsics)
X_world = SVector(1.0mm, 2.0mm, 3.0mm)
u = project(camera, X_world)
```
"""
function project(camera::Camera, X_world::AbstractVector)
    # Apply cached composed transform
    # Note: EuclideanMap operates on unitless values, so we strip/restore units at the boundary
    X_world_unitless = ustrip.(X_world)
    return camera.transform(X_world_unitless)
end

# ============================================================================
# Camera Property Accessors
# ============================================================================

function Base.getproperty(c::Camera, s::Symbol)
    if s === :orientation
        return pose(c).R
    elseif s === :eye_position
        return Meshes.Point(Tuple(pose(c).t))
    elseif s === :forward
        return RotMatrix(pose(c).R)[:,3]
    elseif s === :up
        return -RotMatrix(pose(c).R)[:,2]
    else
        return getfield(c, s)
    end
end

# Include remaining camera system components
include("stereo.jl")
include("utilities.jl")
