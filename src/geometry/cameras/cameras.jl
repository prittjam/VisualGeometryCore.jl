"""
    Cameras

Camera system submodule for VisualGeometryCore.

Provides complete camera modeling including:
- **Sensors**: Physical CMOS sensor specifications
- **Intrinsics**: Calibration matrices and intrinsic parameters (physical/logical)
- **Projections**: Projection models (pinhole, fisheye, orthographic)
- **Camera Models**: Composable camera models (intrinsics + projection)
- **Cameras**: Complete cameras in 3D space (model + pose)
- **Stereo**: Stereo camera rigs and epipolar geometry
- **Utilities**: Helper functions (focal_length, lookat, P3P solvers)

# Main Types
- `Camera`, `PhysicalCamera`, `LogicalCamera`
- `CameraModel`
- `PhysicalIntrinsics`, `LogicalIntrinsics`
- `PinholeProjection`, `FisheyeProjection`, `OrthographicProjection`
- `Sensor`, `CMOS_SENSORS`
- `StereoRig`

# Main Functions
- `project`, `backproject`, `unproject`
- `focal_length`, `lookat`, `pose`
- `p3p`, `sample_p3p`
- `default_frustum_depth`

# Example
```julia
using VisualGeometryCore
using VisualGeometryCore.Cameras

sensor = CMOS_SENSORS["Sony"]["IMX174"]
f = focal_length(40.0°, sensor)
model = CameraModel(f, sensor.pitch, center(Rect(sensor)))
camera = Camera(model, lookat(...))
```
"""
module Cameras

# Import from parent module
using ..VisualGeometryCore: GeometryBasics, Point2, Point2i, Point3, Circle, Vec2, center
import ..VisualGeometryCore: Rect  # Import (not using) to extend the constructor
using ..VisualGeometryCore: StaticArrays, SVector, SMatrix, StaticVector
using ..VisualGeometryCore: LinearAlgebra, CoordinateTransformations, Rotations, RotMatrix
using ..VisualGeometryCore: normalize, cross  # From LinearAlgebra
using ..VisualGeometryCore: AffineMap  # From CoordinateTransformations
using ..VisualGeometryCore: Unitful, uconvert, ustrip, mm, μm, inch, rad, °, 𝐋, Quantity
using ..VisualGeometryCore: Random, randperm
using ..VisualGeometryCore: StructArrays
using ..VisualGeometryCore: ConstructionBase
using ..VisualGeometryCore: ImmutableDict
using ..VisualGeometryCore: IntervalSets, ClosedInterval, leftendpoint, rightendpoint

# Import types and macros defined in core and geometry
using ..VisualGeometryCore: Size2, Len, Met, LogicalPitch, PixelWidth, PixelCount, LogicalDensity, dpi, px, pd
using ..VisualGeometryCore: Rad, Deg  # Rad/Deg type aliases for angles
using ..VisualGeometryCore: EuclideanMap
using ..VisualGeometryCore: @smatrix_wrapper
using ..VisualGeometryCore: PlanarHomographyMat  # Homography matrix type (for constructors)
using ..VisualGeometryCore: image_origin_offset  # Coordinate convention conversion utility
import ..VisualGeometryCore: PerspectiveMap  # From CoordinateTransformations (imported to extend)

# Import correspondences
using ..VisualGeometryCore: Cspond, Pt3ToPt2

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

# ============================================================================
# Camera Type Aliases
# ============================================================================

"""
    PhysicalCamera

Type alias for cameras with physical intrinsics (focal length in mm, sensor in μm/px).

These cameras have a natural physical scale, so operations like frustum visualization
can use the focal length directly as a depth scale.
"""
const PhysicalCamera = Camera{<:CameraModel{<:PhysicalIntrinsics}}

"""
    LogicalCamera

Type alias for cameras with logical intrinsics (focal length in pixels).

These cameras operate in dimensionless pixel space, so depth-related operations
require explicit scale parameters or sensible defaults based on image dimensions.
"""
const LogicalCamera = Camera{<:CameraModel{<:LogicalIntrinsics}}

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

"""
    default_frustum_depth(camera::PhysicalCamera) -> Float64

Get default frustum depth for visualization of physical cameras.

For PhysicalCamera, returns the focal length (in mm) as a natural depth scale
since the camera operates in physical coordinates.

# Returns
- `Float64`: Focal length in mm (unitless)

# Example
```julia
sensor = CMOS_SENSORS["Sony"]["IMX174"]
f = focal_length(40.0°, sensor; dimension=:horizontal)
model = CameraModel(f, sensor.pitch, center(Rect(sensor)))
camera = Camera(model, extrinsics)
depth = default_frustum_depth(camera)  # Returns f in mm
```
"""
default_frustum_depth(camera::PhysicalCamera) = ustrip(focal_length(camera))

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
project(camera::Camera, X::AbstractVector) = camera.transform(X)

# ============================================================================
# Camera Property Accessors
# ============================================================================

function Base.getproperty(c::Camera, s::Symbol)
    if s === :orientation
        return pose(c).R
    elseif s === :eye_position
        return pose(c).t
    elseif s === :forward
        return RotMatrix(pose(c).R)[:,3]
    elseif s === :up
        return -RotMatrix(pose(c).R)[:,2]
    elseif s === :right
        return -RotMatrix(pose(c).R)[:,1]
    else
        return getfield(c, s)
    end
end

# Include remaining camera system components
include("stereo.jl")
include("utilities.jl")
include("random.jl")

# Include planar homography (depends on Camera)
include("../homography.jl")

# ==============================================================================
# Exports
# ==============================================================================

# Sensor types and constants
export Sensor, CMOS_SENSORS, INTRINSICS_COORDINATE_OFFSET

# Intrinsics types and constructors
export AbstractIntrinsics, LogicalIntrinsics, PhysicalIntrinsics
export CameraCalibrationMatrix

# Projection model types
export AbstractProjectionModel, PinholeProjection, FisheyeProjection, OrthographicProjection

# Camera model and camera types
export CameraModel
export Camera, PhysicalCamera, LogicalCamera

# Stereo types
export StereoRig

# Camera construction and utilities
export focal_length, lookat, pose, default_frustum_depth
export pixel_centers

# Projection functions
export project, backproject, unproject

# Pose estimation
export p3p, sample_p3p

# Homography and image warping
export ProjectiveMap, ImageWarp
export PlanarHomographyMat  # Constructor for PlanarHomographyMat (type defined in transforms/homogeneous.jl)

end # module Cameras
