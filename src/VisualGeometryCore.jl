module VisualGeometryCore

# Geometry and coordinate systems
using GeometryBasics: GeometryBasics, Vec2, HyperRectangle, Point2f
import GeometryBasics: Point2, Point2i, Rect, Circle
using StaticArrays
using LinearAlgebra
using CoordinateTransformations
using Rotations

# Units and physical quantities
using Unitful: Unitful, m, mm, cm, inch, ft, @refunit, @unit, @dimension, uconvert, unit, 𝐋, Quantity, μm, rad, °, dimension, ustrip

# Data structures and serialization
using JSON3, StructTypes
using Base: ImmutableDict

# Functional updates
using Accessors
using ConstructionBase

# Plotting functionality (load backend first for interactive plotting)
using GLMakie
using Makie: Makie, campixel!
import Makie: Fixed as MakieFixed
using Colors: Colors, Colorant, Gray
using FixedPointNumbers: FixedPointNumbers, N0f8, N0f16

# Scale space and image processing functionality
using ImageFiltering: ImageFiltering, Kernel, imfilter, centered, Fill, imfilter!, kernelfactors
using ImageTransformations: imresize, warp
using ImageCore: channelview
using Interpolations
using StructArrays
using FileIO: save
using Transducers

# Export geometry basics
export Point2, Rect, Vec2, HyperRectangle, Circle, Point2f
export cartesian_ranges

# Export transforms and conics functionality
export HomRotMat, HomTransMat, HomScaleIsoMat, HomScaleAnisoMat, EuclideanMat, SimilarityMat, AffineMat, PlanarHomographyMat
export HomEllipseMat, HomCircleMat
export EuclideanMap
export to_homogeneous, to_affine, to_euclidean, result_type
export Ellipse
export gradient
export ConicTrait, CircleTrait, EllipseTrait, conic_trait

# Export blob filtering functions
export light_blobs, dark_blobs

# Export camera models and sensors
export Sensor, CMOS_SENSORS
export AbstractIntrinsics
export Camera, StereoRig, pose, lookat, epipolarmap
export CameraCalibrationMatrix  # 3x3 calibration matrix K
export focal_length, sensor_size, pixel_density, aspect_ratio
export p3p  # P3P solver for camera pose estimation
export HomographyTransform  # Homography transform for warping

# Export composable camera model system (using CoordinateTransformations)
export LogicalIntrinsics, PhysicalIntrinsics
export AbstractProjectionModel, PinholeProjection, FisheyeProjection, OrthographicProjection
export CameraModel
export project, backproject, unproject

# Export plotting module
export Spec  # Export the Spec module for import VisualGeometryCore.Spec

# Export scale space functionality
export AbstractScaleSpace, ScaleLevel, ScaleSpace, ScaleSpaceResponse
export ScaleOctave, ScaleLevelView

export GaussianImage, HessianImages, LaplacianImage
export Size2
export Gray, N0f8, N0f16

# Export local features (kernels and derivatives)
export DERIVATIVE_KERNELS, DERIVATIVE_KERNELS_3D
export hessian_determinant_response, laplacian_response
export detect_features  # Main public API - returns IsoBlobDetection
export detect_extrema   # Deprecated - for backward compatibility
export Extremum3D, find_extrema_3d, refine_extremum_3d, refine_extrema  # Low-level API
export apply!  # Apply transform to ScaleSpaceResponse

# Export image processing utilities
export vlfeat_upsample

# Export type utilities
export neltype



# Core
include("core/units.jl")
include("core/types.jl")
include("core/utils.jl")

# Geometry
include("geometry/transforms.jl")
include("geometry/conversions.jl")
include("geometry/blobs.jl")         # Load blobs before primitives (AbstractBlob needed)
include("geometry/primitives.jl")    # primitives uses AbstractBlob
include("geometry/solvers.jl")       # P3P and other geometric solvers
include("geometry/cameras/cameras.jl")  # Camera system (includes all camera submodules)
include("geometry/homography.jl")  # Homography for planar scenes

# Feature Detection
include("feature/scalespace.jl")
include("feature/responses.jl")  # Must come before kernels.jl (defines ScaleSpaceResponse)
include("feature/kernels.jl")
include("feature/extrema.jl")
include("feature/detection.jl")
include("feature/io.jl")

# Visualization
include("draw/spec.jl")
include("draw/recipes.jl")

function __init__()
    Unitful.register(@__MODULE__)
    GLMakie.activate!()  # Set GLMakie as the active Makie backend
end

end # module VisualGeometryCore
