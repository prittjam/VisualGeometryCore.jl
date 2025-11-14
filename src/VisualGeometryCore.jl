module VisualGeometryCore

# Geometry and coordinate systems
using GeometryBasics: GeometryBasics, Vec2, HyperRectangle, Point2f, Rect2, radius
import GeometryBasics: Point2, Point2i, Rect, Circle
using StaticArrays
using LinearAlgebra
using CoordinateTransformations
using CoordinateTransformations: PerspectiveMap
using Rotations
using Random
import Random: rand

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
import Makie.SpecApi as MakieSpec
using Colors: Colors, Colorant, Gray
using FixedPointNumbers: FixedPointNumbers, N0f8, N0f16, FixedPoint

# Scale space and image processing functionality
using ImageFiltering: ImageFiltering, Kernel, imfilter, centered, Fill, imfilter!, kernelfactors
using ImageTransformations: imresize, warp
using ImageCore: channelview
using Interpolations
using IntervalSets
using IntervalSets: ClosedInterval, leftendpoint, rightendpoint
using StructArrays
using FileIO: save, load
using Transducers

# Export geometry basics
export Point2, Rect, Rect2, Vec2, HyperRectangle, Circle, Point2f
export cartesian_ranges, center, ranges
export coord_map, CANONICAL_SQUARE, UNIT_CIRCLE  # Generic coordinate mapping API
export imgmap  # Image coordinate adaptation for warping
export logpolar_to_cartesian, logpolar_map, logpolar_patch
export ClosedInterval  # Re-export from IntervalSets for convenience

# Export transforms and conics functionality
export HomRotMat, HomTransMat, HomScaleIsoMat, HomScaleAnisoMat, EuclideanMat, SimilarityMat, AffineMat, PlanarHomographyMat
export HomEllipseMat, HomCircleMat
export EuclideanMap
export to_homogeneous, to_affine, to_euclidean, result_type
export Ellipse, is_ellipse
export gradient
export ConicTrait, CircleTrait, EllipseTrait, conic_trait
export dilate  # Dilate circles and ellipses (scale size, keep position)
export intersects  # Check intersection between geometric primitives

# Export blob filtering functions
export light_blobs, dark_blobs

# Export feature polarity types
export FeaturePolarity, PositiveFeature, NegativeFeature, ImageFeature

# Export camera models and sensors
export Sensor, CMOS_SENSORS
export AbstractIntrinsics
export Camera, StereoRig, pose, lookat, epipolarmap
export CameraCalibrationMatrix  # 3x3 calibration matrix K
export focal_length, sensor_size, pixel_density, aspect_ratio, pixel_centers
export p3p, sample_p3p  # P3P solver and pose sampling
export ProjectiveMap  # Projective homography R^2 → P^2 (follows CoordinateTransformations naming)
export ImageWarp  # Image warping with PerspectiveMap ∘ ProjectiveMap composition

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
export px, pd, dpi, pt  # Custom logical units

# Export local features (kernels and derivatives)
export DERIVATIVE_KERNELS, DERIVATIVE_KERNELS_3D
export hessian_determinant_response, laplacian_response
export detect_features  # Main public API - returns IsoBlobDetection
export Extremum3D, find_extrema_3d, refine_extremum_3d, refine_extrema  # Low-level API
export apply!  # Apply transform to ScaleSpaceResponse

# Export correspondence types
export AbstractCspond, Cspond, AttributedCspond, ScoredCspond, Pt2ToPt2, Pt3ToPt2
export BlobToBlob, BlobToPt2, Pt2ToBlob
export CspondSet, AttributedCspondSet
export csponds  # Convenience function to create type-stable StructArray of Cspond

# Export image processing utilities
export vlfeat_upsample

# Export type utilities
export neltype



# Core
include("core/units.jl")
include("core/types.jl")
include("core/utils.jl")

# Geometry
include("geometry/transforms/homogeneous.jl")  # Homogeneous transforms (defines HomEllipseMat) - must come first
include("geometry/blobs.jl")               # Load blobs before primitives (AbstractBlob needed)
include("geometry/primitives/primitives.jl")  # Geometric primitives (uses AbstractBlob and HomEllipseMat)
include("geometry/transforms/conversions.jl")  # Coordinate system conversions (uses primitives)
include("geometry/transforms/coord_maps.jl")  # Coordinate mappings (uses primitives)
include("geometry/transforms/logpolar.jl")  # Log-polar transforms (uses primitives)
include("geometry/solvers.jl")             # P3P and other geometric solvers
include("geometry/cameras/cameras.jl")     # Camera system (includes all camera submodules)
include("geometry/homography.jl")          # Homography for planar scenes

# Feature Detection
include("feature/scalespace.jl")
include("feature/responses.jl")  # Must come before kernels.jl (defines ScaleSpaceResponse)
include("feature/kernels.jl")
include("feature/extrema.jl")
include("feature/detection.jl")
include("feature/correspondences.jl")  # Correspondence types for feature matching
include("feature/io.jl")

# Visualization
include("plotting/spec.jl")
include("plotting/recipes.jl")

function __init__()
    Unitful.register(@__MODULE__)
    GLMakie.activate!()  # Set GLMakie as the active Makie backend
end

end # module VisualGeometryCore
