module VisualGeometryCore

# Geometry and coordinate systems
using GeometryBasics: GeometryBasics, Vec2, HyperRectangle, Point2f
import GeometryBasics: Point2, Point2i, Rect, Circle
using StaticArrays
using LinearAlgebra
using CoordinateTransformations
using Rotations

# Units and physical quantities
using Unitful: Unitful, m, mm, cm, inch, ft, @refunit, @unit, @dimension, uconvert, unit, 𝐋, Quantity, μm, rad, °, dimension

# Data structures and serialization
using JSON3, StructTypes

# Functional updates
using Accessors
using ConstructionBase

# Plotting functionality
using Makie: Makie, campixel!
import Makie.SpecApi as Spec
import Makie: Fixed as MakieFixed
using Colors: Colors, Colorant, Gray
using FixedPointNumbers: FixedPointNumbers, N0f8, N0f16
using GLMakie

# Scale space and image processing functionality
using ImageFiltering: ImageFiltering, Kernel, imfilter, centered, Fill, imfilter!, kernelfactors
using ImageTransformations: imresize
using ImageCore: channelview
using StructArrays
using FileIO: save

# Export geometry basics
export Point2, Rect, Vec2, HyperRectangle, Circle, Point2f
export cartesian_ranges

# Export transforms and conics functionality
export HomRotMat, HomTransMat, HomScaleIsoMat, HomScaleAnisoMat, EuclideanMat, SimilarityMat, AffineMat
export EuclideanMap
export to_homogeneous, to_euclidean, result_type
export HomogeneousConic, Ellipse
export push_conic, gradient, plotellipses

# Export scale space functionality
export AbstractScaleSpace, ScaleLevel, ScaleSpace, ScaleSpaceResponse
export ScaleOctave, ScaleLevelView

export GaussianImage, HessianImages, LaplacianImage
export Size2
export Gray, N0f8, N0f16

# Export local features (kernels and derivatives)
export HESSIAN_KERNELS, DERIVATIVE_KERNELS, DERIVATIVE_KERNELS_3D, LAPLACIAN_KERNEL
export LAPLACIAN_DIRECT, HESSIAN_DIRECT
export laplacian, hessian_determinant
export hessian_determinant_response, laplacian_response
export Extremum3D, detect_extrema, find_extrema_3d, refine_extremum_3d

# Export image processing utilities
export vlfeat_upsample



# Include units first (defines custom units and types)
include("units/types.jl")
include("units/conversions.jl")

# Include utilities (depends on units)
include("utils.jl")

# Include other types (EuclideanMap needed by conversions)
include("types.jl")

# Include geometry (core mathematical operations)
include("geometry/transforms.jl")
include("geometry/conversions.jl")
include("geometry/conics.jl")
include("geometry/blobs.jl")

# Include scale space
include("scalespace.jl")

# Include local features
include("local_features/scalespace_response.jl")
include("local_features/kernels.jl")
include("local_features/extrema.jl")
include("local_features/io.jl")

# Include plotting (depends on geometry)
include("plotting/specs.jl")
include("plotting/recipes.jl")

# Include API
include("api.jl")

function __init__()
    Unitful.register(@__MODULE__)
end

end # module VisualGeometryCore
