module VisualGeometryCore

# Geometry and coordinate systems
using GeometryBasics
using GeometryBasics: Point2, Rect, Vec2, HyperRectangle, Circle, Point2f
using StaticArrays
using LinearAlgebra
using CoordinateTransformations
using Rotations

# Units and physical quantities
using Unitful
using Unitful: m, mm, cm, inch, @refunit, @unit, @dimension, uconvert, unit, 𝐋, Quantity, μm, rad, °, dimension

# Data structures and serialization
using JSON3, StructTypes

# Functional updates
using Accessors
using ConstructionBase

# Plotting functionality
using Makie
using Makie: campixel!, Fixed
import Makie.SpecApi as Spec
using Colors
using Colors: Colorant
using GLMakie

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
export ScaleLevel, ScaleSpace
export HessianScaleSpace, LaplacianScaleSpace
export populate_scale_space!, populate_hessian_scale_space!, populate_laplacian_scale_space!
export get_level, get_scale_level

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

# Include plotting (depends on geometry)
include("plotting/specs.jl")
include("plotting/recipes.jl")

# Include API
include("api.jl")

function __init__()
    Unitful.register(@__MODULE__)
end

end # module VisualGeometryCore
