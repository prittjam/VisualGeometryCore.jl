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

# Export transforms and conics functionality
export HomRotMat, HomTransMat, HomScaleIsoMat, HomScaleAnisoMat, EuclideanMat, AffineMat
export to_homogeneous, to_euclidean, materialize, result_type
export HomogeneousConic, Ellipse
export push_conic, gradient

include("utils.jl")        # Units, Size2, geometry utilities
include("blobs.jl")        # Blob types and operations
include("types.jl")        # Other types and StructTypes
include("transforms.jl")   # Homogeneous 2D transforms
include("conics.jl")       # Conics and ellipses
include("plotting.jl")     # Plotting functionality for blobs and patterns
include("api.jl")

function __init__()
    Unitful.register(@__MODULE__)
end

end # module VisualGeometryCore
