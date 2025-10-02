module VisualGeometryCore

using GeometryBasics
using GeometryBasics: Point2, Rect, Vec2, HyperRectangle

# Units and physical quantities
using Unitful
using Unitful: m, mm, cm, inch, @refunit, @unit, @dimension, uconvert, unit, 𝐋, Quantity, μm, rad, °, dimension

# Data structures and arrays
using LinearAlgebra
using CoordinateTransformations, Rotations, StaticArrays
using JSON3, StructTypes

# Functional updates
using Accessors
using ConstructionBase

# Plotting functionality
using Makie
using Makie: campixel!
import Makie.SpecApi as Spec
using Colors
using Colors: Colorant

# Define PlottableImage struct here to avoid precompilation issues
struct PlottableImage
    data::Any
    interpolate::Bool
end
PlottableImage(data; interpolate=false) = PlottableImage(data, interpolate)

include("utils.jl")        # Units, Size2, geometry utilities
include("blobs.jl")        # Blob types and operations
include("types.jl")        # Other types and StructTypes
include("plotting.jl")     # Plotting functionality for blobs and patterns
include("api.jl")

function __init__()
    Unitful.register(@__MODULE__)
end

end # module VisualGeometryCore
