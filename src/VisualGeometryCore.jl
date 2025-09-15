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

include("utils.jl")        # Units, Size2, geometry utilities
include("types.jl")        # Structs and StructTypes
include("api.jl")

function __init__()
    Unitful.register(@__MODULE__)
end

end # module VisualGeometryCore
