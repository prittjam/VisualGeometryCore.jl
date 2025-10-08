# =============================================================================
# General Utilities and Helper Functions
# =============================================================================

# =============================================================================
# JSON3 and StructTypes Support
# =============================================================================

StructTypes.StructType(::Type{<:Unitful.Quantity}) = StructTypes.CustomStruct()

StructTypes.lower(q::Unitful.Quantity) = (
    value = ustrip(q),
    unit = string(unit(q)),
)

function StructTypes.construct(::Type{<:Unitful.Quantity}, obj)
    # Parse unit; trust the numeric value as saved (JSON provides Int for integers, Float64 otherwise)
    u = Unitful.uparse(obj["unit"]; unit_context=@__MODULE__)
    v = obj["value"]
    return v * u
end

# ScalarOrQuantity is defined in types.jl

# JSON3 serialization support for Size2
# Use CustomStruct to handle the FieldVector serialization properly
StructTypes.StructType(::Type{<:Size2}) = StructTypes.CustomStruct()

# Custom serialization - convert to named tuple
StructTypes.lower(s::Size2) = (width = s.width, height = s.height)

# Custom deserialization - reconstruct Size2 with proper Unitful handling
function StructTypes.construct(::Type{Size2}, x)
    get_field = x isa Dict ? (key) -> x[key] : (key) -> getproperty(x, key)

    # Reconstruct Unitful quantities if needed
    width_data = get_field("width")
    height_data = get_field("height")
    
    width = if width_data isa Dict && haskey(width_data, "value") && haskey(width_data, "unit")
        # Use proper Unitful deserialization to preserve numeric types
        StructTypes.construct(ScalarOrQuantity, width_data)
    else
        width_data
    end
    
    height = if height_data isa Dict && haskey(height_data, "value") && haskey(height_data, "unit")
        # Use proper Unitful deserialization to preserve numeric types
        StructTypes.construct(Unitful.Quantity, height_data)
    else
        height_data
    end
    
    return Size2(width=width, height=height)
end

function StructTypes.construct(::Type{Size2{T}}, x) where T
    return StructTypes.construct(Size2, x)
end

# =============================================================================
# Geometry Utilities
# =============================================================================

# ustrip methods for HyperRectangle (Rect)
"""
    Unitful.ustrip(r::HyperRectangle)

Remove units from HyperRectangle, returning raw numbers.
"""
Unitful.ustrip(r::HyperRectangle) = HyperRectangle(ustrip.(r.origin), ustrip.(r.widths))

"""
    Unitful.ustrip(u::Unitful.Units, r::HyperRectangle)

Remove units from HyperRectangle, converting to specified unit first.
"""
Unitful.ustrip(u::Unitful.Units, r::HyperRectangle) = HyperRectangle(ustrip.(u, r.origin), ustrip.(u, r.widths))

# Round method for HyperRectangle (which is what Rect actually is)
"""
    Base.round(::Type{T}, r::HyperRectangle)

Round HyperRectangle to a specific type.
"""
function Base.round(::Type{T}, r::HyperRectangle{N,S}) where {T,S,N}
    # Use broadcasting to round each component of the origin and widths
    rounded_origin = round.(T, r.origin)
    rounded_widths = round.(T, r.widths)
    
    return HyperRectangle{N,T}(rounded_origin, rounded_widths)
end

# Size2 constructor from Rect using widths field
"""
    Size2(r::Rect{N,T}) where {N,T}

Create a Size2 from a Rect using its widths field.
"""
Size2(r::Rect) = Size2(width = r.widths[1], height = r.widths[2])

"""
    Rect(s::Size2{T}) where T

Create a Rect from a Size2 with origin at (0,0) and widths filled from the size dimensions.
"""
Rect(s::Size2{T}) where T = Rect(Point2{T}(zero(T), zero(T)), Vec2{T}(s.width, s.height))

Base.CartesianIndices(bounds::Tuple{Point2{Int}, Point2{Int}}) =
    return CartesianIndices((bounds[1][2]:bounds[2][2], bounds[1][1]:bounds[2][1]))

Base.CartesianIndices(bounds::Tuple{Point2{<:LogicalCount}, Point2{<:LogicalCount}}) = CartesianIndices(ustrip.(bounds))

# Generic constructor for any StaticVector (handles both Point2 and Vec2)
Base.CartesianIndices(bounds::Tuple{StaticVector{2,Int}, StaticVector{2,Int}}) =
    return CartesianIndices((bounds[1][2]:bounds[2][2], bounds[1][1]:bounds[2][1]))

Base.CartesianIndices(bounds::Tuple{StaticVector{2,<:LogicalCount}, StaticVector{2,<:LogicalCount}}) = CartesianIndices(ustrip.(bounds))

"""
    CartesianIndices(r::Rect)

Create CartesianIndices from a Rect.
"""
Base.CartesianIndices(r::Rect{Integer}) =
    CartesianIndices((r.origin[2]:r.origin[2]+r.widths[2]-1, r.origin[1]:r.origin[1]+r.widths[1]-1))

"""
    Rect(indices::CartesianIndices)

Create a Rect from CartesianIndices.
"""
function Rect(indices::CartesianIndices{2})
    ranges = indices.indices
    y_range, x_range = ranges  # CartesianIndices uses (row, col) = (y, x) ordering
    origin = Point2(first(x_range), first(y_range))
    widths = Vec2(length(x_range), length(y_range))

    return Rect(origin, widths)
end

Point2i(index::CartesianIndex{2}) = Point2{Int}(index[2], index[1])
Point2(index::CartesianIndex{2}) = Point2i(index)

# Generic StaticVector constructors that work with both Point2 and Vec2
StaticVector{2,Int}(index::CartesianIndex{2}) = Point2{Int}(index[2], index[1])
StaticVector{2,T}(index::CartesianIndex{2}) where T = StaticVector{2,T}(T(index[2]), T(index[1]))

# =============================================================================
# Helper Functions
# =============================================================================

"""
    filter_kwargs(T; kwargs...) -> (allowed, unknown)

Split keyword args into those that match fieldnames of `T` and those that don't.
"""
function filter_kwargs(::Type{T}; kwargs...) where {T}
    fns = Set(fieldnames(T))
    ks  = keys(kwargs)
    keep = Tuple(k for k in ks if k in fns)
    drop = Tuple(k for k in ks if k ∉ fns)
    
    # Create NamedTuples with only the relevant keys and values
    keep_values = Tuple(kwargs[k] for k in keep)
    drop_values = Tuple(kwargs[k] for k in drop)
    
    return NamedTuple{keep}(keep_values), NamedTuple{drop}(drop_values)
end

"""
    validate_dir(dir::AbstractString; create::Bool=false)

Validate that a directory exists, optionally creating it if it doesn't.

# Arguments
- `dir`: Path to the directory to validate
- `create=false`: If true, create the directory if it doesn't exist

# Returns
- The validated directory path

# Throws
- `ArgumentError` if the directory doesn't exist and `create=false`
"""
function validate_dir(dir::AbstractString; create::Bool=false)
    if create && !isdir(dir)
        mkpath(dir)
    end
    if !isdir(dir)
        throw(ArgumentError("Directory does not exist: $dir"))
    end
    return dir
end

# =============================================================================
# UID Generation Constants
# =============================================================================

# Choose one:
const ALPHABET_BASE64 = collect("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/")
const ALPHABET_BASE58 = collect("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")