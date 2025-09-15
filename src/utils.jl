# =============================================================================
# Utilities - Units, Size2, Geometry, and Helper Functions
# =============================================================================

# =============================================================================
# Custom Units and Type Aliases
# =============================================================================

"""
    SIGMA_CUTOFF

Default multiple of σ used to define an effective blob radius/area.
Used by utilities like `radius` and `area` when not explicitly provided.
"""
# Constants used across the package
const SIGMA_CUTOFF = 3.0
const CLEAN_TOL = 1e-9  # tolerance for snapping near integers/zero in JSON and helpers

# Define a dimensionless dimension for pixels/dots
@dimension 𝐍 "𝐍" LogicalUnits   

"""
    pd

Logical dot unit for pattern coordinates (dimension 𝐍). One `px` equals `1pd`.
"""
@refunit pd "pd" Dot 𝐍 false

"""
    px

Logical pixel unit alias (`1px == 1pd`). Useful for expressing image sizes.
"""
@unit px "px" Pixel 1*pd false

"""
    dpi

Dots-per-inch as a logical density unit (𝐍/𝐋). Common values are `150dpi`, `300dpi`.
"""
@unit dpi "dpi" DotsPerInch 1*pd/inch false

"""
    pt

Typography point unit (1/72 inch). Used for precise vector export scaling.
"""
@unit pt "pt" Point (1//72)*inch false

# Type aliases for convenience
const LogicalDensity{T<:Real} = Quantity{T, 𝐍/𝐋, <:Unitful.Units}
const LogicalWidth{T<:Real} = Quantity{T, 𝐍, <:Unitful.Units}
const LogicalCount{T<:Integer} = Quantity{T, 𝐍, <:Unitful.Units}
const PixelCount{T<:Integer} = Quantity{T, 𝐍, typeof(px)}

# Dimensional type aliases (shorthand)
const Len{T} = Quantity{T, 𝐋, <:Unitful.Units}
const Met{T} = Quantity{T, 𝐋, typeof(m)}
const Deg{T} = Quantity{T, Unitful.NoDims, typeof(°)}
const Rad{T} = Quantity{T, Unitful.NoDims, typeof(rad)}

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

# Type conversion utilities for working with Unitful quantities
integer(::Type{Quantity{T,D,U}}) where {T,D,U} = Quantity{Int, D, U}
integer(::Type{<:AbstractArray{<:Quantity{T,D,U}}}) where {T,D,U} = Quantity{Int, D, U}

Unitful.float(::Type{<:AbstractArray{<:Quantity{T,D,U}}}) where {T,D,U} = Quantity{Float64, D, U}

# =============================================================================
# Size2 Type
# =============================================================================

"""
    Size2{T}

Width/height container with unit-carrying elements for image and canvas sizes.

- `width`: columns (x-axis)
- `height`: rows (y-axis)

Constructors:
- `Size2(; width, height)` keyword constructor
- `Size2(matrix)` from a matrix's `(width, height)`
- `Size2(scene::LScene)` from a Makie scene size

Examples:
```julia
Size2(width=800, height=600)
Size2(width=210mm, height=297mm)
Size2(width=800pd, height=600pd)
```
"""
struct Size2{T} <: FieldVector{2, T}
    width::T   # columns, x-axis
    height::T  # rows, y-axis
end

# Private inner constructor
Size2(; width, height) =  Size2(width, height)
Size2(matrix::AbstractMatrix) = Size2(reverse(size(matrix))...)

# Preserve Size2 type under broadcasting operations like ustrip.()
# This follows the same pattern as GeometryBasics.Point
StaticArrays.similar_type(::Type{Size2{T}}, ::Type{T2}, s::StaticArrays.Size{(2,)}) where {T, T2} = Size2{T2}
StaticArrays.similar_type(::Type{Size2{T}}, ::Type{T2}) where {T, T2} = Size2{T2}

# Computed properties for Size2
Base.getproperty(s::Size2, name::Symbol) = begin
    if name === :aspect_ratio
        return s.width / s.height
    elseif name === :area
        return s.width * s.height
    else
        return getfield(s, name)
    end
end

# JSON3 serialization support for Size2
# Use CustomStruct to handle the FieldVector serialization properly
StructTypes.StructType(::Type{<:Size2}) = StructTypes.CustomStruct()

# Custom serialization - convert to named tuple
StructTypes.lower(s::Size2) = (width = s.width, height = s.height)

# Custom deserialization - reconstruct Size2 with proper Unitful handling
function StructTypes.construct(::Type{Size2}, x)
    get_field = x isa Dict ? (key) -> x[key] : (key) -> getproperty(x, key)

    StructTypes.construct(ScalarOrQuantity, width_data)
    StructTypes.construct(ScalarOrQuantity, height_data)
    
    
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

# Quantity rounding/snap helpers
snap(x; tol=CLEAN_TOL) = (isfinite(x) && abs(x - round(x)) ≤ tol) ? round(x) : (abs(x) ≤ tol ? zero(x) : x)
snapq(q; tol=CLEAN_TOL) = snap(ustrip(q); tol=tol) * unit(q)
roundq(q; digits::Integer=6) = round(ustrip(q); digits=digits) * unit(q)

# =============================================================================
# Unit Conversion Functions for IsoBlob
# =============================================================================

"""
    to_logical_units(blob::IsoBlob, render_density::LogicalDensity)

Convert an IsoBlob from physical units to logical units using the specified render density.

# Arguments
- `blob::IsoBlob`: The blob with physical coordinates and scale
- `render_density::LogicalDensity`: Render density (e.g., 300dpi for print, 96dpi for screens)

# Returns
- `IsoBlob`: New blob with logical coordinates (pd units)

# Example
```julia
# Physical blob in millimeters
blob_mm = IsoBlob(Point2(10.0mm, 20.0mm), 2.0mm)

# Convert to logical units at 300 DPI (print) or 96 DPI (screen)
blob_logical = to_logical_units(blob_mm, 300dpi)
```
"""
function to_logical_units(blob::IsoBlob, render_density::LogicalDensity)
    # Convert center coordinates from physical to logical units
    center_logical = uconvert.(pd, blob.center .* render_density)
    
    # Convert σ from physical to logical units
    σ_logical = uconvert(pd, blob.σ * render_density)
    
    return IsoBlob(center_logical, σ_logical)
end

"""
    to_physical_units(blob::IsoBlob, render_density::LogicalDensity)

Convert an IsoBlob from logical units to physical units using the specified render density.

# Arguments
- `blob::IsoBlob`: The blob with logical coordinates and scale (pd units)
- `render_density::LogicalDensity`: Render density (e.g., 300dpi for print, 96dpi for screens)

# Returns
- `IsoBlob`: New blob with physical coordinates (length units like mm, inch)

# Example
```julia
# Logical blob in pixels/dots
blob_logical = IsoBlob(Point2(300.0pd, 600.0pd), 15.0pd)

# Convert to physical units at 300 DPI (print) or 96 DPI (screen)
blob_mm = to_physical_units(blob_logical, 300dpi)
```
"""
function to_physical_units(blob::IsoBlob, render_density::LogicalDensity)
    # Convert center coordinates from logical to physical units
    center_physical = blob.center ./ render_density
    
    # Convert σ from logical to physical units
    σ_physical = blob.σ / render_density
    
    return IsoBlob(center_physical, σ_physical)
end

# =============================================================================
# UID Generation Constants
# =============================================================================

# Choose one:
const ALPHABET_BASE64 = collect("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/")
const ALPHABET_BASE58 = collect("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")
