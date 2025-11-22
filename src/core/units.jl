# =============================================================================
# Custom Units and Type Aliases
# =============================================================================

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
const LogicalDensity{T<:Real} = Quantity{T, 𝐍/𝐋, <:Unitful.Units}  # e.g., dpi, ppi, px/mm
const LogicalPitch{T<:Real} = Quantity{T, 𝐋/𝐍, <:Unitful.Units}    # e.g., μm/px, mm/px, μm/pt
const LogicalWidth{T<:Real} = Quantity{T, 𝐍, <:Unitful.Units}
const LogicalCount{T<:Integer} = Quantity{T, 𝐍, <:Unitful.Units}
const PixelCount{T<:Integer} = Quantity{T, 𝐍, typeof(px)}
const PixelWidth{T<:Real} = Quantity{T, 𝐍, typeof(px)}

# Dimensional type aliases (shorthand)
const Len{T} = Quantity{T, 𝐋, <:Unitful.Units}
const Met{T} = Quantity{T, 𝐋, typeof(m)}
const Deg{T} = Quantity{T, Unitful.NoDims, typeof(°)}
const Rad{T} = Quantity{T, Unitful.NoDims, typeof(rad)}

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
end# =============================================================================
# Unit Conversion Functions - Atomic operations on Quantities
# =============================================================================

"""
    uconvert(target_units::Unitful.Units, q::Quantity{<:Real, 𝐋}, render_density::LogicalDensity)

Convert a physical length quantity to logical units using the specified render density.

# Arguments
- `target_units`: Target logical units (e.g., `px`, `pd`)
- `q::Quantity{<:Real, 𝐋}`: Physical length quantity (e.g., `10.0mm`, `2.5inch`)
- `render_density::LogicalDensity`: Render density (e.g., `300dpi`, `96dpi`)

# Returns
- Quantity in target logical units

# Examples
```julia
uconvert(px, 10.0mm, 300dpi)   # Physical to logical (pixels)
uconvert(pd, 1.0inch, 300dpi)  # Physical to logical (points)
```
"""
Unitful.uconvert(target_units::Unitful.Units, q::Quantity{<:Real, 𝐋}, render_density::LogicalDensity) =
    return uconvert(target_units, q * render_density)

"""
    uconvert(target_units::Unitful.Units, q::Quantity{<:Real, 𝐍}, render_density::LogicalDensity)

Convert a logical quantity to physical length units using the specified render density.

# Arguments
- `target_units`: Target physical units (e.g., `mm`, `inch`)
- `q::Quantity{<:Real, 𝐍}`: Logical quantity (e.g., `300.0px`, `100.0pd`)
- `render_density::LogicalDensity`: Render density (e.g., `300dpi`, `96dpi`)

# Returns
- Quantity in target physical units

# Examples
```julia
uconvert(mm, 300.0px, 300dpi)    # Logical to physical (millimeters)
uconvert(inch, 300.0pd, 300dpi)  # Logical to physical (inches)
```
"""
Unitful.uconvert(target_units::Unitful.Units, q::Quantity{<:Real, 𝐍}, render_density::LogicalDensity) =
    return uconvert(target_units, q / render_density)

# =============================================================================
# Generic uconvert for structs with Unitful quantities
# =============================================================================

"""
    uconvert(target_units::Unitful.Units, obj, render_density::LogicalDensity)

Generic unit conversion for structs containing Unitful quantities.

Recursively converts all fields with physical dimensions (𝐋) to logical units (𝐍)
or vice versa, using the specified render density. Non-quantity fields are preserved.

# Arguments
- `target_units`: Target unit type (e.g., `px`, `mm`)
- `obj`: Object to convert (any struct with Unitful quantity fields)
- `render_density`: Render density for conversion (e.g., `300dpi`, `96dpi`)

# Returns
- New object with converted fields

# Examples
```julia
# Convert blob from physical to logical units
blob_mm = IsoBlob(Point2(10.0mm, 20.0mm), 2.0mm)
blob_px = uconvert(px, blob_mm, 300dpi)

# Convert circle from logical to physical units
circle_px = Circle(Point2(100.0px, 200.0px), 50.0px)
circle_mm = uconvert(mm, circle_px, 300dpi)

# Works for any struct with quantity fields
rect_mm = Rect2(Point2(0.0mm, 0.0mm), Point2(100.0mm, 200.0mm))
rect_px = uconvert(px, rect_mm, 300dpi)
```
"""
function Unitful.uconvert(target_units::Unitful.Units, obj, render_density::LogicalDensity)
    # Get all properties as a NamedTuple (type-stable)
    props = ConstructionBase.getproperties(obj)

    # Convert each property
    new_props = map(p -> _convert_field(target_units, p, render_density), props)

    # Reconstruct object with converted properties
    return ConstructionBase.setproperties(obj, new_props)
end

"""
    _convert_field(target_units, field_value, render_density)

Helper function to convert individual fields based on their type.

Handles:
- Physical quantities (𝐋) → converts using render_density
- Logical quantities (𝐍) → converts using render_density
- Vectors/collections → broadcasts conversion
- Nested structs with ConstructionBase support → recursive conversion
- Other values → pass through unchanged
"""
# Quantity with length dimension (𝐋) → convert to logical
_convert_field(::typeof(px), val::Quantity{<:Real, 𝐋}, density::LogicalDensity) =
    uconvert(px, val, density)

# Quantity with logical dimension (𝐍) → convert to physical
_convert_field(::typeof(mm), val::Quantity{<:Real, 𝐍}, density::LogicalDensity) =
    uconvert(mm, val, density)

# Same dimension → strip/add units as needed
_convert_field(target::Unitful.Units, val::Quantity, density::LogicalDensity) =
    uconvert(target, val)

# Vectors/Points → broadcast
_convert_field(target::Unitful.Units, val::AbstractVector, density::LogicalDensity) =
    _convert_field.(Ref(target), val, Ref(density))

# Nested structs with ConstructionBase support → recursive
function _convert_field(target::Unitful.Units, val, density::LogicalDensity)
    if ConstructionBase.constructorof(typeof(val)) !== nothing
        try
            return uconvert(target, val, density)
        catch
            # If conversion fails, return as-is
            return val
        end
    else
        # Non-quantity, non-convertible fields pass through
        return val
    end
end

# =============================================================================
# Generic ustrip for structs with Unitful quantities
# =============================================================================

"""
    ustrip(obj)

Generic unit stripping for structs containing Unitful quantities.

Recursively strips units from all fields, returning a new object with unitless values.

# Examples
```julia
circle_mm = Circle(Point2(10.0mm, 20.0mm), 5.0mm)
circle_unitless = ustrip(circle_mm)  # Circle(Point2(10.0, 20.0), 5.0)

blob_px = IsoBlob(Point2(100.0px, 200.0px), 10.0px)
blob_unitless = ustrip(blob_px)  # IsoBlob(Point2(100.0, 200.0), 10.0)
```
"""
function Unitful.ustrip(obj)
    # Get all properties as a NamedTuple
    props = ConstructionBase.getproperties(obj)

    # Strip units from each property
    new_props = map(_ustrip_field, props)

    # Reconstruct object with unitless properties
    return ConstructionBase.setproperties(obj, new_props)
end

# Helper to strip units from different field types
_ustrip_field(val::Quantity) = ustrip(val)
_ustrip_field(val::AbstractVector) = _ustrip_field.(val)
_ustrip_field(val) = try
    # Try to recursively ustrip if it's a struct
    if ConstructionBase.constructorof(typeof(val)) !== nothing
        ustrip(val)
    else
        val  # Pass through non-quantity, non-struct fields
    end
catch
    val  # If ustrip fails, return as-is
end

# Type conversion utilities for working with Unitful quantities
integer(::Type{Quantity{T,D,U}}) where {T,D,U} = Quantity{Int, D, U}
integer(::Type{<:AbstractArray{<:Quantity{T,D,U}}}) where {T,D,U} = Quantity{Int, D, U}

Unitful.float(::Type{<:AbstractArray{<:Quantity{T,D,U}}}) where {T,D,U} = Quantity{Float64, D, U}

# Quantity rounding/snap helpers
const CLEAN_TOL = 1e-9  # tolerance for snapping near integers/zero in JSON and helpers

snap(x; tol=CLEAN_TOL) = (isfinite(x) && abs(x - round(x)) ≤ tol) ? round(x) : (abs(x) ≤ tol ? zero(x) : x)
snapq(q; tol=CLEAN_TOL) = snap(ustrip(q); tol=tol) * unit(q)
unitful_round(q; digits::Integer=6) = round(ustrip(q); digits=digits) * unit(q)
