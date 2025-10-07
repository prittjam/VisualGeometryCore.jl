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
const LogicalDensity{T<:Real} = Quantity{T, 𝐍/𝐋, <:Unitful.Units}
const LogicalWidth{T<:Real} = Quantity{T, 𝐍, <:Unitful.Units}
const LogicalCount{T<:Integer} = Quantity{T, 𝐍, <:Unitful.Units}
const PixelCount{T<:Integer} = Quantity{T, 𝐍, typeof(px)}

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
end