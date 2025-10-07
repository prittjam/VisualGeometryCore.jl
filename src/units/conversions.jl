# =============================================================================
# Unit Conversion Functions - Atomic operations on Quantities
# =============================================================================

"""
    to_logical_units(q::Quantity{<:Real, 𝐋}, render_density::LogicalDensity)

Convert a length quantity to logical units (pd or px) using the specified render density.

# Arguments
- `q::Quantity`: Physical length quantity (e.g., 10.0mm, 2.5inch)
- `render_density::LogicalDensity`: Render density (e.g., 300dpi for print, 96dpi for screens)

# Returns
- Quantity with logical units (pd or px) matching the density's numerator

# Example
```julia
to_logical_units(10.0mm, 300dpi)  # Returns value in pd
to_logical_units(10.0mm, 300px/inch)  # Returns value in px
```
"""
function to_logical_units(q::Quantity{<:Real, Unitful.𝐋}, render_density::LogicalDensity)
    # Multiplication by density (𝐍 𝐋^-1) converts physical (𝐋) to logical (𝐍)
    scaled = q * render_density

    # Determine target logical unit (pd or px) from density's unit type parameters
    unit_tuple = typeof(unit(render_density)).parameters[1]

    # Look for logical unit (𝐍 dimension) in the tuple
    target_unit = pd  # default to pd
    for u in unit_tuple
        if dimension(u) == 𝐍
            if u isa Unitful.Unit{:Pixel}
                target_unit = px
            elseif u isa Unitful.Unit{:Dot}
                target_unit = pd
            end
            break
        end
    end

    return uconvert(target_unit, scaled)
end

"""
    to_physical_units(q::Quantity{<:Real, 𝐍}, render_density::LogicalDensity)

Convert a logical quantity (pd or px) to physical length units using the specified render density.

# Arguments
- `q::Quantity`: Logical quantity (e.g., 300.0pd, 100.0px)
- `render_density::LogicalDensity`: Render density (e.g., 300dpi for print, 96dpi for screens)

# Returns
- Quantity with length units matching the density's denominator (e.g., inch for dpi, mm for pd/mm)

# Example
```julia
to_physical_units(300.0pd, 300dpi)  # Returns value in inches
to_physical_units(300.0pd, 300pd/mm)  # Returns value in mm
```
"""
function to_physical_units(q::Quantity{<:Real, 𝐍}, render_density::LogicalDensity)
    # Division by density (𝐍 𝐋^-1) converts logical (𝐍) to physical (𝐋)
    scaled = q / render_density

    # Determine target length unit from density's unit type parameters
    unit_tuple = typeof(unit(render_density)).parameters[1]

    # Special case: dpi is a named compound unit (pd/inch)
    target_unit = if any(u -> u isa Unitful.Unit{:DotsPerInch}, unit_tuple)
        inch
    else
        # Look for inverted length unit (𝐋^-1 dimension) in the tuple
        found_unit = mm  # default
        for u in unit_tuple
            if dimension(u) == Unitful.𝐋^-1
                # Extract the base unit (e.g., mm^-1 → mm)
                unit_str = string(u)
                unit_name = Symbol(unit_str[1:end-3])  # Remove "^-1"
                found_unit = getfield(Unitful, unit_name)
                break
            elseif dimension(u) == Unitful.𝐋
                found_unit = u
                break
            end
        end
        found_unit
    end

    return uconvert(target_unit, scaled)
end

# Type conversion utilities for working with Unitful quantities
integer(::Type{Quantity{T,D,U}}) where {T,D,U} = Quantity{Int, D, U}
integer(::Type{<:AbstractArray{<:Quantity{T,D,U}}}) where {T,D,U} = Quantity{Int, D, U}

Unitful.float(::Type{<:AbstractArray{<:Quantity{T,D,U}}}) where {T,D,U} = Quantity{Float64, D, U}

# Quantity rounding/snap helpers
const CLEAN_TOL = 1e-9  # tolerance for snapping near integers/zero in JSON and helpers

snap(x; tol=CLEAN_TOL) = (isfinite(x) && abs(x - round(x)) ≤ tol) ? round(x) : (abs(x) ≤ tol ? zero(x) : x)
snapq(q; tol=CLEAN_TOL) = snap(ustrip(q); tol=tol) * unit(q)
roundq(q; digits::Integer=6) = round(ustrip(q); digits=digits) * unit(q)