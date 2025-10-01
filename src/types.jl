
# ========================================================================
# EuclideanMap Type and Operations
# ========================================================================

"""
    EuclideanMap{N,T,R<:Rotation{N,T},V<:SVector{N,T}} <: Transformation

Rigid transform `x ↦ R*x + t` in `N` dims.
"""
struct EuclideanMap{N,T,R<:Rotation{N,T},V<:StaticVector{N,T}} <: CoordinateTransformations.Transformation
    R::R
    t::V
end

# Constructors
EuclideanMap(R::Rotation{N,T}, t::AbstractVector) where {N,T} =
    EuclideanMap{N,T,typeof(R),SVector{N,T}}(R, SVector{N,T}(t))

EuclideanMap(R::Rotation{N,T}, t::NTuple{N,T}) where {N,T} =
    EuclideanMap(R, SVector{N,T}(t))

EuclideanMap(R::MRP{T}, t::AbstractVector{T}) where {T} =
    EuclideanMap{3,T,MRP{T},SVector{3,T}}(R, SVector{3,T}(t))

EuclideanMap(R::MRP{T}, tx::T, ty::T, tz::T) where {T} =
    EuclideanMap(R, SVector{3,T}(tx,ty,tz))

EuclideanMap(E::EuclideanMap) = E

EuclideanMap((R, t)::Tuple{<:Rotation,<:AbstractVector}) = EuclideanMap(R, t)

# Operations
(E::EuclideanMap)(x) = E.R * x + E.t

@inline function Base.inv(E::EuclideanMap{N,T}) where {N,T}
    Rin = inv(E.R)
    EuclideanMap{N,T,typeof(Rin),typeof(E.t)}(Rin, -(Rin * E.t))
end

function Base.:∘(A::EuclideanMap{N,TA}, B::EuclideanMap{N,TB}) where {N,TA,TB}
    T = promote_type(TA, TB)
    R_composed = A.R * B.R
    t_composed = A.R * B.t + A.t
    return EuclideanMap(R_composed, t_composed)
end

# Helper functions with unique name to avoid collisions
extract_field(x::Dict{String,Any}, key::String) = x[key]
extract_field(x::JSON3.Object, key::String) = getproperty(x, Symbol(key))
extract_field(x::NamedTuple, key::String) = getproperty(x, Symbol(key))

const ScalarOrQuantity = Union{Number, Unitful.Quantity}

StructTypes.construct(::Type{ScalarOrQuantity}, data) = 
    data isa Number ? data : StructTypes.construct(Unitful.Quantity, data)

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

# =============================================================================
# Arithmetic Operations for AbstractBlob
# =============================================================================

"""
    +(blob::AbstractBlob, offset)

Translate a blob by adding an offset to its center position.

Uses Accessors.jl for functional updates, allowing translation without
knowing all fields of the blob type.

# Arguments
- `blob::AbstractBlob`: The blob to translate
- `offset`: A tuple or vector representing the translation (x, y)

# Returns
- New blob of the same type with translated center

# Example
```julia
blob = IsoBlob(Point2(10.0mm, 20.0mm), 2.0mm)
translated = blob + (5.0mm, 5.0mm)  # Center at (15.0mm, 25.0mm)
```
"""
Base.:+(blob::AbstractBlob, offset) = @set blob.center = blob.center .+ offset

"""
    -(blob::AbstractBlob, offset)

Translate a blob by subtracting an offset from its center position.

Uses Accessors.jl for functional updates, allowing translation without
knowing all fields of the blob type.

# Arguments
- `blob::AbstractBlob`: The blob to translate
- `offset`: A tuple or vector representing the translation (x, y)

# Returns
- New blob of the same type with translated center

# Example
```julia
blob = IsoBlob(Point2(10.0mm, 20.0mm), 2.0mm)
translated = blob - (5.0mm, 5.0mm)  # Center at (5.0mm, 15.0mm)
```
"""
Base.:-(blob::AbstractBlob, offset) = @set blob.center = blob.center .- offset

"""
    *(blob::AbstractBlob, scale)

Scale a blob's center coordinates and σ by a scalar factor.

Uses Accessors.jl for functional updates, allowing scaling without
knowing all fields of the blob type.

# Arguments
- `blob::AbstractBlob`: The blob to scale
- `scale`: A scalar value to multiply center and σ by

# Returns
- New blob of the same type with scaled center and σ

# Example
```julia
blob = IsoBlob(Point2(10.0mm, 20.0mm), 2.0mm)
scaled = blob * 300dpi  # Scale to logical units
```
"""
function Base.:*(blob::AbstractBlob, scale)
    # Compute new values first, then update both fields simultaneously
    new_center = blob.center .* scale
    new_σ = blob.σ * scale
    # Use ConstructionBase to set both fields at once, preserving other fields
    return ConstructionBase.setproperties(blob, (center=new_center, σ=new_σ))
end

"""
    /(blob::AbstractBlob, scale)

Divide a blob's center coordinates and σ by a scalar factor.

Uses Accessors.jl for functional updates, allowing scaling without
knowing all fields of the blob type.

# Arguments
- `blob::AbstractBlob`: The blob to scale
- `scale`: A scalar value to divide center and σ by

# Returns
- New blob of the same type with scaled center and σ

# Example
```julia
blob = IsoBlob(Point2(300.0pd, 600.0pd), 15.0pd)
scaled = blob / 300dpi  # Scale to physical units
```
"""
function Base.:/(blob::AbstractBlob, scale)
    # Compute new values first, then update both fields simultaneously
    new_center = blob.center ./ scale
    new_σ = blob.σ / scale
    # Use ConstructionBase to set both fields at once, preserving other fields
    return ConstructionBase.setproperties(blob, (center=new_center, σ=new_σ))
end

