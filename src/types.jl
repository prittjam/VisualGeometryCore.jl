
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
# Unit Conversion Functions for IsoBlob
# =============================================================================

"""
    to_logical_units(blob::IsoBlob, render_density::LogicalDensity)

Convert an IsoBlob from physical units to logical units using the specified render density.

# Arguments
- `blob::IsoBlob`: The blob with physical coordinates and scale
- `render_density::LogicalDensity`: Render density (e.g., 300dpi for print, 96dpi for screens)

# Returns
- `IsoBlob`: New blob with logical coordinates (pd or px units matching the density)

# Example
```julia
# Physical blob in millimeters
blob_mm = IsoBlob(Point2(10.0mm, 20.0mm), 2.0mm)

# Convert to logical units at 300 DPI (print) or 96 DPI (screen)
blob_logical = to_logical_units(blob_mm, 300dpi)
```
"""
function to_logical_units(blob::AbstractBlob, render_density::LogicalDensity)
    # Scale blob by render density to get logical units
    # Multiplication by density (𝐍 𝐋^-1) converts physical (𝐋) to logical (𝐍)
    scaled_blob = blob * render_density

    # Determine target logical unit (pd or px) from density's numerator
    # Check unit string to detect px vs pd (dpi uses pd)
    unit_str = string(unit(render_density))
    target_unit = occursin("px", unit_str) ? px : pd

    # Simplify to clean logical units (pd or px)
    center_simplified = uconvert.(target_unit, scaled_blob.center)
    σ_simplified = uconvert(target_unit, scaled_blob.σ)

    return ConstructionBase.setproperties(scaled_blob, (center=center_simplified, σ=σ_simplified))
end

"""
    to_physical_units(blob::AbstractBlob, render_density::LogicalDensity)

Convert an AbstractBlob from logical units to physical units using the specified render density.

# Arguments
- `blob::AbstractBlob`: The blob with logical coordinates and scale (pd or px units)
- `render_density::LogicalDensity`: Render density (e.g., 300dpi for print, 96dpi for screens)

# Returns
- New blob of the same type with physical coordinates in mm

# Example
```julia
# Logical blob in pixels/dots
blob_logical = IsoBlob(Point2(300.0pd, 600.0pd), 15.0pd)

# Convert to physical units at 300 DPI (print) or 96 DPI (screen)
blob_mm = to_physical_units(blob_logical, 300dpi)
```
"""
function to_physical_units(blob::AbstractBlob, render_density::LogicalDensity)
    # Scale blob by dividing by render density to get physical units
    # Division by density (𝐍 𝐋^-1) converts logical (𝐍) to physical (𝐋)
    scaled_blob = blob / render_density

    # Simplify to mm for clean physical units
    center_simplified = uconvert.(mm, scaled_blob.center)
    σ_simplified = uconvert(mm, scaled_blob.σ)

    return ConstructionBase.setproperties(scaled_blob, (center=center_simplified, σ=σ_simplified))
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

