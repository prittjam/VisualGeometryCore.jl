
# ========================================================================
# EuclideanMap Type and Operations
# ========================================================================

"""
    EuclideanMap{N,T,R<:Rotation{N,T},V<:StaticVector{N,T}} <: Transformation

Rigid transformation `x ↦ R*x + t` in `N` dimensions, combining rotation and translation.

This represents a Euclidean (rigid body) transformation that preserves distances and angles.
It consists of a rotation `R` followed by a translation `t`.

# Fields
- `R::Rotation{N,T}`: Rotation component (from Rotations.jl)
- `t::StaticVector{N,T}`: Translation vector

# Constructors
```julia
# From rotation and translation vector
R = RotMatrix{2}(π/4)  # 45° rotation
t = [2.0, 1.0]         # translation
euclidean = EuclideanMap(R, t)

# From MRP (Modified Rodrigues Parameters) for 3D
mrp = MRP(0.1, 0.2, 0.3)
euclidean_3d = EuclideanMap(mrp, [1.0, 2.0, 3.0])

# From tuple
euclidean = EuclideanMap((R, t))
```

# Usage
```julia
# Apply transformation to points
point = Point2f(1.0, 0.0)
transformed = euclidean(point)

# Compose transformations
combined = euclidean2 ∘ euclidean1

# Inverse transformation
inverse_euclidean = inv(euclidean)
```
"""
struct EuclideanMap{N,T,R<:Rotation{N,T},V<:StaticVector{N,T}} <: CoordinateTransformations.Transformation
    R::R
    t::V  # Stored unitless in mm (canonical unit for Length)
end

# Constructors with unit enforcement
"""
    EuclideanMap(R::Rotation{N}, t::AbstractVector{<:Unitful.Length})

Construct EuclideanMap with unit-enforced translation vector.
Translation is converted to mm and stored as unitless Float64.
"""
function EuclideanMap(R::Rotation{N,T}, t::AbstractVector{<:Unitful.Length}) where {N,T}
    # Convert to canonical units (mm) and strip
    t_mm = ustrip.(mm, t)
    return EuclideanMap(R, SVector{N,Float64}(t_mm))
end

# Fallback for unitless (internal use, backwards compatibility)
# Always convert to Float64 for consistent storage
EuclideanMap(R::Rotation{N,T}, t::AbstractVector{<:Real}) where {N,T} =
    EuclideanMap(R, SVector{N,Float64}(t))

EuclideanMap(R::Rotation{N,T}, t::NTuple{N,<:Real}) where {N,T} =
    EuclideanMap(R, SVector{N,Float64}(t))

# Three-argument constructor with unit enforcement
function EuclideanMap(R::Rotation{3,T}, tx::Unitful.Length, ty::Unitful.Length, tz::Unitful.Length) where T
    t_mm = SVector{3,Float64}(ustrip(mm, tx), ustrip(mm, ty), ustrip(mm, tz))
    return EuclideanMap(R, t_mm)
end

EuclideanMap(R::Rotation{3,T}, tx::Real, ty::Real, tz::Real) where T =
    EuclideanMap(R, SVector{3,Float64}(tx, ty, tz))

EuclideanMap(E::EuclideanMap) = E

EuclideanMap((R, t)::Tuple{<:Rotation,<:AbstractVector}) = EuclideanMap(R, t)

# Accessor to reconstruct with units
"""
    translation_mm(e::EuclideanMap) -> SVector{N, Quantity{Float64, mm}}

Get translation vector with mm units attached.
"""
translation_mm(e::EuclideanMap{N}) where N = e.t .* mm

# Operations
(E::EuclideanMap)(x) = E.R * x + E.t

@inline function Base.inv(E::EuclideanMap{N,T}) where {N,T}
    Rin = inv(E.R)
    return EuclideanMap(Rin, -(Rin * E.t))
end

function Base.:∘(A::EuclideanMap{N,TA}, B::EuclideanMap{N,TB}) where {N,TA,TB}
    T = promote_type(TA, TB)
    R_composed = A.R * B.R
    t_composed = A.R * B.t + A.t
    return EuclideanMap(R_composed, t_composed)
end

# Trait method for EuclideanMap
trait(::EuclideanMap) = EuclideanTrait()
trait(::Type{<:EuclideanMap}) = EuclideanTrait()

# Helper functions with unique name to avoid collisions
extract_field(x::Dict{String,Any}, key::String) = x[key]
extract_field(x::JSON3.Object, key::String) = getproperty(x, Symbol(key))
extract_field(x::NamedTuple, key::String) = getproperty(x, Symbol(key))

const ScalarOrQuantity = Union{Number, Unitful.Quantity}

StructTypes.construct(::Type{ScalarOrQuantity}, data) = 
    data isa Number ? data : StructTypes.construct(Unitful.Quantity, data)

# Unit conversion functions are defined in units/conversions.jl

