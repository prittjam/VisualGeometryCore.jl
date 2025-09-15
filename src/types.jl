"""
    FeaturePolarity

Polarity of an image feature or blob response.
- `PositiveFeature`: bright-on-dark response
- `NegativeFeature`: dark-on-bright response
"""
@enum FeaturePolarity begin
    PositiveFeature
    NegativeFeature
end

abstract type ImageFeature end
"""
    AbstractBlob <: ImageFeature

Abstract supertype for blob-like features with a `center` and scale `σ`.
Concrete implementations include `IsoBlob` and `IsoBlobDetection`.
"""
abstract type AbstractBlob <: ImageFeature end

"""
    IsoBlob{S,T} <: AbstractBlob

Isotropic blob with center coordinates and scale parameter.
Supports both physical and pixel coordinate systems.

# Fields
- `center::Point2{S}`: Blob center coordinates
- `σ::T`: Scale parameter (standard deviation of Gaussian)

# Examples
```julia
# Pixel coordinates
blob_px = IsoBlob(Point2(100.0pd, 200.0pd), 5.0pd)

# Physical coordinates  
blob_mm = IsoBlob(Point2(10.0mm, 20.0mm), 2.0mm)
```
"""
struct IsoBlob{S,T} <: AbstractBlob
    center::Point2{S}
    σ::T
    
    function IsoBlob(center::Point2{S}, σ::T) where {S,T}
        # Safe dimension check - treat types without dimension method as NoDims
        safe_dimension(::Type{U}) where U = hasmethod(dimension, (Type{U},)) ? dimension(U) : Unitful.NoDims
        @assert safe_dimension(S) == safe_dimension(T) "Center coordinates and σ must have compatible units. Got $(safe_dimension(S)) for coordinates and $(safe_dimension(T)) for σ."
        new{S,T}(center, σ)
    end
end

StructTypes.StructType(::Type{IsoBlob}) = StructTypes.Struct()

"""
    IsoBlobDetection{S,T} <: AbstractBlob

Detected isotropic blob with a response score and `FeaturePolarity`.

Fields
- `center::Point2{S}`
- `σ::T`
- `response::Float64`
- `polarity::FeaturePolarity`
"""
struct IsoBlobDetection{S, T} <: AbstractBlob
    center::Point2{S}
    σ::T
    response::Float64
    polarity::FeaturePolarity
    
    function IsoBlobDetection(center::Point2{S}, σ::T, response::Float64, polarity::FeaturePolarity) where {S,T}
        # Safe dimension check - treat types without dimension method as NoDims
        safe_dimension(::Type{U}) where U = hasmethod(dimension, (Type{U},)) ? dimension(U) : Unitful.NoDims
        @assert safe_dimension(S) == safe_dimension(T) "Center coordinates and σ must have compatible units. Got $(safe_dimension(S)) for coordinates and $(safe_dimension(T)) for σ."
        new{S,T}(center, σ, response, polarity)
    end
end

StructTypes.StructType(::Type{IsoBlobDetection}) = StructTypes.Struct()

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

const ScalarOrQuantity = Union{Number, Unitful.Quantity}

function StructTypes.construct(::Type{ScalarOrQuantity}, data)
    data isa Number ? data : StructTypes.construct(Unitful.Quantity, data)
end
