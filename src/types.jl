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
        @assert dimension(S) == dimension(T) "Center coordinates and σ must have compatible units. Got $(dimension(S)) for coordinates and $(dimension(T)) for σ."
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
        @assert dimension(S) == dimension(T) "Center coordinates and σ must have compatible units. Got $(dimension(S)) for coordinates and $(dimension(T)) for σ."
        new{S,T}(center, σ, response, polarity)
    end
end

StructTypes.StructType(::Type{IsoBlobDetection}) = StructTypes.Struct()

