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

# JSON3 serialization support for IsoBlob
StructTypes.StructType(::Type{<:IsoBlob}) = StructTypes.CustomStruct()

StructTypes.lower(blob::IsoBlob) = (center = blob.center, σ = blob.σ)

function StructTypes.construct(::Type{IsoBlob}, x::Union{Dict{String,Any}, JSON3.Object, NamedTuple})
    center_data = extract_field(x, "center")
    center = Point2(
        StructTypes.construct(ScalarOrQuantity, center_data[1]),
        StructTypes.construct(ScalarOrQuantity, center_data[2])
    )
    
    σ = StructTypes.construct(ScalarOrQuantity, extract_field(x, "σ"))
    
    return IsoBlob(center, σ)
end

# =============================================================================
# Blob Utility Functions
# =============================================================================

"""
    radius(blob::AbstractBlob, k=SIGMA_CUTOFF)

Effective radius defined as `k * σ`.
"""
radius(blob::AbstractBlob, k=SIGMA_CUTOFF) = k * blob.σ

"""
    area(blob::AbstractBlob, k=SIGMA_CUTOFF)

Effective area of a blob modeled as a disk of radius `kσ`.
"""
area(blob::AbstractBlob, k=SIGMA_CUTOFF) = π * (k * blob.σ)^2

"""
    Circle(blob::AbstractBlob, cutoff=SIGMA_CUTOFF) -> GeometryBasics.Circle

Create a `GeometryBasics.Circle` centered at the blob with radius `kσ` (unitless),
useful for spatial indexing and plotting.
"""
function Circle(blob::AbstractBlob, cutoff=SIGMA_CUTOFF)
    center_vals = float.(ustrip.(blob.center))
    radius_val = float(ustrip(radius(blob, cutoff)))
    return GeometryBasics.Circle(Point2(center_vals), radius_val)
end

"""
    intersects(c1::GeometryBasics.Circle, c2::GeometryBasics.Circle) -> Bool

Check if two circles intersect (distance between centers < sum of radii).
"""
intersects(c1::GeometryBasics.Circle, c2::GeometryBasics.Circle) =
    norm(c1.center - c2.center) < (c1.r + c2.r)

"""
    intersects(p::AbstractBlob, q::AbstractBlob, cutoff=SIGMA_CUTOFF) -> Bool

Check if two blobs intersect by constructing circles with radius `cutoff*σ` and testing intersection.
"""
intersects(p::AbstractBlob, q::AbstractBlob, cutoff=SIGMA_CUTOFF) =
    intersects(Circle(p, cutoff), Circle(q, cutoff))

# =================================
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

