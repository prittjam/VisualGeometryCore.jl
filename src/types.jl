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
# Blob Arithmetic Operations
# =============================================================================

"""
    *(blob::IsoBlob, scalar::Number) -> IsoBlob
    *(scalar::Number, blob::IsoBlob) -> IsoBlob

Scale a blob by multiplying both its center coordinates and σ by the scalar.
This is useful for oversampling operations where the entire blob needs to be scaled.
"""
Base.:*(blob::IsoBlob, scalar::Number) = IsoBlob(blob.center .* scalar, blob.σ * scalar)
Base.:*(scalar::Number, blob::IsoBlob) = blob * scalar

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
    intersects(p::AbstractBlob, q::AbstractBlob, cutoff=SIGMA_CUTOFF) -> Bool

True if the distance between centers is less than the sum of effective radii.
"""
intersects(p::AbstractBlob, q::AbstractBlob, cutoff=SIGMA_CUTOFF) =
    norm(p.center - q.center) < (radius(p, cutoff) + radius(q, cutoff))

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
# Arithmetic Operations for AbstractBlob
# =============================================================================

"""
    +(blob::AbstractBlob, offset)

Translate a blob by adding an offset to its center position.

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
Base.:+(blob::AbstractBlob, offset) = typeof(blob)(blob.center .+ offset, blob.σ, blob.polarity)

"""
    -(blob::AbstractBlob, offset)

Translate a blob by subtracting an offset from its center position.

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
Base.:-(blob::AbstractBlob, offset) = typeof(blob)(blob.center .- offset, blob.σ, blob.polarity)

