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
    IsoBlob{T} <: AbstractBlob

Isotropic blob with center coordinates and scale parameter.
Supports both physical and pixel coordinate systems.
Both center and σ use the same type for consistency.

# Fields
- `center::Point2{T}`: Blob center coordinates
- `σ::T`: Scale parameter (standard deviation of Gaussian)

# Examples
```julia
# Pixel coordinates
blob_px = IsoBlob(Point2(100.0pd, 200.0pd), 5.0pd)

# Physical coordinates
blob_mm = IsoBlob(Point2(10.0mm, 20.0mm), 2.0mm)
```
"""
struct IsoBlob{T} <: AbstractBlob
    center::Point2{T}
    σ::T

    function IsoBlob(center::Point2{S}, σ::T) where {S,T}
        # Promote types to common type
        U = promote_type(S, T)
        center_promoted = Point2(convert(U, center[1]), convert(U, center[2]))
        σ_promoted = convert(U, σ)
        new{U}(center_promoted, σ_promoted)
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
# Blob Intersection
# =============================================================================

"""
    intersects(c1::GeometryBasics.Circle, c2::GeometryBasics.Circle) -> Bool

Check if two circles intersect (distance between centers < sum of radii).
"""
intersects(c1::GeometryBasics.Circle, c2::GeometryBasics.Circle) =
    norm(c1.center - c2.center) < (c1.r + c2.r)

"""
    intersects(p::AbstractBlob, q::AbstractBlob, cutoff=SIGMA_CUTOFF) -> Bool

Check if two blobs intersect by constructing circles with radius `cutoff*σ` and testing intersection.
Constructs GeometryBasics.Circle objects with unitless centers and radii.
"""
function intersects(p::AbstractBlob, q::AbstractBlob, cutoff=SIGMA_CUTOFF)
    c1 = GeometryBasics.Circle(Point2(float.(ustrip.(p.center))...), float(ustrip(cutoff * p.σ)))
    c2 = GeometryBasics.Circle(Point2(float.(ustrip.(q.center))...), float(ustrip(cutoff * q.σ)))
    return intersects(c1, c2)
end

# =================================
"""
    IsoBlobDetection{T} <: AbstractBlob

Detected isotropic blob with a response score and `FeaturePolarity`.
Both center and σ use the same type for consistency.

Fields
- `center::Point2{T}`
- `σ::T`
- `response::Float64`
- `polarity::FeaturePolarity`
"""
struct IsoBlobDetection{T} <: AbstractBlob
    center::Point2{T}
    σ::T
    response::Float64
    polarity::FeaturePolarity

    function IsoBlobDetection(center::Point2{S}, σ::T, response::Float64, polarity::FeaturePolarity) where {S,T}
        # Promote types to common type
        U = promote_type(S, T)
        center_promoted = Point2(convert(U, center[1]), convert(U, center[2]))
        σ_promoted = convert(U, σ)
        new{U}(center_promoted, σ_promoted, response, polarity)
    end
end

StructTypes.StructType(::Type{IsoBlobDetection}) = StructTypes.Struct()

# =============================================================================
# Unit Conversion Functions for AbstractBlob
# =============================================================================

"""
    to_logical_units(blob::AbstractBlob, render_density::LogicalDensity)

Convert an AbstractBlob from physical units to logical units using the specified render density.

# Arguments
- `blob::AbstractBlob`: The blob with physical coordinates and scale
- `render_density::LogicalDensity`: Render density (e.g., 300dpi for print, 96dpi for screens)

# Returns
- New blob with logical coordinates (pd or px units matching the density)

# Example
```julia
# Physical blob in millimeters
blob_mm = IsoBlob(Point2(10.0mm, 20.0mm), 2.0mm)

# Convert to logical units at 300 DPI (print) or 96 DPI (screen)
blob_logical = to_logical_units(blob_mm, 300dpi)
```
"""
function to_logical_units(blob::AbstractBlob, render_density::LogicalDensity)
    # Use atomic quantity conversion for each component
    center_logical = to_logical_units.(blob.center, Ref(render_density))
    σ_logical = to_logical_units(blob.σ, render_density)

    # Create new blob with converted units
    return ConstructionBase.setproperties(blob, (center=center_logical, σ=σ_logical))
end

"""
    to_physical_units(blob::AbstractBlob, render_density::LogicalDensity)

Convert an AbstractBlob from logical units to physical units using the specified render density.

# Arguments
- `blob::AbstractBlob`: The blob with logical coordinates and scale (pd or px units)
- `render_density::LogicalDensity`: Render density (e.g., 300dpi for print, 96dpi for screens)

# Returns
- New blob of the same type with physical coordinates (length unit matching density denominator)

# Example
```julia
# Logical blob in pixels/dots
blob_logical = IsoBlob(Point2(300.0pd, 600.0pd), 15.0pd)

# Convert to physical units at 300 DPI (gives inches) or 300pd/mm (gives mm)
blob_physical = to_physical_units(blob_logical, 300dpi)
```
"""
function to_physical_units(blob::AbstractBlob, render_density::LogicalDensity)
    # Use atomic quantity conversion for each component
    center_physical = to_physical_units.(blob.center, Ref(render_density))
    σ_physical = to_physical_units(blob.σ, render_density)

    # Create new blob with converted units
    return ConstructionBase.setproperties(blob, (center=center_physical, σ=σ_physical))
end

