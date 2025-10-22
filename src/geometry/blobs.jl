"""
    FeaturePolarity

Structure type of a detected feature based on Hessian determinant sign.

- `BlobFeature`: Blob-like structure (positive det(H), both eigenvalues same sign)
  - Includes both bright blobs (intensity peaks) and dark blobs (intensity valleys)
  - det(H) = Lxx*Lyy - Lxy² > 0

- `SaddleFeature`: Saddle/edge-like structure (negative det(H), eigenvalues opposite signs)
  - Ridges, valleys, corners, and edge-like structures
  - det(H) = Lxx*Lyy - Lxy² < 0

# Distinguishing Bright vs Dark Blobs

To distinguish bright blobs from dark blobs, use the Laplacian (trace of Hessian):
- Laplacian = Lxx + Lyy < 0: bright blob (intensity peak, both Lxx and Lyy negative)
- Laplacian = Lxx + Lyy > 0: dark blob (intensity valley, both Lxx and Lyy positive)

Note: VLFeat's basic Hessian detector does not distinguish bright vs dark blobs.
Use `VL_COVDET_METHOD_HESSIAN_LAPLACE` for Laplacian-based bright/dark classification.
"""
@enum FeaturePolarity begin
    BlobFeature      # Positive det(H) - blob-like structure
    SaddleFeature    # Negative det(H) - saddle/edge-like structure
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
# Circle Construction from Blobs
# =============================================================================

"""
    GeometryBasics.Circle(blob::AbstractBlob, cutoff::Real) -> Circle

Construct a GeometryBasics.Circle from a blob with radius `cutoff * σ`.

The `cutoff` parameter determines the effective radius as a multiple of σ
(e.g., 3.0 for 3σ radius). Centers and radii are converted to unitless Float64.

# Arguments
- `blob::AbstractBlob`: Blob with center and scale σ
- `cutoff::Real`: Radius multiplier (typically 2-4)

# Returns
- `GeometryBasics.Circle`: Circle with unitless center and radius

# Example
```julia
blob = IsoBlob(Point2(100.0pd, 200.0pd), 5.0pd)
circle = Circle(blob, 3.0)  # Circle with radius 15.0
```
"""
function GeometryBasics.Circle(blob::AbstractBlob, cutoff::Real)
    center = Point2(float.(ustrip.(blob.center))...)
    radius = float(ustrip(cutoff * blob.σ))
    return GeometryBasics.Circle(center, radius)
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
    intersects(p::AbstractBlob, q::AbstractBlob, cutoff::Real) -> Bool

Check if two blobs intersect by constructing circles with radius `cutoff*σ` and testing intersection.
The `cutoff` parameter determines the effective radius as a multiple of σ (e.g., 3.0 for 3σ radius).
Constructs GeometryBasics.Circle objects with unitless centers and radii.

# Example
```julia
# Test if blobs overlap at 3σ radius
intersects(blob1, blob2, 3.0)
```
"""
function intersects(p::AbstractBlob, q::AbstractBlob, cutoff::Real)
    c1 = GeometryBasics.Circle(Point2(float.(ustrip.(p.center))...), float(ustrip(cutoff * p.σ)))
    c2 = GeometryBasics.Circle(Point2(float.(ustrip.(q.center))...), float(ustrip(cutoff * q.σ)))
    return intersects(c1, c2)
end

# =================================
"""
    IsoBlobDetection{T} <: AbstractBlob

Detected isotropic blob with complete Hessian-based characterization.
Both center and σ use the same type for consistency.

Matches VLFeat's VlCovDetFeature structure with fields:
- Position/scale (frame)
- Response strength (peakScore = det(H))
- Edge score (edgeScore)
- Laplacian value (laplacianScaleScore for bright/dark classification)

All detections are true blobs (saddle points with det(H) < 0 are rejected during refinement).

# Fields
- `center::Point2{T}`: Blob center in input image coordinates
- `σ::T`: Scale parameter (standard deviation of Gaussian)
- `response::Float64`: Peak response value (Hessian determinant, always positive)
- `edge_score::Float64`: Edge score (lower = more blob-like, higher = more edge-like)
- `laplacian_scale_score::Float64`: Laplacian (trace of Hessian) value at detection
  - `< 0`: Bright blob (intensity peak, Lxx + Lyy < 0)
  - `> 0`: Dark blob (intensity valley, Lxx + Lyy > 0)
  - `NaN`: Laplacian not computed (basic Hessian detector without Laplacian response)
"""
struct IsoBlobDetection{T} <: AbstractBlob
    center::Point2{T}
    σ::T
    response::Float64
    edge_score::Float64
    laplacian_scale_score::Float64

    function IsoBlobDetection(center::Point2{S}, σ::T, response::Float64, edge_score::Float64, laplacian_scale_score::Float64) where {S,T}
        # Promote types to common type
        U = promote_type(S, T)
        center_promoted = Point2(convert(U, center[1]), convert(U, center[2]))
        σ_promoted = convert(U, σ)
        new{U}(center_promoted, σ_promoted, response, edge_score, laplacian_scale_score)
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

# =============================================================================
# Coordinate Convention Support
# =============================================================================

"""
    GeometryBasics.origin(blob::AbstractBlob) -> Point2

Get the center of a blob as its origin, making it compatible with generic
GeometryBasics primitives for coordinate convention conversion.

# Example
```julia
blob = IsoBlob(Point2(10.0pd, 20.0pd), 2.0pd)
pos = origin(blob)  # Returns Point2(10.0pd, 20.0pd)
```
"""
GeometryBasics.origin(blob::AbstractBlob) = blob.center

"""
    shift_origin(blob::AbstractBlob, offset) -> typeof(blob)

Shift the center of a blob by the given offset.

This specialized method enables `change_image_origin` to work with blob types.

# Arguments
- `blob::AbstractBlob`: Blob to shift (IsoBlob or IsoBlobDetection)
- `offset`: Offset to add to center (Point, Vec, or tuple with units)

# Returns
New blob with shifted center, preserving all other properties

# Example
```julia
blob = IsoBlob(Point2(10.0pd, 20.0pd), 2.0pd)
shifted = shift_origin(blob, Point2(0.5pd, 0.5pd))  # center at (10.5pd, 20.5pd)
```
"""
function shift_origin(blob::AbstractBlob, offset)
    new_center = blob.center .+ offset
    return ConstructionBase.setproperties(blob, (center=new_center,))
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