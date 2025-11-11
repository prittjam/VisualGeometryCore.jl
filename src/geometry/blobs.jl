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
    Unitful.ustrip(blob::AbstractBlob)

Strip units from a blob, returning a new blob with unitless coordinates and scale.

This enables the convenient syntax `ustrip.(blobs)` to strip units from a vector of blobs.

# Example
```julia
using Unitful: ustrip

# Blob with units
blob_px = IsoBlob(Point2(100.0px, 200.0px), 5.0px)

# Strip units
blob_unitless = ustrip(blob_px)  # IsoBlob(Point2(100.0, 200.0), 5.0)

# Strip units from vector of blobs
blobs_unitless = ustrip.(blobs)
```
"""
function Unitful.ustrip(blob::AbstractBlob)
    # Strip units using broadcasting - much simpler!
    return ConstructionBase.setproperties(blob, (center=ustrip.(blob.center), σ=ustrip(blob.σ)))
end

"""
    logical_units(blob::AbstractBlob, render_density::LogicalDensity)

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
blob_logical = logical_units(blob_mm, 300dpi)
```
"""
function logical_units(blob::AbstractBlob, render_density::LogicalDensity)
    # Use atomic quantity conversion for each component
    center_logical = logical_units.(blob.center, Ref(render_density))
    σ_logical = logical_units(blob.σ, render_density)

    # Create new blob with converted units
    return ConstructionBase.setproperties(blob, (center=center_logical, σ=σ_logical))
end

"""
    physical_units(blob::AbstractBlob, render_density::LogicalDensity)

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
blob_physical = physical_units(blob_logical, 300dpi)
```
"""
function physical_units(blob::AbstractBlob, render_density::LogicalDensity)
    # Use atomic quantity conversion for each component
    center_physical = physical_units.(blob.center, Ref(render_density))
    σ_physical = physical_units(blob.σ, render_density)

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

# =============================================================================
# Blob Polarity Filtering (Light/Dark Classification)
# =============================================================================

"""
    light_blobs(blobs::Vector{IsoBlobDetection}) -> Vector{IsoBlobDetection}

Filter blob detections to return only light/bright blobs (intensity peaks).

Light blobs are characterized by negative Laplacian values (Lxx + Lyy < 0),
indicating local intensity maxima where both second derivatives are negative.

# Arguments
- `blobs`: Vector of IsoBlobDetection objects

# Returns
- Vector containing only blobs with `laplacian_scale_score < 0` (light/bright blobs)

# Notes
- Blobs with `NaN` laplacian_scale_score (from basic Hessian detector) are excluded
- Only works with detections from Hessian-Laplace detector (`:hessian_laplace` method)
- For basic Hessian detector, use polarity-aware detector methods instead

# Example
```julia
# Detect all blobs with Hessian-Laplace
all_blobs = detect_features(img; method=:hessian_laplace)

# Filter to get only light blobs (bright spots)
bright_spots = light_blobs(all_blobs)
println("Found \$(length(bright_spots)) bright blobs out of \$(length(all_blobs)) total")
```

See also: [`dark_blobs`](@ref), [`detect_features`](@ref)
"""
function light_blobs(blobs::Vector{IsoBlobDetection{T}}) where T
    return filter(b -> !isnan(b.laplacian_scale_score) && b.laplacian_scale_score < 0, blobs)
end

"""
    dark_blobs(blobs::Vector{IsoBlobDetection}) -> Vector{IsoBlobDetection}

Filter blob detections to return only dark blobs (intensity valleys).

Dark blobs are characterized by positive Laplacian values (Lxx + Lyy > 0),
indicating local intensity minima where both second derivatives are positive.

# Arguments
- `blobs`: Vector of IsoBlobDetection objects

# Returns
- Vector containing only blobs with `laplacian_scale_score > 0` (dark blobs)

# Notes
- Blobs with `NaN` laplacian_scale_score (from basic Hessian detector) are excluded
- Only works with detections from Hessian-Laplace detector (`:hessian_laplace` method)
- For basic Hessian detector, use polarity-aware detector methods instead

# Example
```julia
# Detect all blobs with Hessian-Laplace
all_blobs = detect_features(img; method=:hessian_laplace)

# Filter to get only dark blobs (dark spots)
dark_spots = dark_blobs(all_blobs)
println("Found \$(length(dark_spots)) dark blobs out of \$(length(all_blobs)) total")

# Visualize with different colors
import VisualGeometryCore.Spec as S
lscene = S.Imshow(img)
# Convert blobs to circles for visualization (sigma_cutoff = 3.0)
light_circles = Circle.(light_blobs(all_blobs), Ref(3.0))
dark_circles = Circle.(dark_blobs(all_blobs), Ref(3.0))
append!(lscene.plots, S.Locus.(light_circles; color=:yellow, linewidth=2.0))  # Bright blobs in yellow
append!(lscene.plots, S.Locus.(dark_circles; color=:blue, linewidth=2.0))     # Dark blobs in blue
```

See also: [`light_blobs`](@ref), [`detect_features`](@ref)
"""
function dark_blobs(blobs::Vector{IsoBlobDetection{T}}) where T
    return filter(b -> !isnan(b.laplacian_scale_score) && b.laplacian_scale_score > 0, blobs)
end