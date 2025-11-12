# =============================================================================
# Geometry Utilities
# =============================================================================

# ustrip methods for HyperRectangle (Rect)
"""
    Unitful.ustrip(r::HyperRectangle)

Remove units from HyperRectangle, returning raw numbers.
"""
Unitful.ustrip(r::HyperRectangle) = HyperRectangle(ustrip.(r.origin), ustrip.(r.widths))

"""
    Unitful.ustrip(u::Unitful.Units, r::HyperRectangle)

Remove units from HyperRectangle, converting to specified unit first.
"""
Unitful.ustrip(u::Unitful.Units, r::HyperRectangle) = HyperRectangle(ustrip.(u, r.origin), ustrip.(u, r.widths))

# Round method for HyperRectangle (which is what Rect actually is)
"""
    Base.round(::Type{T}, r::HyperRectangle)

Round HyperRectangle to a specific type.
"""
function Base.round(::Type{T}, r::HyperRectangle{N,S}) where {T,S,N}
    # Use broadcasting to round each component of the origin and widths
    rounded_origin = round.(T, r.origin)
    rounded_widths = round.(T, r.widths)

    return HyperRectangle{N,T}(rounded_origin, rounded_widths)
end

"""
    center(r::HyperRectangle)

Get the center point of a HyperRectangle (Rect/Rect2).

# Examples
```julia
rect = Rect2(0.5, 0.5, 1920.0, 1200.0)
c = center(rect)  # [960.5, 600.5]
```
"""
center(r::HyperRectangle) = r.origin .+ r.widths ./ 2

# Size2 constructor from Rect using widths field
"""
    Size2(r::Rect{N,T}) where {N,T}

Create a Size2 from a Rect using its widths field.
"""
Size2(r::Rect) = Size2(width = r.widths[1], height = r.widths[2])

"""
    Rect(s::Size2{T}) where T

Create a Rect from a Size2 with origin at (0,0) and widths filled from the size dimensions.
"""
Rect(s::Size2{T}) where T = Rect(Point2{T}(zero(T), zero(T)), Vec2{T}(s.width, s.height))

"""
    intervals(r::HyperRectangle{N}) -> NTuple{N, ClosedInterval}

Convert an N-dimensional HyperRectangle to a tuple of closed intervals, one per axis.

# Returns
An N-tuple of `ClosedInterval` objects where each interval spans from
`origin[i]` to `origin[i] + widths[i]` for dimension `i`.

# Examples
```julia
# 2D rectangle
rect = Rect(Point2(1.0, 2.0), Vec2(10.0, 20.0))
x_int, y_int = intervals(rect)  # (1.0..11.0, 2.0..22.0)

# 3D box
box = HyperRectangle(Point3(0.0, 0.0, 0.0), Vec3(1.0, 2.0, 3.0))
x_int, y_int, z_int = intervals(box)  # (0.0..1.0, 0.0..2.0, 0.0..3.0)
```
"""
intervals(r::HyperRectangle{N}) where N =
    ntuple(i -> r.origin[i] .. (r.origin[i] + r.widths[i]), N)

"""
    ranges(p1::StaticVector{N,<:Integer}, p2::StaticVector{N,<:Integer}) -> NTuple{N, UnitRange{<:Integer}}

Convert two N-dimensional integer points to a tuple of ranges suitable for CartesianIndices.

Creates ranges p1[i]:p2[i] for each dimension i, which can be passed directly to
CartesianIndices to create array indices.

# Arguments
- `p1`: Lower bound point (each component becomes start of range)
- `p2`: Upper bound point (each component becomes end of range)

# Returns
Tuple of UnitRange objects, one per dimension

"""
ranges(p1::StaticVector{N,<:Integer}, p2::StaticVector{N,<:Integer}) where N =
    ntuple(i -> p1[i]:p2[i], N)

# =============================================================================
# Pixel Coordinate Convention Conversion
# =============================================================================

"""
    image_origin_offset(; from::Symbol, to::Symbol) -> Vec2{<:Quantity{px}}

Compute the offset vector needed to convert coordinates from one image origin convention to another.

This is a compositional function that returns just the offset (with units of px), allowing you to apply
it however you need (e.g., to points, blobs, or other geometric primitives).

# Image Origin Conventions
- `:opencv`, `:vlfeat`: Top-left pixel center at (0, 0)
- `:makie`, `:colmap`: Top-left pixel center at (0.5, 0.5)
- `:matlab`, `:julia`: Top-left pixel center at (1, 1)

# Arguments
- `from::Symbol`: Source image origin convention
- `to::Symbol`: Target image origin convention

# Returns
`Vec2` offset (with units of px) to add to coordinates when converting from `from` to `to` convention.

# Examples
```julia
# Get offset from OpenCV to MATLAB convention
offset = image_origin_offset(from=:opencv, to=:matlab)  # Vec2(1.0px, 1.0px)

# Apply to a point
pt_opencv = Point2(10.0px, 20.0px)
pt_matlab = pt_opencv .+ offset  # Point2(11.0px, 21.0px)

# Apply to blob centers
blobs_opencv = detect_features(img)
offset = image_origin_offset(from=:opencv, to=:makie)
blobs_makie = [@set blob.center = blob.center .+ offset for blob in blobs_opencv]
```
"""
function image_origin_offset(; from::Symbol, to::Symbol)
    # Use INTRINSICS_COORDINATE_OFFSET from sensors.jl
    if !haskey(INTRINSICS_COORDINATE_OFFSET, from)
        valid = join(sort(collect(keys(INTRINSICS_COORDINATE_OFFSET))), ", :")
        error("Unknown source convention :$from. Valid options: :$valid")
    end
    if !haskey(INTRINSICS_COORDINATE_OFFSET, to)
        valid = join(sort(collect(keys(INTRINSICS_COORDINATE_OFFSET))), ", :")
        error("Unknown target convention :$to. Valid options: :$valid")
    end

    # Offset = destination_corner - source_corner (with units of px)
    return (INTRINSICS_COORDINATE_OFFSET[to] .- INTRINSICS_COORDINATE_OFFSET[from]) .* px
end


Point2i(index::CartesianIndex{2}) = Point2{Int}(index[2], index[1])
Point2(index::CartesianIndex{2}) = Point2i(index)

# Generic StaticVector constructors that work with both Point2 and Vec2
StaticVector{2,Int}(index::CartesianIndex{2}) = Point2{Int}(index[2], index[1])
StaticVector{2,T}(index::CartesianIndex{2}) where T = StaticVector{2,T}(T(index[2]), T(index[1]))
