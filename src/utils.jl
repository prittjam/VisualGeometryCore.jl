# =============================================================================
# General Utilities and Helper Functions
# =============================================================================

# =============================================================================
# JSON3 and StructTypes Support
# =============================================================================

StructTypes.StructType(::Type{<:Unitful.Quantity}) = StructTypes.CustomStruct()

StructTypes.lower(q::Unitful.Quantity) = (
    value = ustrip(q),
    unit = string(unit(q)),
)

function StructTypes.construct(::Type{<:Unitful.Quantity}, obj)
    # Parse unit; trust the numeric value as saved (JSON provides Int for integers, Float64 otherwise)
    u = Unitful.uparse(obj["unit"]; unit_context=@__MODULE__)
    v = obj["value"]
    return v * u
end

# ScalarOrQuantity is defined in types.jl

# JSON3 serialization support for Size2
# Use CustomStruct to handle the FieldVector serialization properly
StructTypes.StructType(::Type{<:Size2}) = StructTypes.CustomStruct()

# Custom serialization - convert to named tuple
StructTypes.lower(s::Size2) = (width = s.width, height = s.height)

# Custom deserialization - reconstruct Size2 with proper Unitful handling
function StructTypes.construct(::Type{Size2}, x)
    get_field = x isa Dict ? (key) -> x[key] : (key) -> getproperty(x, key)

    # Reconstruct Unitful quantities if needed
    width_data = get_field("width")
    height_data = get_field("height")
    
    width = if width_data isa Dict && haskey(width_data, "value") && haskey(width_data, "unit")
        # Use proper Unitful deserialization to preserve numeric types
        StructTypes.construct(ScalarOrQuantity, width_data)
    else
        width_data
    end
    
    height = if height_data isa Dict && haskey(height_data, "value") && haskey(height_data, "unit")
        # Use proper Unitful deserialization to preserve numeric types
        StructTypes.construct(Unitful.Quantity, height_data)
    else
        height_data
    end
    
    return Size2(width=width, height=height)
end

function StructTypes.construct(::Type{Size2{T}}, x) where T
    return StructTypes.construct(Size2, x)
end

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
    cartesian_ranges(p1::StaticVector{N,<:Integer}, p2::StaticVector{N,<:Integer}) -> NTuple{N, UnitRange{<:Integer}}

Convert two N-dimensional integer points to a tuple of ranges suitable for CartesianIndices.

Creates ranges p1[i]:p2[i] for each dimension i, which can be passed directly to
CartesianIndices to create array indices.

# Arguments
- `p1`: Lower bound point (each component becomes start of range)
- `p2`: Upper bound point (each component becomes end of range)

# Returns
Tuple of UnitRange objects, one per dimension

# Examples
```julia
# 2D example
lo = Point2(1, 3)
hi = Point2(5, 10)
ranges = cartesian_ranges(lo, hi)  # Returns (1:5, 3:10)
ci = CartesianIndices(ranges)

# 3D example
lo = Point3(1, 2, 3)
hi = Point3(5, 6, 7)
ranges = cartesian_ranges(lo, hi)  # Returns (1:5, 2:6, 3:7)
ci = CartesianIndices(ranges)

# Can also use Vec instead of Point
lo = Vec2(1, 3)
hi = Vec2(5, 10)
ci = CartesianIndices(cartesian_ranges(lo, hi))
```
"""
cartesian_ranges(p1::StaticVector{N,<:Integer}, p2::StaticVector{N,<:Integer}) where N =
    ntuple(i -> p1[i]:p2[i], N)

# =============================================================================
# Pixel Coordinate Convention Conversion
# =============================================================================

"""
    apply_pixel_convention(primitive, convention::Symbol) -> typeof(primitive)

Apply pixel coordinate convention shift to any GeometryBasics primitive that supports `origin()`.

This generic helper shifts the position of geometric primitives to match different pixel
coordinate conventions used across computer vision systems.

# Pixel Coordinate Conventions

Different computer vision systems use different conventions for pixel coordinates:

- **Native/Makie** (`:native`): First pixel center at (0.5, 0.5), corner at (0, 0)
- **MATLAB** (`:matlab`): First pixel center at (1, 1), corner at (0.5, 0.5)
- **OpenCV** (`:opencv`): First pixel center at (0, 0), corner at (-0.5, -0.5)

# Arguments
- `primitive`: Any GeometryBasics type with `origin()` method (Circle, Sphere, Rect, IsoBlob, etc.)
- `convention::Symbol`: Target pixel coordinate convention

# Returns
New primitive of same type with shifted origin

# Examples
```julia
using VisualGeometryCore
using GeometryBasics

# Works with any GeometryBasics primitive
circle = Circle(Point2(10.0, 20.0), 5.0)
circle_matlab = apply_pixel_convention(circle, :matlab)  # center at (10.5, 20.5)
circle_opencv = apply_pixel_convention(circle, :opencv)  # center at (9.5, 19.5)

# Works with units
blob = IsoBlob(Point2(10.0pd, 20.0pd), 2.0pd)
blob_matlab = apply_pixel_convention(blob, :matlab)  # center at (10.5pd, 20.5pd)

# Can broadcast over collections
blobs = [IsoBlob(Point2(i, i), 1.0) for i in 1:5]
blobs_opencv = apply_pixel_convention.(blobs, :opencv)
```
"""
function apply_pixel_convention(primitive, convention::Symbol)
    if convention == :native
        return primitive  # No shift needed
    elseif convention == :matlab
        # MATLAB: first pixel center at (1, 1) - add 0.5 offset
        pos = origin(primitive)
        offset = (0.5, 0.5) .* unit(eltype(pos))
        return shift_origin(primitive, offset)
    elseif convention == :opencv
        # OpenCV: first pixel center at (0, 0) - subtract 0.5 offset
        pos = origin(primitive)
        offset = (0.5, 0.5) .* unit(eltype(pos))
        return shift_origin(primitive, .-offset)
    else
        error("Unknown convention :$convention. Valid options: :native, :matlab, :opencv")
    end
end

"""
    shift_origin(primitive, offset) -> typeof(primitive)

Shift the origin of a geometric primitive by the given offset.

This is an internal helper function used by `apply_pixel_convention`.
Add specialized methods for custom types that don't follow standard GeometryBasics patterns.

# Arguments
- `primitive`: Geometric primitive to shift
- `offset`: Offset to add to origin (Point, Vec, or tuple with units)

# Examples
```julia
# Default implementation for GeometryBasics Circle
circle = Circle(Point2(10.0, 20.0), 5.0)
shifted = shift_origin(circle, Point2(0.5, 0.5))  # center at (10.5, 20.5)
```
"""
function shift_origin(primitive::GeometryBasics.HyperSphere, offset)
    new_center = origin(primitive) .+ offset
    return typeof(primitive)(new_center, primitive.r)
end

"""
    CartesianIndices(r::Rect)

Create CartesianIndices from a Rect.
"""
Base.CartesianIndices(r::Rect{Integer}) =
    CartesianIndices((r.origin[2]:r.origin[2]+r.widths[2]-1, r.origin[1]:r.origin[1]+r.widths[1]-1))

"""
    Rect(indices::CartesianIndices)

Create a Rect from CartesianIndices.
"""
function Rect(indices::CartesianIndices{2})
    ranges = indices.indices
    y_range, x_range = ranges  # CartesianIndices uses (row, col) = (y, x) ordering
    origin = Point2(first(x_range), first(y_range))
    widths = Vec2(length(x_range), length(y_range))

    return Rect(origin, widths)
end

Point2i(index::CartesianIndex{2}) = Point2{Int}(index[2], index[1])
Point2(index::CartesianIndex{2}) = Point2i(index)

# Generic StaticVector constructors that work with both Point2 and Vec2
StaticVector{2,Int}(index::CartesianIndex{2}) = Point2{Int}(index[2], index[1])
StaticVector{2,T}(index::CartesianIndex{2}) where T = StaticVector{2,T}(T(index[2]), T(index[1]))

# =============================================================================
# Helper Functions
# =============================================================================

"""
    filter_kwargs(T; kwargs...) -> (allowed, unknown)

Split keyword args into those that match fieldnames of `T` and those that don't.
"""
function filter_kwargs(::Type{T}; kwargs...) where {T}
    fns = Set(fieldnames(T))
    ks  = keys(kwargs)
    keep = Tuple(k for k in ks if k in fns)
    drop = Tuple(k for k in ks if k ∉ fns)
    
    # Create NamedTuples with only the relevant keys and values
    keep_values = Tuple(kwargs[k] for k in keep)
    drop_values = Tuple(kwargs[k] for k in drop)
    
    return NamedTuple{keep}(keep_values), NamedTuple{drop}(drop_values)
end

"""
    validate_dir(dir::AbstractString; create::Bool=false)

Validate that a directory exists, optionally creating it if it doesn't.

# Arguments
- `dir`: Path to the directory to validate
- `create=false`: If true, create the directory if it doesn't exist

# Returns
- The validated directory path

# Throws
- `ArgumentError` if the directory doesn't exist and `create=false`
"""
function validate_dir(dir::AbstractString; create::Bool=false)
    if create && !isdir(dir)
        mkpath(dir)
    end
    if !isdir(dir)
        throw(ArgumentError("Directory does not exist: $dir"))
    end
    return dir
end

# =============================================================================
# IMAGE PROCESSING UTILITIES
# =============================================================================

"""
    vlfeat_upsample(src::Matrix{Gray{T}}) where T

Upsample image using VLFeat's bilinear interpolation method.

This matches the `copy_and_upsample` function from VLFeat scalespace.c,
which creates a 2x upsampled image using the following pattern for each pixel:
- dst[0] = v00
- dst[1] = 0.5 * (v00 + v10)  # horizontal neighbor
- dst[width*2] = 0.5 * (v00 + v01)  # vertical neighbor
- dst[width*2+1] = 0.25 * (v00 + v01 + v10 + v11)  # diagonal

# Arguments
- `src`: Source grayscale image as Matrix{Gray{T}}

# Returns
- Upsampled image as Matrix{Gray{Float32}} with dimensions 2x the input

# Examples
```julia
using Colors
img = rand(Gray{Float32}, 32, 32)
upsampled = vlfeat_upsample(img)  # Returns 64×64 image
```
"""
function vlfeat_upsample(src::Matrix{Gray{T}}) where T
    height, width = size(src)
    dst = Matrix{Gray{Float32}}(undef, height * 2, width * 2)

    for y in 1:height
        oy = (y < height) ? width : 0
        v10 = src[y, 1]
        v11 = (y < height) ? src[y+1, 1] : src[y, 1]

        for x in 1:width
            ox = (x < width) ? 1 : 0
            v00 = v10
            v01 = v11
            v10 = src[y, x + ox]
            v11 = (y < height) ? src[y+1, x + ox] : src[y, x + ox]

            # Convert to Float32 for arithmetic
            f00, f01, f10, f11 = Float32(v00.val), Float32(v01.val), Float32(v10.val), Float32(v11.val)

            dst_y = 2*y - 1
            dst_x = 2*x - 1
            dst[dst_y, dst_x] = Gray{Float32}(f00)
            dst[dst_y, dst_x+1] = Gray{Float32}(0.5f0 * (f00 + f10))
            dst[dst_y+1, dst_x] = Gray{Float32}(0.5f0 * (f00 + f01))
            dst[dst_y+1, dst_x+1] = Gray{Float32}(0.25f0 * (f00 + f01 + f10 + f11))
        end
    end

    return dst
end

"""
    vlfeat_upsample!(dst::AbstractMatrix{Gray{Float32}}, src::Matrix{Gray{T}}) where T

Mutating version of VLFeat's bilinear upsampling that writes directly into pre-allocated destination.

This matches the `copy_and_upsample` function from VLFeat scalespace.c,
which creates a 2x upsampled image using the following pattern for each pixel:
- dst[0] = v00
- dst[1] = 0.5 * (v00 + v10)  # horizontal neighbor
- dst[width*2] = 0.5 * (v00 + v01)  # vertical neighbor
- dst[width*2+1] = 0.25 * (v00 + v01 + v10 + v11)  # diagonal
"""
function vlfeat_upsample!(dst::AbstractMatrix{Gray{Float32}}, src::Matrix{Gray{T}}) where T
    height, width = size(src)
    @assert size(dst) == (height * 2, width * 2) "Destination must be 2x the source size"

    for y in 1:height
        oy = (y < height) ? width : 0
        v10 = src[y, 1]
        v11 = (y < height) ? src[y+1, 1] : src[y, 1]

        for x in 1:width
            ox = (x < width) ? 1 : 0
            v00 = v10
            v01 = v11
            v10 = src[y, x + ox]
            v11 = (y < height) ? src[y+1, x + ox] : src[y, x + ox]

            # Convert to Float32 for arithmetic
            f00, f01, f10, f11 = Float32(v00.val), Float32(v01.val), Float32(v10.val), Float32(v11.val)

            dst_y = 2*y - 1
            dst_x = 2*x - 1
            dst[dst_y, dst_x] = Gray{Float32}(f00)
            dst[dst_y, dst_x+1] = Gray{Float32}(0.5f0 * (f00 + f10))
            dst[dst_y+1, dst_x] = Gray{Float32}(0.5f0 * (f00 + f01))
            dst[dst_y+1, dst_x+1] = Gray{Float32}(0.25f0 * (f00 + f01 + f10 + f11))
        end
    end
    
    return dst
end

# =============================================================================
# UID Generation Constants
# =============================================================================

# Choose one:
const ALPHABET_BASE64 = collect("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/")
const ALPHABET_BASE58 = collect("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")