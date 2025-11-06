# =============================================================================
# General Utilities and Helper Functions
# =============================================================================

# =============================================================================
# Type Utilities
# =============================================================================

"""
    neltype(x)
    neltype(::Type{T})

Get the nested element type by recursively descending through aggregate types.

For arrays and other container types, this recursively calls `eltype` until
reaching a non-container (atomic) type. For non-container types, returns the type itself.

# Examples
```julia
# Vector of vectors
neltype(Vector{Vector{Float64}}) # Float64

# Vector of static arrays with units
blob_positions = [SVector(1.0px, 2.0px), SVector(3.0px, 4.0px)]
neltype(blob_positions) # Unitful.Quantity{Float64, 𝐍, Unitful.FreeUnits{(px,), 𝐍, nothing}}

# Get zero with proper units
z = zero(neltype(blob_positions)) # 0.0 px
```
"""
neltype(x) = neltype(typeof(x))
neltype(::Type{T}) where T <: AbstractArray = neltype(eltype(T))
neltype(::Type{T}) where T = T

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

# =============================================================================
# Random Sampling
# =============================================================================

"""
    rand([rng::AbstractRNG], r::HyperRectangle{N,T}) -> Point{N,T}

Generate a single random point uniformly distributed inside the HyperRectangle.

# Arguments
- `rng::AbstractRNG`: Random number generator (optional, defaults to global RNG)
- `r::HyperRectangle{N,T}`: The rectangle to sample from

# Returns
A `Point{N,T}` uniformly sampled from the interior of the rectangle.

# Examples
```julia
using Random

# 2D rectangle
rect = Rect(Point2(0.0, 0.0), Vec2(10.0, 20.0))
pt = rand(rect)  # Random point in [0,10] × [0,20]

# With custom RNG
rng = MersenneTwister(1234)
pt = rand(rng, rect)

# 3D box
box = HyperRectangle(Point3(1.0, 2.0, 3.0), Vec3(4.0, 5.0, 6.0))
pt = rand(box)  # Random point in [1,5] × [2,7] × [3,9]
```
"""
function Random.rand(rng::AbstractRNG, r::HyperRectangle{N,T}) where {N,T}
    # Sample uniformly in [0,1]^N and scale/offset to rectangle bounds
    random_offset = T.(rand(rng, N))
    return r.origin .+ random_offset .* r.widths
end

Random.rand(r::HyperRectangle) = rand(Random.default_rng(), r)

"""
    rand([rng::AbstractRNG], r::HyperRectangle{N,T}, n::Integer) -> Vector{Point{N,T}}

Generate `n` random points uniformly distributed inside the HyperRectangle.

# Arguments
- `rng::AbstractRNG`: Random number generator (optional, defaults to global RNG)
- `r::HyperRectangle{N,T}`: The rectangle to sample from
- `n::Integer`: Number of points to generate

# Returns
A `Vector` of `n` points, each uniformly sampled from the interior of the rectangle.

# Examples
```julia
using Random

# Generate 1000 random points in a 2D rectangle
rect = Rect(Point2(0.0, 0.0), Vec2(10.0, 20.0))
points = rand(rect, 1000)

# With custom RNG
rng = MersenneTwister(1234)
points = rand(rng, rect, 1000)

# 3D box
box = HyperRectangle(Point3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0))
points = rand(box, 500)  # 500 random points in unit cube
```
"""
function Random.rand(rng::AbstractRNG, r::HyperRectangle{N,T}, n::Integer) where {N,T}
    return [rand(rng, r) for _ in 1:n]
end

Random.rand(r::HyperRectangle, n::Integer) = rand(Random.default_rng(), r, n)

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
