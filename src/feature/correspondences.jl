# Correspondence types for associating features between images

"""
    AbstractCspond{S,T}

Abstract type for correspondences between source and target elements.

All correspondence types should have at minimum `source::S` and `target::T` fields.
"""
abstract type AbstractCspond{S,T} end

"""
    Cspond{S, T} <: AbstractCspond{S,T}

Generic correspondence type associating a source element with a target element.

# Fields
- `source::S`: Source element (e.g., point, feature, blob)
- `target::T`: Target element

# Examples
```julia
# Point-to-point correspondence
cs = Cspond(Point2(10.0, 20.0), Point2(15.0, 25.0))
source, target = cs  # Iterable interface

# Blob-to-blob correspondence
cs = Cspond(blob1, blob2)
```
"""
struct Cspond{S,T} <: AbstractCspond{S,T}
    source::S
    target::T
end

# Convenience constructor from tuple
Cspond(t::Tuple{S,T}) where {S,T} = Cspond{S,T}(t[1], t[2])

Base.iterate(cs::Cspond, state=1) = state == 1 ? (cs.source, 2) : state == 2 ? (cs.target, 3) : nothing

"""
    csponds(sources::AbstractVector{S}, targets::AbstractVector{T}) -> StructArray{Cspond{S,T}}

Create a type-stable StructArray of correspondences from two vectors.

This is a convenience function that replaces the verbose:
```julia
StructArray{Cspond{eltype(sources), eltype(targets)}}((sources, targets))
```

with the much simpler:
```julia
csponds(sources, targets)
```

# Examples
```julia
circles = [Circle(Point2(1.0, 2.0), 3.0), Circle(Point2(4.0, 5.0), 6.0)]
ellipses = [Ellipse(Point2(7.0, 8.0), 9.0, 10.0, 0.5), Ellipse(Point2(11.0, 12.0), 13.0, 14.0, 0.5)]

# Create correspondences
cs = csponds(circles, ellipses)

# Type-stable access
cs.source  # Vector{Circle{Float64}}
cs.target  # Vector{Ellipse{Float64}}
```
"""
function csponds(sources::AbstractVector{S}, targets::AbstractVector{T}) where {S,T}
    length(sources) == length(targets) || error("Source and target vectors must have the same length")
    return StructArrays.StructArray{Cspond{S,T}}((sources, targets))
end

"""
    AttributedCspond{S,T,M} <: AbstractCspond{S,T}

Correspondence with associated metadata/attributes.

# Fields
- `source::S`: Source element
- `target::T`: Target element
- `metadata::M`: Associated metadata (e.g., score, descriptor distance, epipolar error)

# Examples
```julia
# With numeric score
cs = AttributedCspond(Point2(10.0, 20.0), Point2(15.0, 25.0), 0.95)

# With named tuple metadata
cs = AttributedCspond(blob1, blob2, (score=0.95, distance=2.3, inlier=true))

# Convenience alias for scored correspondences
cs = ScoredCspond(Point2(10.0, 20.0), Point2(15.0, 25.0), 0.95)
```
"""
struct AttributedCspond{S,T,M} <: AbstractCspond{S,T}
    source::S
    target::T
    metadata::M
end

Base.iterate(cs::AttributedCspond, state=1) = state == 1 ? (cs.source, 2) : state == 2 ? (cs.target, 3) : nothing

"""
    ScoredCspond{S,T}

Type alias for `AttributedCspond{S,T,Float64}` - a correspondence with a numeric score.

# Examples
```julia
cs = ScoredCspond(Point2(10.0, 20.0), Point2(15.0, 25.0), 0.95)
cs.metadata  # Access score via metadata field
```
"""
const ScoredCspond{S,T} = AttributedCspond{S,T,Float64}

# Constructor for ScoredCspond that forwards to AttributedCspond
ScoredCspond(source::S, target::T, score::Float64) where {S,T} = AttributedCspond(source, target, score)

# Type aliases for common correspondence patterns
const Pt2ToPt2{S <: StaticVector{2}, T <: StaticVector{2}} = Cspond{S,T}
const Pt3ToPt2{S <: StaticVector{3}, T <: StaticVector{2}} = Cspond{S,T}

# Type aliases for blob correspondences
const BlobToBlob{S <: AbstractBlob, T <: AbstractBlob} = Cspond{S,T}
const BlobToPt2{S <: AbstractBlob, T <: StaticVector{2}} = Cspond{S,T}
const Pt2ToBlob{S <: StaticVector{2}, T <: AbstractBlob} = Cspond{S,T}

# Add conversion methods to convert to named tuples
Base.convert(::Type{NamedTuple}, cs::Cspond) = (source = cs.source, target = cs.target)
Base.convert(::Type{NamedTuple}, cs::AttributedCspond) = (source = cs.source, target = cs.target, metadata = cs.metadata)

# Convenience method to get a named tuple from a correspondence
Base.NamedTuple(cs::AbstractCspond) = convert(NamedTuple, cs)

# =============================================================================
# N-way Correspondences (all N features correspond together)
# =============================================================================

"""
    CspondSet{N,T}

N-way correspondence where all N features correspond together.
Used for multi-view geometry, feature tracks, grouped features, etc.

# Fields
- `features::SVector{N,T}`: The N corresponding features

# Examples
```julia
# Feature track across 3 frames
track = CspondSet(SVector(pt1, pt2, pt3))

# Access individual features
track.features[1]  # First feature
track[2]           # Second feature (via indexing)

# Iteration
for feat in track
    println(feat)
end

# Construction from tuple
track = CspondSet((pt1, pt2, pt3))
```
"""
struct CspondSet{N,T}
    features::SVector{N,T}
end

# Convenience constructor from tuple
CspondSet(features::Tuple) = CspondSet(SVector(features))

# Indexing interface
Base.getindex(cs::CspondSet, i::Int) = cs.features[i]
Base.length(cs::CspondSet{N}) where {N} = N
Base.size(cs::CspondSet{N}) where {N} = (N,)

# Iteration interface
Base.iterate(cs::CspondSet, state=1) = state > length(cs) ? nothing : (cs.features[state], state + 1)

"""
    AttributedCspondSet{N,T,M}

N-way correspondence with associated metadata.

# Fields
- `features::SVector{N,T}`: The N corresponding features
- `metadata::M`: Associated metadata (e.g., track quality, reprojection error)

# Examples
```julia
# Feature track with quality score
track = AttributedCspondSet(SVector(pt1, pt2, pt3), 0.95)

# Feature track with rich metadata
track = AttributedCspondSet(
    SVector(pt1, pt2, pt3),
    (quality=0.95, reproj_error=0.5, track_length=10)
)
```
"""
struct AttributedCspondSet{N,T,M}
    features::SVector{N,T}
    metadata::M
end

# Convenience constructor from tuple
AttributedCspondSet(features::Tuple, metadata::M) where {M} =
    AttributedCspondSet(SVector(features), metadata)

# Indexing interface
Base.getindex(cs::AttributedCspondSet, i::Int) = cs.features[i]
Base.length(cs::AttributedCspondSet{N}) where {N} = N
Base.size(cs::AttributedCspondSet{N}) where {N} = (N,)

# Iteration interface (iterates over features only, not metadata)
Base.iterate(cs::AttributedCspondSet, state=1) =
    state > length(cs) ? nothing : (cs.features[state], state + 1)

# NamedTuple conversion
Base.convert(::Type{NamedTuple}, cs::CspondSet) = (features = cs.features,)
Base.convert(::Type{NamedTuple}, cs::AttributedCspondSet) =
    (features = cs.features, metadata = cs.metadata)
Base.NamedTuple(cs::Union{CspondSet, AttributedCspondSet}) = convert(NamedTuple, cs)
