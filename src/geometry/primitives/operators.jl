
# =============================================================================
# Translation Operators for Circles and Ellipses
# =============================================================================

"""
    Base.+(c::Circle, offset) -> Circle

Translate a circle by adding a 2D offset vector to its center.

# Example
```julia
circle = Circle(Point2(5.0, 3.0), 2.5)
offset = SVector(1.0, -2.0)
translated = circle + offset  # Circle at (6.0, 1.0) with radius 2.5
```
"""
Base.:+(c::GeometryBasics.Circle, offset) =
    GeometryBasics.Circle(c.center + offset, c.r)

"""
    Base.-(c::Circle, offset) -> Circle

Translate a circle by subtracting a 2D offset vector from its center.

# Example
```julia
circle = Circle(Point2(5.0, 3.0), 2.5)
offset = SVector(1.0, -2.0)
translated = circle - offset  # Circle at (4.0, 5.0) with radius 2.5
```
"""
Base.:-(c::GeometryBasics.Circle, offset) =
    GeometryBasics.Circle(c.center - offset, c.r)

"""
    Base.+(e::Ellipse, offset) -> Ellipse

Translate an ellipse by adding a 2D offset vector to its center.

# Example
```julia
ellipse = Ellipse(Point2(10.0, 20.0), 5.0, 3.0, π/4)
offset = SVector(2.0, -1.0)
translated = ellipse + offset  # Ellipse at (12.0, 19.0) with same axes and orientation
```
"""
Base.:+(e::Ellipse{T}, offset) where {T} =
    Ellipse{T}(e.center + offset, e.a, e.b, e.θ)

"""
    Base.-(e::Ellipse, offset) -> Ellipse

Translate an ellipse by subtracting a 2D offset vector from its center.

# Example
```julia
ellipse = Ellipse(Point2(10.0, 20.0), 5.0, 3.0, π/4)
offset = SVector(2.0, -1.0)
translated = ellipse - offset  # Ellipse at (8.0, 21.0) with same axes and orientation
```
"""
Base.:-(e::Ellipse{T}, offset) where {T} =
    Ellipse{T}(e.center - offset, e.a, e.b, e.θ)

# =============================================================================
# Uniform Scaling Operators for Circles and Ellipses
# =============================================================================

"""
    Base.:*(s::Real, c::Circle) -> Circle
    Base.:*(c::Circle, s::Real) -> Circle

Uniformly scale a circle by multiplying its radius by a scale factor.

# Arguments
- `s::Real`: Scale factor (must be positive)
- `c::Circle`: Circle to scale

# Returns
- `Circle`: Scaled circle with same center

# Example
```julia
circle = Circle(Point2(5.0, 3.0), 2.5)
scaled = 2.0 * circle  # Circle at (5.0, 3.0) with radius 5.0
```
"""
Base.:*(s::Real, c::GeometryBasics.Circle) =
    GeometryBasics.Circle(c.center, s * c.r)

Base.:*(c::GeometryBasics.Circle, s::Real) = s * c

"""
    Base.:*(s::Real, e::Ellipse) -> Ellipse
    Base.:*(e::Ellipse, s::Real) -> Ellipse

Uniformly scale an ellipse by multiplying both semi-axes by a scale factor.

The center and orientation angle remain unchanged. This operation preserves
the ellipse's aspect ratio while scaling its size.

# Arguments
- `s::Real`: Scale factor (must be positive)
- `e::Ellipse`: Ellipse to scale

# Returns
- `Ellipse`: Scaled ellipse with same center and orientation

# Example
```julia
ellipse = Ellipse(Point2(10.0, 20.0), 5.0, 3.0, π/4)
scaled = 2.0 * ellipse  # Ellipse at (10.0, 20.0) with a=10.0, b=6.0, θ=π/4

# Useful for patch extraction with context
patch_region = 1.5 * detected_ellipse
transform = patch_to_ellipse_transform(patch_region, 128)
```
"""
Base.:*(s::Real, e::Ellipse{T}) where {T} =
    Ellipse{T}(e.center, s * e.a, s * e.b, e.θ)

Base.:*(e::Ellipse{T}, s::Real) where {T} = s * e

# =============================================================================
# Circle Intersection
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

Uses the `intersects(::Circle, ::Circle)` method.

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
