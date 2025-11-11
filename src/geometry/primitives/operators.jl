
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

Uniformly scale a circle by multiplying both its center position and radius.

# Arguments
- `s::Real`: Scale factor (must be positive)
- `c::Circle`: Circle to scale

# Returns
- `Circle`: Scaled circle with both position and size scaled

# Example
```julia
circle = Circle(Point2(5.0, 3.0), 2.5)
scaled = 2.0 * circle  # Circle at (10.0, 6.0) with radius 5.0
```

See also: [`dilate`](@ref) to scale only radius while keeping center fixed.
"""
Base.:*(s::Real, c::GeometryBasics.Circle) =
    GeometryBasics.Circle(s * c.center, s * c.r)

Base.:*(c::GeometryBasics.Circle, s::Real) = s * c

"""
    Base.:*(s::Real, e::Ellipse) -> Ellipse
    Base.:*(e::Ellipse, s::Real) -> Ellipse

Uniformly scale an ellipse by multiplying center position and both semi-axes.

The orientation angle remains unchanged.

# Arguments
- `s::Real`: Scale factor (must be positive)
- `e::Ellipse`: Ellipse to scale

# Returns
- `Ellipse`: Scaled ellipse with both position and size scaled

# Example
```julia
ellipse = Ellipse(Point2(10.0, 20.0), 5.0, 3.0, π/4)
scaled = 2.0 * ellipse  # Ellipse at (20.0, 40.0) with a=10.0, b=6.0, θ=π/4
```

See also: [`dilate`](@ref) to scale only axes while keeping center fixed.
"""
Base.:*(s::Real, e::Ellipse{T}) where {T} =
    Ellipse{T}(s * e.center, s * e.a, s * e.b, e.θ)

Base.:*(e::Ellipse{T}, s::Real) where {T} = s * e

# =============================================================================
# Dilation Operators for Circles and Ellipses
# =============================================================================

"""
    dilate(c::Circle, factor::Real) -> Circle

Dilate a circle by scaling its radius while keeping the center fixed.

# Arguments
- `c::Circle`: Circle to dilate
- `factor::Real`: Dilation factor (must be positive)

# Returns
- `Circle`: Dilated circle with same center but scaled radius

# Example
```julia
circle = Circle(Point2(5.0, 3.0), 2.5)
dilated = dilate(circle, 2.0)  # Circle at (5.0, 3.0) with radius 5.0

# Useful for patch extraction with context
patch_region = dilate(detected_circle, 3.0)
```

See also: [`*`](@ref) to scale both position and radius.
"""
dilate(c::GeometryBasics.Circle, factor::Real) =
    GeometryBasics.Circle(c.center, factor * c.r)

"""
    dilate(e::Ellipse, factor::Real) -> Ellipse

Dilate an ellipse by scaling both semi-axes while keeping center and orientation fixed.

Preserves the ellipse's aspect ratio and orientation while scaling its size.

# Arguments
- `e::Ellipse`: Ellipse to dilate
- `factor::Real`: Dilation factor (must be positive)

# Returns
- `Ellipse`: Dilated ellipse with same center and orientation but scaled axes

# Example
```julia
ellipse = Ellipse(Point2(10.0, 20.0), 5.0, 3.0, π/4)
dilated = dilate(ellipse, 1.5)  # Ellipse at (10.0, 20.0) with a=7.5, b=4.5, θ=π/4

# Useful for patch extraction with context
patch_region = dilate(detected_ellipse, 1.5)
transform = patch_to_ellipse_transform(patch_region, 128)
```

See also: [`*`](@ref) to scale both position and axes.
"""
dilate(e::Ellipse{T}, factor::Real) where {T} =
    Ellipse{T}(e.center, factor * e.a, factor * e.b, e.θ)

# =============================================================================
# Intersection Symmetrization
# =============================================================================

"""
    _should_swap_intersects(::Type{T1}, ::Type{T2}) -> Bool

Trait function to determine canonical argument order for `intersects`.
Returns `true` if arguments should be swapped to canonical order.

Override this for new types to enable automatic symmetrization.
Default: use objectid for deterministic type-stable ordering.
"""
_should_swap_intersects(::Type{T1}, ::Type{T2}) where {T1, T2} =
    objectid(T1) > objectid(T2)

# Define canonical orderings for geometry types
_should_swap_intersects(::Type{<:Rect}, ::Type{<:Ellipse}) = true
_should_swap_intersects(::Type{<:Rect}, ::Type{<:GeometryBasics.Circle}) = true

"""
    intersects(a, b)

Generic fallback for automatic symmetrization of `intersects`.
If called with types in non-canonical order, swaps and retries.
"""
function intersects(a::T1, b::T2) where {T1, T2}
    if _should_swap_intersects(T1, T2)
        return intersects(b, a)
    else
        # Canonical order but no method - let Julia's dispatch handle the error
        throw(MethodError(intersects, (a, b)))
    end
end

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

# =============================================================================
# Ellipse-Rectangle Intersection
# =============================================================================

"""
    intersects(e::Ellipse, r::Rect) -> Bool

Check if an ellipse intersects with a rectangle.

Uses an affine transformation approach:
1. Quick rejection via bounding box overlap check
2. Transform to canonical space where ellipse becomes unit circle
3. Test if transformed rectangle intersects the unit circle

This gives accurate results by checking:
- If any rectangle corner is inside the unit circle
- If any rectangle edge intersects the unit circle

# Example
```julia
rect = Rect2(0.0, 0.0, 200.0, 200.0)
ellipse1 = Ellipse(Point2(100.0, 100.0), 30.0, 20.0, π/4)
ellipse2 = Ellipse(Point2(250.0, 100.0), 30.0, 20.0, 0.0)

intersects(ellipse1, rect)  # true - center inside
intersects(ellipse2, rect)  # true - partially overlaps
```
"""
function intersects(e::Ellipse, r::Rect)
    min_e, max_e = extrema(e)
    min_r = minimum(r)
    max_r = maximum(r)

    # Quick rejection: check if bounding boxes overlap
    if !(min_e[1] <= max_r[1] && max_e[1] >= min_r[1] &&
         min_e[2] <= max_r[2] && max_e[2] >= min_r[2])
        return false
    end

    # Build affine transformation: ellipse → unit circle
    T = coord_map(e, UNIT_CIRCLE)

    # Get the 4 corners of the rectangle and transform them
    corners = GeometryBasics.coordinates(r)
    transformed_corners = [Tuple(T(c)) for c in corners]

    # Check if any corner is inside the unit circle
    for (x, y) in transformed_corners
        if x^2 + y^2 <= 1.0
            return true
        end
    end

    # Check if any edge intersects the unit circle
    for i in 1:4
        p1 = transformed_corners[i]
        p2 = transformed_corners[i % 4 + 1]  # Next corner (wraps around)

        if _line_segment_intersects_unit_circle(p1, p2)
            return true
        end
    end

    return false
end

"""
    _line_segment_intersects_unit_circle(p1, p2) -> Bool

Check if a line segment from p1 to p2 intersects the unit circle centered at origin.
Uses the geometric distance from origin to line segment.
"""
function _line_segment_intersects_unit_circle(p1::Tuple{T,T}, p2::Tuple{T,T}) where T
    x1, y1 = p1
    x2, y2 = p2

    # Vector from p1 to p2
    dx = x2 - x1
    dy = y2 - y1

    # Compute parameter t for closest point on line segment to origin
    # Closest point: p1 + t*(p2-p1), where t is clamped to [0,1]
    t = -(x1*dx + y1*dy) / (dx^2 + dy^2)
    t = clamp(t, 0.0, 1.0)

    # Closest point on segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    # Check if closest point is inside unit circle
    return closest_x^2 + closest_y^2 <= 1.0
end
