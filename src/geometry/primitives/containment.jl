
# =============================================================================
# Containment (Base.in extensions)
# =============================================================================

"""
    Base.in(inner::Rect, outer::Rect) -> Bool

Check if inner rectangle is completely contained within outer rectangle.

A rectangle is considered inside another if both its minimum and maximum corners
are contained within the outer rectangle.

# Example
```julia
outer = Rect2(0.0, 0.0, 100.0, 100.0)
inner = Rect2(10.0, 10.0, 50.0, 50.0)
inner in outer  # true
```
"""
function Base.in(inner::Rect, outer::Rect)
    min_inner = minimum(inner)
    max_inner = maximum(inner)
    return min_inner in outer && max_inner in outer
end

"""
    Base.in(e::Ellipse, r::Rect) -> Bool

Check if ellipse's axis-aligned bounding box is completely contained within rectangle.

Uses the analytical `extrema(e)` to compute the ellipse's bounding box efficiently,
then checks if that bounding box is inside the rectangle.

# Example
```julia
rect = Rect2(0.0, 0.0, 200.0, 200.0)
ellipse = Ellipse(Point2(100.0, 100.0), 30.0, 20.0, π/4)
ellipse in rect  # true if bounding box fits
```
"""
function Base.in(e::Ellipse, r::Rect)
    min_pt, max_pt = extrema(e)
    return min_pt in r && max_pt in r
end
