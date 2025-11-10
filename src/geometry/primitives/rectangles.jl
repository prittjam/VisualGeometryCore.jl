# ========================================================================
# Interval Constructors from Ranges
# ========================================================================

"""
    ClosedInterval(range::AbstractRange; align_corners::Bool=false)
        -> ClosedInterval

Construct a closed interval from a range representing its geometric extent.

# Arguments
- `range::AbstractRange`: The range to convert
- `align_corners::Bool`: Alignment mode (default: false)
  - `true`: Interval endpoints are range endpoints [first, last]
  - `false`: Range values are centers, extend by ±step/2

# Returns
- `ClosedInterval`: Geometric interval [a, b]

# Examples
```julia
# Discrete endpoints (closed interval of range bounds)
r = 1:10
ClosedInterval(r; align_corners=true)   # [1, 10]

# Pixel centers with extent
ClosedInterval(r; align_corners=false)  # [0.5, 10.5]

# Non-unit step
r = 0.0:0.1:1.0
ClosedInterval(r; align_corners=false)  # [-0.05, 1.05]
```
"""
function ClosedInterval(r::AbstractRange; align_corners::Bool=false)
    T = eltype(r)

    if align_corners
        # Interval defined by range endpoints
        return ClosedInterval(T(first(r)), T(last(r)))
    else
        # Range values are centers, extend by half-step
        T_out = T <: Integer ? Float64 : T
        s = T_out(step(r))
        return ClosedInterval(T_out(first(r)) - s/2, T_out(last(r)) + s/2)
    end
end

# ========================================================================
# Rect Constructors from Intervals
# ========================================================================

"""
    Rect(intervals::Tuple{ClosedInterval, ClosedInterval})
        -> GeometryBasics.Rect

Construct a Rect from a tuple of closed intervals.

The intervals directly define the rectangle bounds with no interpretation.

# Arguments
- `intervals`: Tuple of (interval_x, interval_y)

# Returns
- `Rect`: Rectangle with origin at (left, bottom) and widths (Δx, Δy)

# Examples
```julia
using IntervalSets

# From explicit intervals
ix = ClosedInterval(0.5, 640.5)
iy = ClosedInterval(0.5, 480.5)
rect = Rect((ix, iy))  # origin=[0.5, 0.5], widths=[640, 480]

# From ranges via explicit interval conversion
rx, ry = 1:640, 1:480
ix = ClosedInterval(rx; align_corners=false)
iy = ClosedInterval(ry; align_corners=false)
rect = Rect((ix, iy))
```
"""
function GeometryBasics.Rect(intervals::Tuple{ClosedInterval, ClosedInterval})
    ix, iy = intervals
    T = promote_type(eltype(ix), eltype(iy))

    origin = Point2{T}(T(leftendpoint(ix)), T(leftendpoint(iy)))
    widths = Vec2{T}(T(rightendpoint(ix) - leftendpoint(ix)),
                     T(rightendpoint(iy) - leftendpoint(iy)))

    return GeometryBasics.Rect(origin, widths)
end
