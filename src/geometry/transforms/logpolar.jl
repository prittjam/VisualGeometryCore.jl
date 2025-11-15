"""
    logpolar_to_cartesian(; r_min=0.01)

Create a transformation from log-polar to Cartesian coordinates in normalized unit space.

Maps points in unit space [-1,1]² interpreted as (log_r, θ) to Cartesian (x, y) in unit space.
This is a pure coordinate transformation that operates entirely in normalized space.

# Coordinate Convention
- x-axis (horizontal/columns) = scale (radius)
- y-axis (vertical/rows) = rotation (angle)

# Mapping
- Input x-coordinate in [-1,1] → radius from r_min to 1.0 (logarithmically)
- Input y-coordinate in [-1,1] → angle from 0 to 2π
- Output: Cartesian (x, y) in unit circle (radius ≤ 1.0)

# Arguments
- `r_min::Real`: Minimum radius (default: 0.01, avoids singularity at origin)

# Example
```julia
transform = logpolar_to_cartesian()
# Maps log-polar space to unit circle
transform(Point2(0.0, 0.0))  # Center → mid radius, π angle
transform(Point2(-1.0, -1.0))  # Bottom-left → min radius, 0°
transform(Point2(1.0, 1.0))  # Top-right → max radius (1.0), 2π
```
"""
function logpolar_to_cartesian(r_min::Real=0.01)
    T = Float64
    log_r_min = log(T(r_min))
    log_r_max = log(T(1.0))  # Always map to unit circle

    # Return transformation function
    function transform(p)
        θ = π * (p[2] + 1)  # Maps [-1,1] → [0, 2π]
        normalized_log_r = (p[1] + 1) / 2  # Maps [-1,1] → [0,1]
        log_r = log_r_min + (log_r_max - log_r_min) * normalized_log_r
        r = exp(log_r)

        p2 = r * Point2(cos(θ), sin(θ))

        return p2
    end

    return transform
end

"""
    logpolar_map(geometry::Union{Circle, Ellipse}, logpolar_rect::HyperRectangle{2}, r_min::Real) -> Function

Create a composed transformation that maps from log-polar patch coordinates to image coordinates
via a circle or ellipse.

This is a convenience function that composes three transformations:
1. `logpolar_rect` → `CANONICAL_SQUARE` (normalize patch coordinates)
2. `CANONICAL_SQUARE` → `UNIT_CIRCLE` via log-polar transform
3. `UNIT_CIRCLE` → `geometry` (map to target circle/ellipse in image)

The resulting transformation can be used with `warp` to extract log-polar patches.

# Arguments
- `geometry::Union{Circle, Ellipse}`: Target geometry in image coordinates
- `logpolar_rect::HyperRectangle{2}`: Rectangle defining the log-polar patch domain
- `r_min::Real`: Minimum radius for log-polar transform (avoids singularity at origin)

# Returns
- `Function`: Composed transformation `logpolar_rect → geometry`

# Broadcasting
This function is designed to be broadcastable across multiple geometries:
```julia
# Extract patches from multiple circles
transforms = logpolar_map.(circles, Ref(logpolar_rect), Ref(r_min))
```

# Example
```julia
using VisualGeometryCore
using GeometryBasics: Circle, Point2
using ImageTransformations: warp
using Interpolations

# Define target circle and log-polar patch size
circle = Circle(Point2(100.0, 200.0), 50.0)
logpolar_rect = Rect((1..256, 1..256))
r_min = 0.01

# Create transformation
transform = logpolar_map(circle, logpolar_rect, r_min)

# Use with warp to extract patch
itp = interpolate(image, BSpline(Cubic(Line(OnGrid()))))
etp = extrapolate(itp, Flat())
patch = warp(etp, p->reverse(transform(reverse(p))), (1:256, 1:256))
```

# See Also
- [`coord_map`](@ref): Generic coordinate mapping between geometries
- [`logpolar_to_cartesian`](@ref): Log-polar to Cartesian transformation
"""
function logpolar_map(geometry::Union{Circle, Ellipse}, logpolar_rect::HyperRectangle{2}, r_min::Real)
    return coord_map(UNIT_CIRCLE, geometry) ∘
           logpolar_to_cartesian(r_min) ∘
           coord_map(logpolar_rect, CANONICAL_SQUARE)
end


function canonical_map(geometry::Union{Circle, Ellipse}, canonical_rect::HyperRectangle{2})
    return coord_map(UNIT_CIRCLE, geometry) ∘
        coord_map(canonical_rect, CANONICAL_SQUARE)
end

"""
    logpolar_patch_map(geometry; patch_size=256, r_min=0.01, image_origin=:julia) -> (transform, axes)

Construct a log-polar transformation and axes for patch extraction.

This function sets up the coordinate system and transformation, but does NOT perform
the actual warping. This allows efficient reuse of interpolators when extracting
multiple patches from the same image.

# Returns
- `transform`: Transformation function from logpolar_rect → geometry
- `axes`: Tuple of axes (1:patch_size, 1:patch_size) for warping

# Example
```julia
# Setup interpolator once
itp = interpolate(image, BSpline(Cubic(Line(OnGrid()))))
etp = extrapolate(itp, Gray(1.0))

# Extract patches from multiple geometries
for geom in geometries
    transform, axes = logpolar_patch_map(geom; patch_size=256, r_min=0.01)
    patch = warp(etp, imgmap(transform), axes)
end
```
"""
function logpolar_patch_map(geometry::Union{Circle, Ellipse};
                             patch_size::Integer=256,
                             r_min::Real=0.01,
                             image_origin::Symbol=:julia)
    # Define patch coordinate system
    offset = image_origin_offset(from=:julia, to=image_origin)
    logpolar_axes = (1:patch_size, 1:patch_size)
    ix = ClosedInterval(logpolar_axes[1] .+ ustrip(offset[1]); align_corners=false)
    iy = ClosedInterval(logpolar_axes[2] .+ ustrip(offset[2]); align_corners=false)
    logpolar_rect = Rect((ix, iy))

    # Create transformation
    transform = logpolar_map(geometry, logpolar_rect, r_min)

    return (transform, logpolar_axes)
end

"""
    canonical_patch_map(geometry; patch_size=256, image_origin=:julia) -> (transform, axes)

Construct a canonical transformation and axes for patch extraction.

This function sets up the coordinate system and transformation, but does NOT perform
the actual warping. This allows efficient reuse of interpolators when extracting
multiple patches from the same image.

# Returns
- `transform`: Transformation function from canonical_rect → geometry
- `axes`: Tuple of axes (1:patch_size, 1:patch_size) for warping

# Example
```julia
# Setup interpolator once
itp = interpolate(image, BSpline(Cubic(Line(OnGrid()))))
etp = extrapolate(itp, Gray(1.0))

# Extract patches from multiple geometries
for geom in geometries
    transform, axes = canonical_patch_map(geom; patch_size=256)
    patch = warp(etp, imgmap(transform), axes)
end
```
"""
function canonical_patch_map(geometry::Union{Circle, Ellipse};
                             patch_size::Integer=256,
                             image_origin::Symbol=:julia)
    # Define patch coordinate system
    offset = ustrip.(image_origin_offset(from=:julia, to=image_origin))
    canonical_axes = (1:patch_size, 1:patch_size)
    ix = ClosedInterval(canonical_axes[1] .+ offset[1]; align_corners=false)
    iy = ClosedInterval(canonical_axes[2] .+ offset[2]; align_corners=false)
    canonical_rect = Rect((ix, iy))

    # Create transformation
    transform = canonical_map(geometry, canonical_rect)

    return (transform, canonical_axes)
end
