# =============================================================================
# Random Sampling from Geometric Primitives
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
    # Promote to float type to handle both integer and float rectangles
    FT = float(T)
    random_offset = FT.(rand(rng, N))
    # Result type promotes based on arithmetic (int + float = float)
    return Point{N}(r.origin .+ random_offset .* r.widths)
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
