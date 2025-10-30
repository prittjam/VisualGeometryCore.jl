# ========================================================================
# Ellipses
# ========================================================================

"""
    Ellipse{T}

Semantic ellipse representation with center, semi-major axis `a`, semi-minor axis `b`, 
and orientation angle `θ` in radians.

The constructor automatically ensures `a ≥ b` by swapping axes and adjusting the angle if needed.

# Fields
- `center::Point2{T}`: Center coordinates (cx, cy)
- `a::T`: Semi-major axis length (always ≥ b)
- `b::T`: Semi-minor axis length (always ≤ a)  
- `θ::T`: Orientation angle in radians [0, π)

# Constructors
```julia
# Basic constructor with Point2 center
ellipse = Ellipse(Point2(1.0, 2.0), 3.0, 2.0, π/4)

# Constructor with vector center (automatically converted to Point2)
ellipse = Ellipse([1.0, 2.0], 3.0, 2.0, π/4)

# Mixed numeric types are automatically promoted
ellipse = Ellipse([1, 2], 3.0f0, 2, π/4)
```

# Integration with GeometryBasics
```julia
# Generate boundary points using GeometryBasics.coordinates
points = GeometryBasics.coordinates(ellipse, 64)

# Access geometric properties
center = GeometryBasics.coordinates(ellipse)  # Returns ellipse.center
major_radius = GeometryBasics.radius(ellipse)  # Returns max(a, b)
```
"""
struct Ellipse{T}
    center::Point2{T}  # (cx, cy)
    a::T               # semimajor (should be >= b)
    b::T               # semiminor (should be <= a)
    θ::T               # radians
    
    # Inner constructor to ensure a >= b (swap if needed)
    function Ellipse{T}(center::Point2{T}, a::T, b::T, θ::T) where {T}
        if a >= b
            new{T}(center, a, b, θ)
        else
            # Swap axes and rotate by π/2
            new{T}(center, b, a, θ + T(π/2))
        end
    end
end

# Outer constructors for convenience
Ellipse(center::Point2{T}, a::T, b::T, θ::T) where {T} = Ellipse{T}(center, a, b, θ)

# Accept Point2 with mixed numeric types and promote
Ellipse(center::Point2{T}, a::Real, b::Real, θ::Real) where {T} = begin
    S = promote_type(T, typeof(a), typeof(b), typeof(θ))
    Ellipse{S}(Point2{S}(center), S(a), S(b), S(θ))
end

# Accept AbstractVector (but not Point2) and convert to Point2
Ellipse(center::AbstractVector{T}, a::T, b::T, θ::T) where {T} = begin
    length(center) == 2 || throw(ArgumentError("Center must be a 2D vector, got length $(length(center))"))
    Ellipse{T}(Point2(center[1], center[2]), a, b, θ)
end

# Accept AbstractVector with mixed types and promote
Ellipse(center::AbstractVector, a::Real, b::Real, θ::Real) = begin
    length(center) == 2 || throw(ArgumentError("Center must be a 2D vector, got length $(length(center))"))
    T = promote_type(eltype(center), typeof(a), typeof(b), typeof(θ))
    Ellipse(Point2{T}(center[1], center[2]), T(a), T(b), T(θ))
end

# Unit circle conic: x² + y² - 1 = 0
const CIRCLE = @SMatrix [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 -1.0]

# ---------- GeometryBasics Interface ----------

# Note: We don't define coordinates(ellipse) without nvertices argument
# because GeometryBasics expects coordinates(primitive, nvertices) for boundary points

"""
    GeometryBasics.radius(e::Ellipse)

Return the major axis radius of the ellipse as required by GeometryBasics interface.
"""
GeometryBasics.radius(e::Ellipse) = max(e.a, e.b)  # Major axis radius

# GeometryBasics coordinates method for ellipse boundary points
# This follows the GeometryBasics convention - decompose will call this automatically
function GeometryBasics.coordinates(e::Ellipse{T}, nvertices=32) where {T}
    # Generate angles for boundary points
    θs = range(zero(T), 2π; length=nvertices+1)[1:end-1]
    unit_circle_points = [@SVector [cos(θ), sin(θ)] for θ in θs]

    # Build transform: scale -> rotate -> translate
    transform = CoordinateTransformations.Translation(e.center) ∘
                CoordinateTransformations.LinearMap(Rotations.RotMatrix{2,T}(e.θ)) ∘
                CoordinateTransformations.LinearMap(@SMatrix [e.a 0; 0 e.b])

    # Apply transform and return as Points
    return [Point{2,T}(transform(p)) for p in unit_circle_points]
end

# Note: GeometryBasics.decompose(PointType, ellipse) is provided automatically
# by GeometryBasics and will call coordinates(ellipse) internally

"""
    Base.extrema(e::Ellipse) -> (Point2, Point2)

Compute the axis-aligned bounding box of the ellipse analytically.

Returns `(min_point, max_point)` where min_point contains the minimum x and y coordinates,
and max_point contains the maximum x and y coordinates.

For a rotated ellipse, the bounding box half-widths from center are:
- rx = sqrt((a*cos(θ))^2 + (b*sin(θ))^2)
- ry = sqrt((a*sin(θ))^2 + (b*cos(θ))^2)

# Example
```julia
ellipse = Ellipse(Point2(100.0, 100.0), 50.0, 30.0, π/4)
min_pt, max_pt = extrema(ellipse)
bbox = Rect2(min_pt, max_pt - min_pt)
```
"""
function Base.extrema(e::Ellipse{T}) where {T}
    # For ellipse: x(t) = cx + a*cos(t)*cos(θ) - b*sin(t)*sin(θ)
    #              y(t) = cy + a*cos(t)*sin(θ) + b*sin(t)*cos(θ)
    #
    # Extrema occur when dx/dt = 0 and dy/dt = 0
    # This gives: rx = sqrt((a*cos(θ))² + (b*sin(θ))²)
    #            ry = sqrt((a*sin(θ))² + (b*cos(θ))²)

    cos_θ = cos(e.θ)
    sin_θ = sin(e.θ)

    rx = sqrt((e.a * cos_θ)^2 + (e.b * sin_θ)^2)
    ry = sqrt((e.a * sin_θ)^2 + (e.b * cos_θ)^2)

    min_pt = Point2{T}(e.center[1] - rx, e.center[2] - ry)
    max_pt = Point2{T}(e.center[1] + rx, e.center[2] + ry)

    return (min_pt, max_pt)
end

# =============================================================================
# Circle Construction from Blobs
# =============================================================================

"""
    GeometryBasics.Circle(blob::AbstractBlob, cutoff::Real) -> Circle

Construct a GeometryBasics.Circle from a blob with radius `cutoff * σ`.

The `cutoff` parameter determines the effective radius as a multiple of σ
(e.g., 3.0 for 3σ radius). Centers and radii are converted to unitless Float64.

# Arguments
- `blob::AbstractBlob`: Blob with center and scale σ
- `cutoff::Real`: Radius multiplier (typically 2-4)

# Returns
- `GeometryBasics.Circle`: Circle with unitless center and radius

# Example
```julia
blob = IsoBlob(Point2(100.0pd, 200.0pd), 5.0pd)
circle = Circle(blob, 3.0)  # Circle with radius 15.0
```
"""
function GeometryBasics.Circle(blob::AbstractBlob, cutoff::Real)
    center = Point2(float.(ustrip.(blob.center))...)
    radius = float(ustrip(cutoff * blob.σ))
    return GeometryBasics.Circle(center, radius)
end

# =============================================================================
# HomEllipseMat Construction (Conic Matrix Representation)
# =============================================================================

"""
    HomEllipseMat(e::Ellipse{T}) -> HomEllipseMat{T}

Construct conic matrix Q from Ellipse via Q = A^{-T} * CIRCLE * A^{-1},
where A maps unit circle → ellipse.

The conic matrix Q represents the implicit equation x^T Q x = 0.

# Example
```julia
ellipse = Ellipse(Point2(1.0, 2.0), 3.0, 2.0, π/4)
Q = HomEllipseMat(ellipse)  # 3×3 conic matrix
H = PlanarHomographyMat(camera)
Q_warped = H(Q)  # Apply conic transformation
ellipse_warped = Ellipse(Q_warped)  # Extract warped ellipse
```
"""
function HomEllipseMat(e::Ellipse{T}) where {T}
    # Build transformation matrix A: translate ∘ rotate ∘ scale (unit circle → ellipse)
    cos_θ = cos(e.θ)
    sin_θ = sin(e.θ)

    # A = T * R * S
    A = @SMatrix [
        e.a * cos_θ   -e.b * sin_θ   e.center[1];
        e.a * sin_θ    e.b * cos_θ   e.center[2];
        T(0)           T(0)           T(1)
    ]

    # Apply conic transformation: Q = A^{-T} * CIRCLE * A^{-1}
    invA = inv(A)
    Q = transpose(invA) * CIRCLE * invA

    return HomEllipseMat{T}(Tuple(Q))
end

"""
    HomCircleMat(c::Circle{T}) -> HomCircleMat{T}

Construct conic matrix Q from Circle via Q = A^{-T} * CIRCLE * A^{-1},
where A maps unit circle → circle.

Returns HomCircleMat{T} (typed as a circle conic).

# Example
```julia
circle = Circle(Point2(5.0, 3.0), 2.5)
Q = HomCircleMat(circle)  # 3×3 circle conic matrix
```
"""
function HomCircleMat(c::Circle{T}) where {T}
    # Build transformation matrix A: translate ∘ uniform scale (unit circle → circle)
    # A = T * S where S = diag(r, r, 1)
    A = @SMatrix [
        c.r     T(0)    c.center[1];
        T(0)    c.r     c.center[2];
        T(0)    T(0)    T(1)
    ]

    # Apply conic transformation: Q = A^{-T} * CIRCLE * A^{-1}
    invA = inv(A)
    Q = transpose(invA) * CIRCLE * invA

    return HomCircleMat{T}(Tuple(Q))
end

# Delegate HomEllipseMat(::Circle) to HomCircleMat for type correctness
HomEllipseMat(c::Circle) = HomCircleMat(c)

"""
    conic_trait(Q::AbstractHomEllipseMat) -> ConicTrait

Determine if a conic matrix represents a circle or ellipse using Holy Traits.

Dispatches on the concrete type:
- HomCircleMat → CircleTrait()
- HomEllipseMat → EllipseTrait()

# Example
```julia
circle = Circle(Point2(5.0, 3.0), 2.5)
Q = HomCircleMat(circle)
conic_trait(Q)  # CircleTrait()

ellipse = Ellipse(Point2(5.0, 3.0), 3.0, 2.0, π/4)
Q = HomEllipseMat(ellipse)
conic_trait(Q)  # EllipseTrait()
```
"""
conic_trait(::HomCircleMat) = CircleTrait()
conic_trait(::HomEllipseMat) = EllipseTrait()

"""
    Ellipse(Q::Union{HomEllipseMat{T}, HomCircleMat{T}}) -> Ellipse{T}

Extract Ellipse from conic matrix Q using robust eigenvalue decomposition
to extract center, orientation, and axes.

Works with both HomEllipseMat and HomCircleMat.
This implementation is fully type-stable using SMatrix and SVector.

# Example
```julia
H = PlanarHomographyMat(camera)
Q = HomCircleMat(circle)
Q_warped = H(Q)  # Apply conic transformation
ellipse = Ellipse(Q_warped)  # Extract ellipse parameters
```
"""
function Ellipse(Q::Union{HomEllipseMat{T}, HomCircleMat{T}}) where {T}
    C = SMatrix{3,3,T}(Q)

    # Extract 2×2 submatrix A and 2-vector b using SMatrix/SVector for type stability
    A = SMatrix{2,2,T}(C[1,1], C[2,1], C[1,2], C[2,2])
    b = SVector{2,T}(C[1,3], C[2,3])

    # Compute center: -A⁻¹b
    center_vec = -(A \ b)

    # Compute constant term after centering: γ = c - b'A⁻¹b = C[3,3] + b'center
    γ = C[3,3] + dot(b, center_vec)

    @assert γ < 0 "Conic does not represent an ellipse (discriminant should be < 0)."

    # Use symmetric eigenvalue decomposition (guarantees real eigenvalues)
    # Force A to be exactly symmetric to avoid numerical issues
    A_sym = (A + transpose(A)) / T(2)
    eigen_result = eigen(Symmetric(A_sym))
    eigenvals = eigen_result.values  # Guaranteed to be Vector{T} (real)
    eigenvecs = eigen_result.vectors

    # Check that A is positive definite
    @assert all(>(zero(T)), eigenvals) "Quadratic form must be positive definite for an ellipse."

    # Compute axis lengths: for centered ellipse x'Ax = -γ
    # In principal coordinates: λ₁u² + λ₂v² = -γ, so axes are √(-γ/λᵢ)
    a1 = sqrt(-γ / eigenvals[1])
    a2 = sqrt(-γ / eigenvals[2])

    # Determine which is the major axis (larger length)
    if a1 >= a2
        a, b_axis = a1, a2
        major_axis_vec = SVector{2,T}(eigenvecs[1,1], eigenvecs[2,1])
    else
        a, b_axis = a2, a1
        major_axis_vec = SVector{2,T}(eigenvecs[1,2], eigenvecs[2,2])
    end

    # Compute orientation angle from major axis direction
    # atan2 gives angle in [-π, π], we want [0, π) since ellipse has period π
    θ = atan(major_axis_vec[2], major_axis_vec[1])
    θ = mod(θ, T(π))  # Normalize to [0, π)

    return Ellipse{T}(Point2{T}(center_vec), a, b_axis, θ)
end

"""
    Circle(Q::Union{HomEllipseMat, HomCircleMat}) -> Circle

Extract circle parameters from conic matrix.

Converts the conic to an Ellipse first, then checks if it's geometrically a circle
(i.e., a ≈ b). If so, returns a Circle with the average of a and b as the radius.

Throws an error if the conic doesn't represent a circle (a and b differ significantly).

# Example
```julia
Q = HomCircleMat(Circle(Point2(1.0, 2.0), 3.0))
circle = Circle(Q)  # Extract circle parameters

# Works for HomEllipseMat too if it's geometrically a circle
H = HomScaleIsoMat(...)  # Isotropic scaling
Q_scaled = H(Q)
circle_scaled = Circle(Q_scaled)
```
"""
function GeometryBasics.Circle(Q::Union{HomEllipseMat{T}, HomCircleMat{T}}; atol=1e-6) where {T}
    # Convert to ellipse first
    ellipse = Ellipse(Q)

    # Check if it's geometrically a circle (a ≈ b)
    if !isapprox(ellipse.a, ellipse.b; atol=atol)
        error("Conic does not represent a circle: a=$(ellipse.a), b=$(ellipse.b) (difference: $(abs(ellipse.a - ellipse.b)))")
    end

    # Return circle with average of a and b as radius
    r = (ellipse.a + ellipse.b) / 2
    return GeometryBasics.Circle(ellipse.center, r)
end

"""
    (H::PlanarHomographyMat)(Q::Union{HomEllipseMat, HomCircleMat}) -> HomEllipseMat

Transform conic by homography: Q' = H^{-T} * Q * H^{-1} (contravariant transformation).

Note: Transforming a circle (HomCircleMat) by a general homography typically produces
an ellipse, so the result is always HomEllipseMat.

# Example
```julia
H = PlanarHomographyMat(camera)
Q = HomCircleMat(circle)
Q_warped = H(Q)  # Apply conic transformation → HomEllipseMat
ellipse = Ellipse(Q_warped)  # Extract warped ellipse
```
"""
function (H::PlanarHomographyMat{T1})(Q::Union{HomEllipseMat{T2}, HomCircleMat{T2}}) where {T1, T2}
    T = promote_type(T1, T2)
    H_mat = SMatrix{3,3,T}(H)
    Q_mat = SMatrix{3,3,T}(Q)

    # Apply conic transformation: Q' = H^{-T} * Q * H^{-1}
    invH = inv(H_mat)
    Q_warped = transpose(invH) * Q_mat * invH

    return HomEllipseMat{T}(Tuple(Q_warped))
end

# =============================================================================
# Gradient Computation
# =============================================================================

"""
    gradient(Q::HomEllipseMat{T}, p::Point2{T}) -> SVector{2, T}

Compute the gradient of the conic implicit function at point p.

For conic equation x^T Q x = 0, the gradient is ∇f(x) = 2Qx (homogeneous form).
Returns the 2D Euclidean gradient [∂f/∂x, ∂f/∂y].

# Example
```julia
ellipse = Ellipse(Point2(0.0, 0.0), 3.0, 2.0, 0.0)
Q = HomEllipseMat(ellipse)
grad = gradient(Q, ellipse.center)  # Should be ≈ [0, 0] at center
```
"""
function gradient(Q::HomEllipseMat{T}, p::Point2{T}) where {T}
    # Convert point to homogeneous coordinates
    x_h = SVector{3,T}(p[1], p[2], one(T))

    # Compute gradient: ∇f = 2Qx
    Q_mat = SMatrix{3,3,T}(Q)
    grad_h = 2 * Q_mat * x_h

    # Return Euclidean part (first 2 components)
    return SVector{2,T}(grad_h[1], grad_h[2])
end

# Allow mixed types with automatic promotion
function gradient(Q::HomEllipseMat{T1}, p::Point2{T2}) where {T1, T2}
    T = promote_type(T1, T2)
    return gradient(HomEllipseMat{T}(Q.data), Point2{T}(p))
end

# Allow AbstractVector (e.g., SVector) as input
function gradient(Q::HomEllipseMat{T}, p::AbstractVector) where {T}
    length(p) == 2 || throw(ArgumentError("Point must be 2D, got length $(length(p))"))
    return gradient(Q, Point2{T}(p[1], p[2]))
end

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