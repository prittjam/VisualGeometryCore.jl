# =============================================================================
# Ellipse Type and Conic Matrix Representation
# =============================================================================


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

"""
    GeometryBasics.origin(e::Ellipse) -> Point2

Get the center of an ellipse as its origin, making it compatible with generic
GeometryBasics primitives for coordinate convention conversion and transformations.

# Example
```julia
ellipse = Ellipse(Point2(100.0, 150.0), 30.0, 20.0, π/4)
pos = origin(ellipse)  # Returns Point2(100.0, 150.0)
```
"""
GeometryBasics.origin(e::Ellipse) = e.center

"""
    GeometryBasics.origin(c::Circle) -> Point2

Get the center of a circle as its origin, making it compatible with generic
GeometryBasics primitives for coordinate convention conversion and transformations.

# Example
```julia
circle = Circle(Point2(50.0, 75.0), 10.0)
pos = origin(circle)  # Returns Point2(50.0, 75.0)
```
"""
GeometryBasics.origin(c::GeometryBasics.Circle) = c.center

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
    HomCircleMat(c::Circle{T}) -> HomCircleMat

Construct conic matrix Q from Circle via Q = A^{-T} * CIRCLE * A^{-1},
where A maps unit circle → circle.

Units are stripped from the circle to create a unitless conic matrix.

# Example
```julia
circle = Circle(Point2(5.0, 3.0), 2.5)
Q = HomCircleMat(circle)  # 3×3 circle conic matrix
```
"""
function HomCircleMat(c::Circle{T}) where {T}
    # Strip units for conic matrix operations (inv creates inverse units)
    r = float(ustrip(c.r))
    cx, cy = float.(ustrip.(c.center))
    U = typeof(r)

    # Build transformation matrix A: translate ∘ uniform scale (unit circle → circle)
    # A = T * S where S = diag(r, r, 1)
    A = @SMatrix [
        r       U(0)    cx;
        U(0)    r       cy;
        U(0)    U(0)    U(1)
    ]

    # Apply conic transformation: Q = A^{-T} * CIRCLE * A^{-1}
    invA = inv(A)
    Q = transpose(invA) * CIRCLE * invA

    return HomCircleMat{U}(Tuple(Q))
end

"""
    is_ellipse(Q::Union{HomEllipseMat, HomCircleMat}) -> Bool

Check if a conic matrix represents a valid ellipse without constructing it.

This function is **scale- and sign-invariant**: homogeneous conic matrices are
defined only up to a nonzero scalar multiplier λ, so both C and λC represent
the same geometric curve. The implementation normalizes the matrix by its
Frobenius norm before checking, ensuring consistent classification for ellipses
of any size, from microscopic to astronomical.

Returns `true` if the conic satisfies the conditions for an ellipse:
1. Matrix C is normalized: C ← C/||C||_F (enables uniform scale-invariant checks)
2. Quadratic form A is definite (not indefinite - hyperbolas are rejected)
3. A is non-degenerate: condition number |λ_min/λ_max| >= tol (scale-invariant)
4. A is normalized to be positive definite (flip sign if needed)
5. Discriminant γ < -tol on normalized matrix (ellipse criterion)

All checks use relative tolerances, making the function work correctly for
ellipses ranging from sub-pixel precision (a ~ 1e-5) to mega-pixel scale (a ~ 1e6+).

# Examples
```julia
# Filter valid ellipses from warped conics
valid_conics = filter(is_ellipse, Q_warped)
ellipses = Ellipse.(valid_conics)

# Works regardless of sign or scale
Q1 = HomEllipseMat(ellipse)
Q2 = HomEllipseMat{Float64}(Tuple(-1.0 .* SMatrix{3,3}(Q1)))
Q3 = HomEllipseMat{Float64}(Tuple(1e6 .* SMatrix{3,3}(Q1)))
is_ellipse(Q1) == is_ellipse(Q2) == is_ellipse(Q3)  # true

# Works for ellipses of any size
small = Ellipse(Point2(0.0, 0.0), 1e-6, 5e-7, 0.0)
large = Ellipse(Point2(0.0, 0.0), 1e6, 5e5, 0.0)
is_ellipse(HomEllipseMat(small)) && is_ellipse(HomEllipseMat(large))  # true
```
"""
function is_ellipse(Q::Union{HomEllipseMat{T}, HomCircleMat{T}}) where {T}
    # Symmetrize for numerical robustness
    C_raw = SMatrix{3,3,T}(Q)
    C_sym = (C_raw + transpose(C_raw)) / T(2)

    # Normalize by Frobenius norm for scale-invariant checks
    # This ensures all subsequent checks work uniformly across scales
    frobenius_norm = sqrt(sum(abs2, C_sym))
    C = C_sym / frobenius_norm

    # Extract 2×2 submatrix A, 2-vector b, and scalar c from C = [A b; b' c]
    A = SMatrix{2,2,T}(C[1,1], C[2,1], C[1,2], C[2,2])
    b = SVector{2,T}(C[1,3], C[2,3])
    c = C[3,3]

    # Compute eigenvalues of A to determine definiteness
    eigen_result = eigen(Symmetric(A))
    eigenvals = eigen_result.values
    λ_min, λ_max = extrema(eigenvals)

    # Tolerance for relative checks (scale-invariant)
    # Use smaller tolerance to support sub-pixel precision ellipses (a ~ 1e-5)
    tol = eps(T)^(T(3)/T(4))  # ≈ 3.7e-12 for Float64

    # Determine definiteness of A:
    # - If both eigenvalues have same sign: definite (ellipse or degenerate ellipse)
    # - If eigenvalues have opposite signs: indefinite (hyperbola)
    if λ_min * λ_max < zero(T)
        # Indefinite: hyperbola
        return false
    end

    # Check for degeneracy using relative tolerance (condition number)
    # This is scale-invariant: checks if A is nearly rank-deficient
    if abs(λ_min / λ_max) < tol
        return false
    end

    # Normalize sign: ensure A is positive definite
    # If both eigenvalues are negative, multiply entire matrix by -1
    if λ_max < zero(T)
        # A is negative definite, flip sign
        C = -C
        A = -A
        b = -b
        c = -c
        # Flip eigenvalues too
        λ_min, λ_max = -λ_max, -λ_min
    end

    # At this point, A is positive definite (both eigenvalues > 0)
    # Compute center: x₀ = -A⁻¹b
    center_vec = -(A \ b)

    # Compute discriminant after centering: γ = c - b'A⁻¹b = c + b'x₀
    γ = c + dot(b, center_vec)

    # For a real, non-degenerate ellipse: γ < 0 when A is positive definite
    # Since C is normalized (||C||_F = 1), we can use absolute tolerance
    # This works uniformly for ellipses of any size
    return γ < -tol
end

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
    # Runtime check using is_ellipse predicate
    @assert is_ellipse(Q) "Conic does not represent a valid ellipse (failed discriminant or positive definite check)."

    C = SMatrix{3,3,T}(Q)

    # Extract 2×2 submatrix A and 2-vector b using SMatrix/SVector for type stability
    A = SMatrix{2,2,T}(C[1,1], C[2,1], C[1,2], C[2,2])
    b = SVector{2,T}(C[1,3], C[2,3])

    # Compute center: -A⁻¹b
    center_vec = -(A \ b)

    # Compute constant term after centering: γ = c - b'A⁻¹b = C[3,3] + b'center
    γ = C[3,3] + dot(b, center_vec)

    # Use symmetric eigenvalue decomposition (guarantees real eigenvalues)
    # Force A to be exactly symmetric to avoid numerical issues
    A_sym = (A + transpose(A)) / T(2)
    eigen_result = eigen(Symmetric(A_sym))
    eigenvals = eigen_result.values  # Guaranteed to be Vector{T} (real)
    eigenvecs = eigen_result.vectors

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
