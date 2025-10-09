# ========================================================================
# Conics and Ellipses
# ========================================================================

# Use the same efficient staticmat3 macro for conics
@staticmat3 HomogeneousConic  # Homogeneous (3×3) conic: x' C x = 0

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

# ---------- Transforming conics ----------
#
# Conics can now be transformed using the natural function call syntax:
#   transformed_conic = transform(conic)
#
# This works with:
# - CoordinateTransformations: Translation, LinearMap, AffineMap, composed transforms
# - Rotations: Any Rotation type
# - Homogeneous matrices: HomRotMat, HomTransMat, etc.
#
# Examples:
#   Q_translated = Translation(SVector(2.0, 3.0))(Q)
#   Q_rotated = RotMatrix{2}(π/4)(Q)
#   Q_transformed = (translation ∘ rotation ∘ scaling)(Q)
#

"""
    (A::SMatrix{3,3,T})(Q::HomogeneousConic{T}) -> HomogeneousConic{T}

Transform a conic by homogeneous 3×3 matrix A: Q' = A^(-T) * Q * A^(-1).
This is the contravariant transformation for conics.
"""
function (A::SMatrix{3,3,T})(Q::HomogeneousConic{T}) where {T}
    invA = inv(A)
    Q_mat = SMatrix{3,3,T}(Q)
    return HomogeneousConic{T}(Tuple(transpose(invA) * Q_mat * invA))
end

"""
    (A::HomMatAny)(Q::HomogeneousConic{T}) -> HomogeneousConic{T}

Transform a conic by typed homogeneous wrapper: Q' = A^(-T) * Q * A^(-1).
"""
function (A::HomMatAny)(Q::HomogeneousConic{T}) where {T}
    A_mat = SMatrix{3,3,T}(Tuple(A))
    return A_mat(Q)
end

"""
    (tf::CoordinateTransformations.Translation)(Q::HomogeneousConic{T}) -> HomogeneousConic{T}

Transform a conic by Translation.
"""
function (tf::CoordinateTransformations.Translation)(Q::HomogeneousConic{T}) where {T}
    A = to_homogeneous(tf)
    return A(Q)
end

"""
    (tf::CoordinateTransformations.LinearMap)(Q::HomogeneousConic{T}) -> HomogeneousConic{T}

Transform a conic by LinearMap.
"""
function (tf::CoordinateTransformations.LinearMap)(Q::HomogeneousConic{T}) where {T}
    A = to_homogeneous(tf)
    return A(Q)
end

"""
    (tf::CoordinateTransformations.AffineMap)(Q::HomogeneousConic{T}) -> HomogeneousConic{T}

Transform a conic by AffineMap.
"""
function (tf::CoordinateTransformations.AffineMap)(Q::HomogeneousConic{T}) where {T}
    A = to_homogeneous(tf)
    return A(Q)
end

"""
    (tf::ComposedFunction)(Q::HomogeneousConic{T}) -> HomogeneousConic{T}

Transform a conic by composed transformation.
"""
function (tf::ComposedFunction)(Q::HomogeneousConic{T}) where {T}
    A = to_homogeneous(tf)
    return A(Q)
end

"""
    (R::Rotations.Rotation)(Q::HomogeneousConic{T}) -> HomogeneousConic{T}

Transform a conic by Rotations.Rotation.
"""
function (R::Rotations.Rotation)(Q::HomogeneousConic{T}) where {T}
    A = to_homogeneous(R)
    return A(Q)
end

# Backward compatibility functions (deprecated)
push_conic(A::SMatrix{3,3,T}, Q::HomogeneousConic{T}) where {T} = A(Q)
push_conic(A::HomMatAny, Q::HomogeneousConic{T}) where {T} = A(Q)
push_conic(tf, Q::HomogeneousConic{T}) where {T} = tf(Q)

# ---------- Ellipse ⇄ Conic constructors ----------

"""
    HomogeneousConic(e::Ellipse{T}) -> HomogeneousConic{T}

Construct HomogeneousConic from Ellipse via Q' = A^(-T) * CIRCLE * A^(-1), 
where A maps unit circle → ellipse.
"""
function HomogeneousConic(e::Ellipse{T}) where {T}
    # Build transform: translate ∘ rotate ∘ scale (unit circle → ellipse)
    transform = CoordinateTransformations.Translation(e.center) ∘ 
                CoordinateTransformations.LinearMap(Rotations.RotMatrix{2,T}(e.θ)) ∘ 
                CoordinateTransformations.LinearMap(@SMatrix [e.a 0; 0 e.b])
    
    # Use the new operator syntax!
    return transform(HomogeneousConic{T}(Tuple(CIRCLE)))
end

"""
    HomogeneousConic(c::Circle{T}) -> HomogeneousConic{T}

Construct HomogeneousConic from GeometryBasics Circle via Q' = A^(-T) * CIRCLE * A^(-1), 
where A maps unit circle → circle.
"""
function HomogeneousConic(c::Circle{T}) where {T}
    # Build transform: translate ∘ scale (unit circle → circle)
    # For a circle, we only need translation and uniform scaling (no rotation)
    transform = CoordinateTransformations.Translation(c.center) ∘ 
                CoordinateTransformations.LinearMap(@SMatrix [c.r 0; 0 c.r])
    
    # Use the new operator syntax!
    return transform(HomogeneousConic{T}(Tuple(CIRCLE)))
end

"""
    Ellipse(Q::HomogeneousConic{T}) -> Ellipse{T}

Construct Ellipse from HomogeneousConic using robust eigenvalue decomposition 
to extract center, orientation, and axes.

This method uses eigenvalue decomposition to robustly identify the major axis
and handle numerical precision issues.
"""
function Ellipse(Q::HomogeneousConic{T}) where {T}
    C = SMatrix{3,3,T}(Q)
    
    # Extract center: for conic [A b; b' c], center is at -A⁻¹b
    A = C[1:2, 1:2]
    b = C[1:2, 3]
    center = -A \ b
    
    # Use the standard approach: diagonalize A and compute axes from eigenvalues
    # For an ellipse x'Ax + 2b'x + c = 0, after centering we get x'Ax + (c - b'A⁻¹b) = 0
    # The constant term after centering is: γ = c - b'A⁻¹b = C[3,3] - b'(A⁻¹b) = C[3,3] + b'center
    γ = C[3,3] + dot(b, center)
    
    @assert γ < 0 "Conic does not represent an ellipse (discriminant should be < 0)."
    
    # Check that A is positive definite (handle numerical precision issues)
    eigenvals_A = eigvals(A)
    
    # For symmetric matrices, eigenvalues should be real, but may have tiny imaginary parts due to numerical errors
    max_imag_part = maximum(abs.(imag.(eigenvals_A)))
    if max_imag_part > 1e-12
        error("Quadratic form has significant complex eigenvalues - not a valid ellipse.")
    end
    
    real_eigenvals = real.(eigenvals_A)
    @assert all(>(zero(T)), real_eigenvals) "Quadratic form must be positive definite for an ellipse."
    
    # Use symmetric eigenvalue decomposition for better numerical stability
    # Force A to be exactly symmetric to avoid numerical issues
    A_sym = (A + transpose(A)) / 2
    eigen_result = eigen(Symmetric(A_sym))
    eigenvals = eigen_result.values  # These will be real for symmetric matrices
    eigenvecs = eigen_result.vectors
    
    # Compute axis lengths: for centered ellipse x'Ax = -γ
    # In principal coordinates: λ₁u² + λ₂v² = -γ, so axes are √(-γ/λᵢ)
    axis_lengths = sqrt.(-γ ./ eigenvals)
    
    # Determine which is the major axis (larger length)
    if axis_lengths[1] >= axis_lengths[2]
        # First eigenvalue corresponds to major axis
        a, b = axis_lengths[1], axis_lengths[2]
        major_axis_vec = eigenvecs[:, 1]
    else
        # Second eigenvalue corresponds to major axis
        a, b = axis_lengths[2], axis_lengths[1]
        major_axis_vec = eigenvecs[:, 2]
    end
    
    # Compute orientation angle from major axis direction
    # atan2 gives angle in [-π, π], we want [0, π) since ellipse has period π
    θ = atan(major_axis_vec[2], major_axis_vec[1])
    θ = mod(θ, T(π))  # Normalize to [0, π)
    
    return Ellipse{T}(Point2{T}(center), a, b, θ)
end

"""
    Circle(Q::HomogeneousConic{T}) -> Circle{T}

Construct Circle from HomogeneousConic if the conic represents a circle 
(i.e., when the ellipse has equal semi-axes).

Throws an error if the conic does not represent a circle within numerical tolerance.
"""
function Circle(Q::HomogeneousConic{T}) where {T}
    # First convert to ellipse
    ellipse = Ellipse(Q)
    
    # Check if it's actually a circle (a ≈ b)
    if !isapprox(ellipse.a, ellipse.b, rtol=1e-10)
        error("HomogeneousConic does not represent a circle: semi-axes are $(ellipse.a) and $(ellipse.b)")
    end
    
    # Return circle with average radius
    radius = (ellipse.a + ellipse.b) / 2
    return Circle(ellipse.center, radius)
end

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

# ---------- Utilities ----------

"""
    gradient(u::Point2{T}, Q::HomogeneousConic{T}) -> Point2{T}

Compute ∇(x' C x) at euclidean 2D point u.
"""
gradient(u::Point2{T}, Q::HomogeneousConic{T}) where {T} = begin
    grad_vec = (2T(1) .* (SMatrix{3,3,T}(Q) * to_homogeneous(u)))[1:2]
    return Point2{T}(grad_vec[1], grad_vec[2])
end