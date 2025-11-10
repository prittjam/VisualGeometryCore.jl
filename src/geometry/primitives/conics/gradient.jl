
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
