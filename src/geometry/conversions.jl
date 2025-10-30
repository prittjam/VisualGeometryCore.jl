# ========================================================================
# Coordinate Conversions
# ========================================================================

"""
    to_homogeneous(M::SMatrix{2,2,T}, t::SVector{2,T}) -> SMatrix{3,3,T}

Convert 2×2 linear transformation matrix and 2D translation vector to 3×3 homogeneous matrix.

# Arguments
- `M::SMatrix{2,2,T}`: 2×2 linear transformation matrix
- `t::SVector{2,T}`: 2D translation vector

# Returns
- `SMatrix{3,3,T}`: 3×3 homogeneous transformation matrix

# Example
```julia
M = @SMatrix [2.0 0.0; 0.0 1.5]  # scaling
t = SVector(1.0, 2.0)            # translation
H = to_homogeneous(M, t)         # 3×3 homogeneous matrix
```
"""
@inline to_homogeneous(M::SMatrix{2,2,T}, t::SVector{2,T}) where {T} = 
    @SMatrix [M[1,1] M[1,2] t[1]
              M[2,1] M[2,2] t[2]
              zero(T) zero(T) one(T)]

# -------------------------------------------------
# Bridge: CoordinateTransformations / Rotations -> typed homogeneous
# -------------------------------------------------

# primitives
to_homogeneous(R::Rotations.Rotation{2,T}) where {T} =
    HomRotMat{T}(Tuple(to_homogeneous(SMatrix{2,2,T}(Matrix(R)), SVector{2,T}(0,0))))

to_homogeneous(Tx::CoordinateTransformations.Translation{<:SVector{2,T}}) where {T} =
    HomTransMat{T}(Tuple(@SMatrix [ 1.0 0.0 Tx.translation[1]
                                    0.0 1.0 Tx.translation[2]
                                    0.0 0.0 1.0 ]))

# LinearMap: Type-based dispatch for specific matrix types
# Uniform scaling (isotropic)
to_homogeneous(L::CoordinateTransformations.LinearMap{<:UniformScaling{T}}) where {T} =
    HomScaleIsoMat{T}(Tuple(to_homogeneous(SMatrix{2,2,T}(L.linear), SVector{2,T}(0,0))))

# Diagonal scaling
function to_homogeneous(L::CoordinateTransformations.LinearMap{<:Diagonal{T}}) where {T}
    sx, sy = L.linear.diag[1], L.linear.diag[2]
    M = SMatrix{2,2,T}(L.linear)
    if sx == sy
        return HomScaleIsoMat{T}(Tuple(to_homogeneous(M, SVector{2,T}(0,0))))
    else
        return HomScaleAnisoMat{T}(Tuple(to_homogeneous(M, SVector{2,T}(0,0))))
    end
end

# Rotation matrices (from Rotations.jl)
to_homogeneous(L::CoordinateTransformations.LinearMap{<:Rotations.RotMatrix{2,T}}) where {T} =
    HomRotMat{T}(Tuple(to_homogeneous(SMatrix{2,2,T}(L.linear), SVector{2,T}(0,0))))

# General matrix - fallback with runtime checks
function to_homogeneous(L::CoordinateTransformations.LinearMap{Mat}) where {Mat<:AbstractMatrix}
    T = eltype(L.linear)
    M = SMatrix{2,2,T}(L.linear)
    if iszero(M[1,2]) && iszero(M[2,1])
        sx, sy = M[1,1], M[2,2]
        if sx == sy
            return HomScaleIsoMat{T}(Tuple(to_homogeneous(M, SVector{2,T}(0,0))))
        else
            return HomScaleAnisoMat{T}(Tuple(to_homogeneous(M, SVector{2,T}(0,0))))
        end
    else
        return AffineMat{T}(Tuple(to_homogeneous(M, SVector{2,T}(0,0))))
    end
end

# EuclideanMap: Rotation + Translation
to_homogeneous(E::EuclideanMap{2,T}) where {T} = begin
    M = SMatrix{2,2,T}(Matrix(E.R))
    t = SVector{2,T}(E.t)
    EuclideanMat{T}(Tuple(to_homogeneous(M, t)))
end

# AffineMap: ALWAYS Affine
to_homogeneous(Af::CoordinateTransformations.AffineMap{Mat,V}) where {Mat<:AbstractMatrix,V<:AbstractVector} = begin
    T = promote_type(eltype(Af.linear), eltype(Af.translation))
    M = SMatrix{2,2,T}(Af.linear)
    t = SVector{2,T}(Af.translation)
    AffineMat{T}(Tuple(to_homogeneous(M, t)))
end

# Compositions: ALWAYS Affine (we do not reclassify composites)
to_homogeneous(C::ComposedFunction) = begin
    A = to_homogeneous(C.outer)
    B = to_homogeneous(C.inner)
    T = promote_type(eltype(A), eltype(B))
    H = SMatrix{3,3,T}(Tuple(A)) * SMatrix{3,3,T}(Tuple(B))
    AffineMat{T}(Tuple(H))
end

# Identity methods for already-homogeneous matrices
to_homogeneous(H::HomMatAny) = H

# ========================================================================
# Euclidean Conversions
# ========================================================================

"""
    to_euclidean(v::AbstractVector{T}) -> SVector{N-1,T}

Convert homogeneous coordinates to euclidean by perspective division.
Removes the last coordinate after dividing by it.

# Examples
```julia
h = SVector(2.0, 4.0, 2.0)  # Homogeneous
e = to_euclidean(h)         # SVector(1.0, 2.0) - Euclidean
```
"""
function to_euclidean(v::AbstractVector{T}) where {T}
    n = length(v)
    n >= 2 || throw(ArgumentError("Vector must have at least 2 elements for euclidean conversion"))
    
    w = v[end]
    iszero(w) && throw(ArgumentError("Cannot convert to euclidean: homogeneous coordinate is zero"))
    
    # Promote type to handle division properly (e.g., Int/Int -> Float64)
    R = typeof(v[1] / w)
    return SVector{n-1,R}(v[i] / w for i in 1:(n-1))
end

"""
    to_affine(v::StaticVector{N,T}) -> Point{N+1,T} or SVector{N+1,T}

Convert euclidean coordinates to affine coordinates by appending 1 (setting w=1).
Embeds points in the affine hyperplane of projective space.
Preserves the StaticArray container type (e.g., Point2 → Point3, SVector{2} → SVector{3}).

Note: This restricts points to the affine plane (w=1) and cannot represent ideal points
(points at infinity with w=0). For true projective coordinates with arbitrary w,
use `vcat(v, w)` directly.

Uses `vcat` internally, which automatically preserves type aliases through StaticArrays' type system.

# Examples
```julia
# Works with SVector
e = SVector(1.0, 2.0)       # Euclidean
a = to_affine(e)            # SVector{3}(1.0, 2.0, 1.0)

# Preserves Point type
p = Point2(1.0, 2.0)        # 2D Point
a = to_affine(p)            # Point{3}(1.0, 2.0, 1.0)

# Broadcasting
points = [Point2(1.0, 2.0), Point2(3.0, 4.0)]
affine_points = to_affine.(points)
```
"""
to_affine(v::StaticVector{N,T}) where {N,T} = similar_type(v, Size(N+1))(v..., one(T))

# Fallback for general AbstractVectors (non-static)
function to_affine(v::AbstractVector{T}) where {T}
    n = length(v)
    return SVector{n+1,T}(v..., one(T))
end