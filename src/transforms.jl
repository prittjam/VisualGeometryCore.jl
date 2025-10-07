# ========================================================================
# Homogeneous 2D Transforms
# ========================================================================

# --- 3×3 typed homogeneous wrappers (tuple-backed, minimal StaticArrays interface) ---
macro staticmat3(Name)
    name = esc(Name)
    quote
        struct $name{T} <: StaticArrays.StaticMatrix{3,3,T}
            data::NTuple{9,T}
        end
        StaticArrays.size(::Type{$name{T}}) where {T} = (3,3)
        StaticArrays.similar_type(::Type{$name{T}}, ::Type{S}) where {T,S} = $name{S}
        Base.eltype(::Type{$name{T}}) where {T} = T
        Base.IndexStyle(::Type{$name}) = IndexLinear()
        @inline Base.getindex(A::$name{T}, k::Int) where {T} = A.data[k]
        @inline Base.getindex(A::$name{T}, i::Int, j::Int) where {T} = A.data[(j-1)*3 + i]
        Base.Tuple(A::$name{T}) where {T} = A.data
        $name{T}(args::Vararg{T,9}) where {T} = $name{T}(args)
        $name{T}(M::SMatrix{3,3,T}) where {T} = $name{T}(Tuple(M))
    end
end

@staticmat3 HomRotMat        # homogeneous rotation (no translation)
@staticmat3 HomTransMat      # homogeneous translation (no rotation)
@staticmat3 HomScaleIsoMat   # homogeneous isotropic scaling (s*I)
@staticmat3 HomScaleAnisoMat # homogeneous anisotropic diagonal scaling diag(sx,sy)
@staticmat3 EuclideanMat     # rotation + translation (explicit constructor)
@staticmat3 AffineMat        # general affine (fallback & compositions)

const HomMatAny = Union{HomRotMat, HomTransMat, HomScaleIsoMat, HomScaleAnisoMat, EuclideanMat, AffineMat}

# helpers

# Convert 2×2 matrix and 2D translation to 3×3 homogeneous matrix
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

# LinearMap: ONLY diagonals are treated as scaling; else Affine (zero translation)
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

materialize(tf) = to_homogeneous(tf)

# Identity methods for already-homogeneous matrices
to_homogeneous(H::HomMatAny) = H

# ------------------------------------
# Holy Trait-based Transform Composition
# ------------------------------------

# Transform category traits
abstract type TransformTrait end
struct RotationTrait <: TransformTrait end
struct TranslationTrait <: TransformTrait end
struct IsotropicScaleTrait <: TransformTrait end
struct AnisotropicScaleTrait <: TransformTrait end
struct EuclideanTrait <: TransformTrait end
struct AffineTrait <: TransformTrait end

# Trait dispatch for transform types
transform_trait(::Type{<:HomRotMat}) = RotationTrait()
transform_trait(::Type{<:HomTransMat}) = TranslationTrait()
transform_trait(::Type{<:HomScaleIsoMat}) = IsotropicScaleTrait()
transform_trait(::Type{<:HomScaleAnisoMat}) = AnisotropicScaleTrait()
transform_trait(::Type{<:EuclideanMat}) = EuclideanTrait()
transform_trait(::Type{<:AffineMat}) = AffineTrait()

# No ranking needed! We use dispatch failure as a signal to swap arguments

# Result type computation using traits (canonical order only)
result_type_trait(::RotationTrait, ::RotationTrait) = HomRotMat
result_type_trait(::TranslationTrait, ::TranslationTrait) = HomTransMat
result_type_trait(::IsotropicScaleTrait, ::IsotropicScaleTrait) = HomScaleIsoMat
result_type_trait(::AnisotropicScaleTrait, ::AnisotropicScaleTrait) = HomScaleAnisoMat
result_type_trait(::EuclideanTrait, ::EuclideanTrait) = EuclideanMat
result_type_trait(::AffineTrait, ::AffineTrait) = AffineMat

# Euclidean combinations (rotation + translation)
result_type_trait(::RotationTrait, ::TranslationTrait) = EuclideanMat
result_type_trait(::RotationTrait, ::EuclideanTrait) = EuclideanMat
result_type_trait(::TranslationTrait, ::EuclideanTrait) = EuclideanMat

# Scaling combinations
result_type_trait(::IsotropicScaleTrait, ::AnisotropicScaleTrait) = HomScaleAnisoMat

# Scaling with rigid transforms → Affine
result_type_trait(::RotationTrait, ::IsotropicScaleTrait) = AffineMat
result_type_trait(::RotationTrait, ::AnisotropicScaleTrait) = AffineMat
result_type_trait(::TranslationTrait, ::IsotropicScaleTrait) = AffineMat
result_type_trait(::TranslationTrait, ::AnisotropicScaleTrait) = AffineMat
result_type_trait(::IsotropicScaleTrait, ::EuclideanTrait) = AffineMat
result_type_trait(::AnisotropicScaleTrait, ::EuclideanTrait) = AffineMat

# Anything with Affine → Affine
result_type_trait(::RotationTrait, ::AffineTrait) = AffineMat
result_type_trait(::TranslationTrait, ::AffineTrait) = AffineMat
result_type_trait(::IsotropicScaleTrait, ::AffineTrait) = AffineMat
result_type_trait(::AnisotropicScaleTrait, ::AffineTrait) = AffineMat
result_type_trait(::EuclideanTrait, ::AffineTrait) = AffineMat

# Fallback trait method - if we hit this, swap arguments and try again
# But add a recursion breaker for the ultimate fallback
result_type_trait(a::TransformTrait, b::TransformTrait) = result_type_trait(b, a)

# Ultimate fallback to prevent infinite recursion - this should never be hit if we define methods correctly
result_type_trait(::AffineTrait, ::TransformTrait) = AffineMat

# Public interface - just dispatch to traits
result_type(::Type{A}, ::Type{B}) where {A<:HomMatAny, B<:HomMatAny} = 
    result_type_trait(transform_trait(A), transform_trait(B))

# Additional similar_type methods to construct any result type with new element type
# This allows us to use similar_type(ResultType, NewElementType) even when ResultType 
# is different from the input types
StaticArrays.similar_type(::Type{HomRotMat}, ::Type{T}) where {T} = HomRotMat{T}
StaticArrays.similar_type(::Type{HomTransMat}, ::Type{T}) where {T} = HomTransMat{T}
StaticArrays.similar_type(::Type{HomScaleIsoMat}, ::Type{T}) where {T} = HomScaleIsoMat{T}
StaticArrays.similar_type(::Type{HomScaleAnisoMat}, ::Type{T}) where {T} = HomScaleAnisoMat{T}
StaticArrays.similar_type(::Type{EuclideanMat}, ::Type{T}) where {T} = EuclideanMat{T}
StaticArrays.similar_type(::Type{AffineMat}, ::Type{T}) where {T} = AffineMat{T}

# Single multiplication function
function Base.:*(a::HomMatAny, b::HomMatAny)
    T = promote_type(eltype(a), eltype(b))
    ResultType = result_type(typeof(a), typeof(b))
    C = SMatrix{3,3,T}(Tuple(a)) * SMatrix{3,3,T}(Tuple(b))
    return similar_type(ResultType, T)(Tuple(C))
end

# inverses (keep class; we don't reclassify)
Base.inv(A::HomMatAny) = similar_type(typeof(A), eltype(A))(Tuple(inv(SMatrix{3,3,eltype(A)}(Tuple(A)))))

