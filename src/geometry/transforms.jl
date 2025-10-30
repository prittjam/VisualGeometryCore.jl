# ========================================================================
# Homogeneous 2D Transforms
# ========================================================================

# --- Generic static matrix wrapper macro ---
# Creates a parametric StaticMatrix wrapper type with efficient tuple storage
macro smatrix_wrapper(Name, m, n)
    name = esc(Name)
    size_m = m
    size_n = n
    size_total = m * n
    quote
        struct $name{T} <: StaticArrays.StaticMatrix{$size_m, $size_n, T}
            data::NTuple{$size_total, T}
        end
        StaticArrays.size(::Type{$name{T}}) where {T} = ($size_m, $size_n)
        StaticArrays.similar_type(::Type{$name{T}}, ::Type{S}) where {T,S} = $name{S}
        Base.eltype(::Type{$name{T}}) where {T} = T
        Base.IndexStyle(::Type{$name}) = IndexLinear()
        @inline Base.getindex(A::$name{T}, k::Int) where {T} = A.data[k]
        @inline Base.getindex(A::$name{T}, i::Int, j::Int) where {T} = A.data[(j-1)*$size_m + i]
        Base.Tuple(A::$name{T}) where {T} = A.data
        $name{T}(args::Vararg{T, $size_total}) where {T} = $name{T}(args)
        $name{T}(M::SMatrix{$size_m, $size_n, T}) where {T} = $name{T}(Tuple(M))
    end
end

# 3×3 homogeneous transform matrices
@smatrix_wrapper HomRotMat 3 3              # homogeneous rotation (no translation)
@smatrix_wrapper HomTransMat 3 3            # homogeneous translation (no rotation)
@smatrix_wrapper HomScaleIsoMat 3 3         # homogeneous isotropic scaling (s*I)
@smatrix_wrapper HomScaleAnisoMat 3 3       # homogeneous anisotropic diagonal scaling diag(sx,sy)
@smatrix_wrapper EuclideanMat 3 3           # rotation + translation (rigid body)
@smatrix_wrapper SimilarityMat 3 3          # rotation + translation + uniform scaling
@smatrix_wrapper AffineMat 3 3              # general affine (fallback & compositions)
@smatrix_wrapper PlanarHomographyMat 3 3    # planar projective homography (most general)

# Conic matrices (geometric objects, not transforms)
@smatrix_wrapper HomEllipseMat 3 3          # conic matrix for general ellipse
@smatrix_wrapper HomCircleMat 3 3           # conic matrix for circle (special case)

const HomMatAny = Union{HomRotMat, HomTransMat, HomScaleIsoMat, HomScaleAnisoMat, EuclideanMat, SimilarityMat, AffineMat, PlanarHomographyMat}
const HomConicAny = Union{HomEllipseMat, HomCircleMat}

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
struct SimilarityTrait <: TransformTrait end
struct AffineTrait <: TransformTrait end
struct ProjectiveTrait <: TransformTrait end

# Trait dispatch for transform types
transform_trait(::Type{<:HomRotMat}) = RotationTrait()
transform_trait(::Type{<:HomTransMat}) = TranslationTrait()
transform_trait(::Type{<:HomScaleIsoMat}) = IsotropicScaleTrait()
transform_trait(::Type{<:HomScaleAnisoMat}) = AnisotropicScaleTrait()
transform_trait(::Type{<:EuclideanMat}) = EuclideanTrait()
transform_trait(::Type{<:SimilarityMat}) = SimilarityTrait()
transform_trait(::Type{<:AffineMat}) = AffineTrait()
transform_trait(::Type{<:PlanarHomographyMat}) = ProjectiveTrait()

# No ranking needed! We use dispatch failure as a signal to swap arguments

# Result type computation using traits (canonical order only)
result_type_trait(::RotationTrait, ::RotationTrait) = HomRotMat
result_type_trait(::TranslationTrait, ::TranslationTrait) = HomTransMat
result_type_trait(::IsotropicScaleTrait, ::IsotropicScaleTrait) = HomScaleIsoMat
result_type_trait(::AnisotropicScaleTrait, ::AnisotropicScaleTrait) = HomScaleAnisoMat
result_type_trait(::EuclideanTrait, ::EuclideanTrait) = EuclideanMat
result_type_trait(::SimilarityTrait, ::SimilarityTrait) = SimilarityMat
result_type_trait(::AffineTrait, ::AffineTrait) = AffineMat

# Euclidean combinations (rotation + translation)
result_type_trait(::RotationTrait, ::TranslationTrait) = EuclideanMat
result_type_trait(::RotationTrait, ::EuclideanTrait) = EuclideanMat
result_type_trait(::TranslationTrait, ::EuclideanTrait) = EuclideanMat

# Scaling combinations
result_type_trait(::IsotropicScaleTrait, ::AnisotropicScaleTrait) = HomScaleAnisoMat

# Similarity combinations (rotation + translation + uniform scaling)
result_type_trait(::RotationTrait, ::IsotropicScaleTrait) = SimilarityMat
result_type_trait(::TranslationTrait, ::IsotropicScaleTrait) = SimilarityMat
result_type_trait(::IsotropicScaleTrait, ::EuclideanTrait) = SimilarityMat
result_type_trait(::RotationTrait, ::SimilarityTrait) = SimilarityMat
result_type_trait(::TranslationTrait, ::SimilarityTrait) = SimilarityMat
result_type_trait(::IsotropicScaleTrait, ::SimilarityTrait) = SimilarityMat
result_type_trait(::EuclideanTrait, ::SimilarityTrait) = SimilarityMat

# Anisotropic scaling breaks similarity → Affine
result_type_trait(::RotationTrait, ::AnisotropicScaleTrait) = AffineMat
result_type_trait(::TranslationTrait, ::AnisotropicScaleTrait) = AffineMat
result_type_trait(::AnisotropicScaleTrait, ::EuclideanTrait) = AffineMat
result_type_trait(::AnisotropicScaleTrait, ::SimilarityTrait) = AffineMat

# Anything with Affine → Affine
result_type_trait(::RotationTrait, ::AffineTrait) = AffineMat
result_type_trait(::TranslationTrait, ::AffineTrait) = AffineMat
result_type_trait(::IsotropicScaleTrait, ::AffineTrait) = AffineMat
result_type_trait(::AnisotropicScaleTrait, ::AffineTrait) = AffineMat
result_type_trait(::EuclideanTrait, ::AffineTrait) = AffineMat
result_type_trait(::SimilarityTrait, ::AffineTrait) = AffineMat

# Projective combinations (most general - anything with projective is projective)
result_type_trait(::ProjectiveTrait, ::TransformTrait) = PlanarHomographyMat

# Fallback trait method - if we hit this, swap arguments and try again
result_type_trait(a::TransformTrait, b::TransformTrait) = result_type_trait(b, a)

# Public interface - just dispatch to traits
result_type(::Type{A}, ::Type{B}) where {A<:HomMatAny, B<:HomMatAny} =
    result_type_trait(transform_trait(A), transform_trait(B))

# ------------------------------------
# Holy Trait-based Conic Classification
# ------------------------------------

# Conic shape traits
abstract type ConicTrait end
struct CircleTrait <: ConicTrait end
struct EllipseTrait <: ConicTrait end

# Additional similar_type methods to construct any result type with new element type
# This allows us to use similar_type(ResultType, NewElementType) even when ResultType 
# is different from the input types
StaticArrays.similar_type(::Type{HomRotMat}, ::Type{T}) where {T} = HomRotMat{T}
StaticArrays.similar_type(::Type{HomTransMat}, ::Type{T}) where {T} = HomTransMat{T}
StaticArrays.similar_type(::Type{HomScaleIsoMat}, ::Type{T}) where {T} = HomScaleIsoMat{T}
StaticArrays.similar_type(::Type{HomScaleAnisoMat}, ::Type{T}) where {T} = HomScaleAnisoMat{T}
StaticArrays.similar_type(::Type{EuclideanMat}, ::Type{T}) where {T} = EuclideanMat{T}
StaticArrays.similar_type(::Type{SimilarityMat}, ::Type{T}) where {T} = SimilarityMat{T}
StaticArrays.similar_type(::Type{AffineMat}, ::Type{T}) where {T} = AffineMat{T}
StaticArrays.similar_type(::Type{PlanarHomographyMat}, ::Type{T}) where {T} = PlanarHomographyMat{T}

# Single multiplication function
function Base.:*(a::HomMatAny, b::HomMatAny)
    T = promote_type(eltype(a), eltype(b))
    ResultType = result_type(typeof(a), typeof(b))
    # Convert to SMatrix for multiplication, then construct result directly from Tuple
    C = SMatrix{3,3,T}(a) * SMatrix{3,3,T}(b)
    return similar_type(ResultType, T)(Tuple(C))
end

# inverses (keep class; we don't reclassify)
Base.inv(A::HomMatAny) = similar_type(typeof(A), eltype(A))(Tuple(inv(SMatrix{3,3,eltype(A)}(A))))