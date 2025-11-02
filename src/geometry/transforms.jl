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

# Transform category traits with hierarchy for dispatch

"""
    TransformTrait

Abstract base type for transformation category traits.
Uses Holy Traits pattern with promote_rule for automatic composition inference.
"""
abstract type TransformTrait end

"""
    RigidTrait <: TransformTrait

Abstract parent for rigid transformations (preserve distances).
Includes: Rotation, Translation, Euclidean
"""
abstract type RigidTrait <: TransformTrait end

"""
    ScaleTrait <: TransformTrait

Abstract parent for transformations including scaling.
"""
abstract type ScaleTrait <: TransformTrait end

# Concrete trait types
struct RotationTrait <: RigidTrait end
struct TranslationTrait <: RigidTrait end
struct IsotropicScaleTrait <: ScaleTrait end
struct AnisotropicScaleTrait <: ScaleTrait end
struct EuclideanTrait <: RigidTrait end
struct SimilarityTrait <: ScaleTrait end
struct AffineTrait <: TransformTrait end
struct ProjectiveTrait <: TransformTrait end

# ------------------------------------
# Trait Promotion Rules
# ------------------------------------

# Symmetric helper macro for bidirectional promotion
macro symmetric_promote_rule(T1, T2, Result)
    quote
        Base.promote_rule(::Type{$(esc(T1))}, ::Type{$(esc(T2))}) = $(esc(Result))
        Base.promote_rule(::Type{$(esc(T2))}, ::Type{$(esc(T1))}) = $(esc(Result))
    end
end

# Identity promotions (same trait)
Base.promote_rule(::Type{T}, ::Type{T}) where {T<:TransformTrait} = T

# Level 1→2: Component rigids → Euclidean
@symmetric_promote_rule RotationTrait TranslationTrait EuclideanTrait

# Level 2: Euclidean absorbs other rigid traits
Base.promote_rule(::Type{EuclideanTrait}, ::Type{<:RigidTrait}) = EuclideanTrait
Base.promote_rule(::Type{<:RigidTrait}, ::Type{EuclideanTrait}) = EuclideanTrait

# Scaling composition
@symmetric_promote_rule IsotropicScaleTrait AnisotropicScaleTrait AnisotropicScaleTrait

# Level 2→3: Rigid + IsotropicScale → Similarity
Base.promote_rule(::Type{<:RigidTrait}, ::Type{IsotropicScaleTrait}) = SimilarityTrait
Base.promote_rule(::Type{IsotropicScaleTrait}, ::Type{<:RigidTrait}) = SimilarityTrait
@symmetric_promote_rule SimilarityTrait IsotropicScaleTrait SimilarityTrait

# Level 3→4: Rigid + AnisotropicScale → Affine (breaks similarity)
Base.promote_rule(::Type{<:RigidTrait}, ::Type{AnisotropicScaleTrait}) = AffineTrait
Base.promote_rule(::Type{AnisotropicScaleTrait}, ::Type{<:RigidTrait}) = AffineTrait

# Level 3→4: Similarity + AnisotropicScale → Affine
@symmetric_promote_rule SimilarityTrait AnisotropicScaleTrait AffineTrait

# Level 4: Affine absorbs all lower-level transforms
Base.promote_rule(::Type{AffineTrait}, ::Type{<:TransformTrait}) = AffineTrait
Base.promote_rule(::Type{<:TransformTrait}, ::Type{AffineTrait}) = AffineTrait

# Level 5: Projective absorbs everything (most general)
Base.promote_rule(::Type{ProjectiveTrait}, ::Type{<:TransformTrait}) = ProjectiveTrait
Base.promote_rule(::Type{<:TransformTrait}, ::Type{ProjectiveTrait}) = ProjectiveTrait

# Trait dispatch for transform types
transform_trait(::Type{<:HomRotMat}) = RotationTrait()
transform_trait(::Type{<:HomTransMat}) = TranslationTrait()
transform_trait(::Type{<:HomScaleIsoMat}) = IsotropicScaleTrait()
transform_trait(::Type{<:HomScaleAnisoMat}) = AnisotropicScaleTrait()
transform_trait(::Type{<:EuclideanMat}) = EuclideanTrait()
transform_trait(::Type{<:SimilarityMat}) = SimilarityTrait()
transform_trait(::Type{<:AffineMat}) = AffineTrait()
transform_trait(::Type{<:PlanarHomographyMat}) = ProjectiveTrait()

# ------------------------------------
# Result Type Computation via Promotion
# ------------------------------------

# Map from promoted trait type to matrix type
matrix_type(::Type{RotationTrait}) = HomRotMat
matrix_type(::Type{TranslationTrait}) = HomTransMat
matrix_type(::Type{IsotropicScaleTrait}) = HomScaleIsoMat
matrix_type(::Type{AnisotropicScaleTrait}) = HomScaleAnisoMat
matrix_type(::Type{EuclideanTrait}) = EuclideanMat
matrix_type(::Type{SimilarityTrait}) = SimilarityMat
matrix_type(::Type{AffineTrait}) = AffineMat
matrix_type(::Type{ProjectiveTrait}) = PlanarHomographyMat

# Compute result type using trait promotion
function result_type(::Type{A}, ::Type{B}) where {A<:HomMatAny, B<:HomMatAny}
    trait_a = typeof(transform_trait(A))
    trait_b = typeof(transform_trait(B))
    promoted_trait = promote_type(trait_a, trait_b)
    return matrix_type(promoted_trait)
end

# ------------------------------------
# Holy Trait-based Conic Classification
# ------------------------------------

# Conic shape traits
abstract type ConicTrait end
struct CircleTrait <: ConicTrait end
struct EllipseTrait <: ConicTrait end

# Trait dispatch for conic types
conic_trait(::Type{<:HomCircleMat}) = CircleTrait()
conic_trait(::Type{<:HomEllipseMat}) = EllipseTrait()

# ------------------------------------
# Conic Transformation Result Types
# ------------------------------------

# Promotion rules: Transform × Conic → Conic
# Circle + Similarity (rotation/translation/uniform scale) → Circle (preserves circularity)
Base.promote_rule(::Type{RotationTrait}, ::Type{CircleTrait}) = CircleTrait
Base.promote_rule(::Type{TranslationTrait}, ::Type{CircleTrait}) = CircleTrait
Base.promote_rule(::Type{EuclideanTrait}, ::Type{CircleTrait}) = CircleTrait
Base.promote_rule(::Type{IsotropicScaleTrait}, ::Type{CircleTrait}) = CircleTrait
Base.promote_rule(::Type{SimilarityTrait}, ::Type{CircleTrait}) = CircleTrait

# Symmetric versions (Conic × Transform)
Base.promote_rule(::Type{CircleTrait}, ::Type{RotationTrait}) = CircleTrait
Base.promote_rule(::Type{CircleTrait}, ::Type{TranslationTrait}) = CircleTrait
Base.promote_rule(::Type{CircleTrait}, ::Type{EuclideanTrait}) = CircleTrait
Base.promote_rule(::Type{CircleTrait}, ::Type{IsotropicScaleTrait}) = CircleTrait
Base.promote_rule(::Type{CircleTrait}, ::Type{SimilarityTrait}) = CircleTrait

# Circle + Anisotropic/Affine/Projective → Ellipse (breaks circularity)
Base.promote_rule(::Type{AnisotropicScaleTrait}, ::Type{CircleTrait}) = EllipseTrait
Base.promote_rule(::Type{AffineTrait}, ::Type{CircleTrait}) = EllipseTrait
Base.promote_rule(::Type{ProjectiveTrait}, ::Type{CircleTrait}) = EllipseTrait

Base.promote_rule(::Type{CircleTrait}, ::Type{AnisotropicScaleTrait}) = EllipseTrait
Base.promote_rule(::Type{CircleTrait}, ::Type{AffineTrait}) = EllipseTrait
Base.promote_rule(::Type{CircleTrait}, ::Type{ProjectiveTrait}) = EllipseTrait

# Ellipse + any transform → Ellipse (ellipses stay ellipses)
Base.promote_rule(::Type{<:TransformTrait}, ::Type{EllipseTrait}) = EllipseTrait
Base.promote_rule(::Type{EllipseTrait}, ::Type{<:TransformTrait}) = EllipseTrait

# Map from promoted trait to conic matrix type
conic_matrix_type(::Type{CircleTrait}) = HomCircleMat
conic_matrix_type(::Type{EllipseTrait}) = HomEllipseMat

# Compute result type for transform × conic
function conic_result_type(::Type{H}, ::Type{Q}) where {H<:HomMatAny, Q<:HomConicAny}
    trait_h = typeof(transform_trait(H))
    trait_q = typeof(conic_trait(Q))
    promoted_conic = promote_type(trait_h, trait_q)
    return conic_matrix_type(promoted_conic)
end

# Additional similar_type methods to construct any result type with new element type
# This allows us to use similar_type(ResultType, NewElementType) even when ResultType
# is different from the input types

# Transform matrix types
StaticArrays.similar_type(::Type{HomRotMat}, ::Type{T}) where {T} = HomRotMat{T}
StaticArrays.similar_type(::Type{HomTransMat}, ::Type{T}) where {T} = HomTransMat{T}
StaticArrays.similar_type(::Type{HomScaleIsoMat}, ::Type{T}) where {T} = HomScaleIsoMat{T}
StaticArrays.similar_type(::Type{HomScaleAnisoMat}, ::Type{T}) where {T} = HomScaleAnisoMat{T}
StaticArrays.similar_type(::Type{EuclideanMat}, ::Type{T}) where {T} = EuclideanMat{T}
StaticArrays.similar_type(::Type{SimilarityMat}, ::Type{T}) where {T} = SimilarityMat{T}
StaticArrays.similar_type(::Type{AffineMat}, ::Type{T}) where {T} = AffineMat{T}
StaticArrays.similar_type(::Type{PlanarHomographyMat}, ::Type{T}) where {T} = PlanarHomographyMat{T}

# Conic matrix types
StaticArrays.similar_type(::Type{HomCircleMat}, ::Type{T}) where {T} = HomCircleMat{T}
StaticArrays.similar_type(::Type{HomEllipseMat}, ::Type{T}) where {T} = HomEllipseMat{T}

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