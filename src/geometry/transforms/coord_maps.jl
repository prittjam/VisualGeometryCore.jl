
# =============================================================================
# Canonical Geometries (Singleton Types)
# =============================================================================

"""
    CanonicalSquare

Singleton type representing the canonical square [-1,1] × [-1,1].
Use the constant `CANONICAL_SQUARE` in code.
"""
struct CanonicalSquare end

"""
    UnitCircle

Singleton type representing the unit circle (centered at origin with radius 1).
Use the constant `UNIT_CIRCLE` in code.
"""
struct UnitCircle end

"""
    CANONICAL_SQUARE

The canonical square [-1,1] × [-1,1] commonly used in computer vision and graphics.
This is the domain for normalized device coordinates and many geometric transformations.

Note: Despite the name, this has side length 2, not 1. It's "canonical" because it's
centered at the origin with unit extent in each direction.
"""
const CANONICAL_SQUARE = CanonicalSquare()

"""
    UNIT_CIRCLE

The unit circle centered at the origin with radius 1.
This is the fundamental domain for many geometric operations including log-polar transforms.
"""
const UNIT_CIRCLE = UnitCircle()

# =============================================================================
# Image Coordinate Adaptation
# =============================================================================

"""
    imgmap(transform) -> Function

Adapt a geometric transformation for use with image warping.

Images use (row, col) indexing where rows increase downward and columns increase rightward.
Geometric transformations typically use Cartesian (x, y) coordinates where x increases rightward
and y increases upward. This function handles the coordinate convention swap: (row, col) ↔ (x, y).

The returned function applies the transformation in Cartesian space while handling the
row/col ↔ x/y conversion automatically.

# Arguments
- `transform`: A function mapping Point2 → Point2 in Cartesian coordinates

# Returns
- A function suitable for use with `ImageTransformations.warp`

# Example
```julia
using ImageTransformations: warp
using Interpolations

# Create a log-polar transform
circle = Circle(Point2(100.0, 100.0), 50.0)
logpolar_rect = Rect((1..256, 1..256))
transform = logpolar_map(circle, logpolar_rect, 0.01)

# Warp image using the adapted transform
itp = interpolate(image, BSpline(Cubic(Line(OnGrid()))))
etp = extrapolate(itp, Flat())
patch = warp(etp, imgmap(transform), (1:256, 1:256))

# Compare to manual coordinate swapping:
# patch = warp(etp, p->reverse(transform(reverse(p))), (1:256, 1:256))
```

# See Also
- [`logpolar_map`](@ref): Log-polar coordinate transformation
- [`coord_map`](@ref): Generic coordinate mapping between geometries
"""
imgmap(f) = p -> reverse(f(reverse(p)))

# =============================================================================
# Generic Coordinate Mapping (coord_map)
# =============================================================================

"""
    coord_map(source, target) -> AffineMap

Create an affine transformation that maps coordinates from `source` geometry to `target` geometry.

This is a generic interface using multiple dispatch. The argument order is always consistent:
coordinates flow from source → target.

# Supported Geometries
- `Rect` (rectangles)
- `Circle` (circles)
- `Ellipse` (ellipses)
- `CANONICAL_SQUARE` (the canonical square [-1,1]²)
- `UNIT_CIRCLE` (unit circle with radius 1)

# Arguments
- `source`: Source geometry (where coordinates come from)
- `target`: Target geometry (where coordinates map to)

# Returns
- `AffineMap`: Coordinate transformation from source to target

# Examples
```julia
# Map from canonical square to a rectangle
rect = Rect((0..100, 0..200))
tform = coord_map(CANONICAL_SQUARE, rect)

# Map from circle to canonical square (for normalization)
circle = Circle(Point2(50.0, 50.0), 30.0)
tform = coord_map(circle, CANONICAL_SQUARE)

# Compose transformations for log-polar mapping
logpolar_rect = Rect((1..512, 1..512))
circle_region = Circle(Point2(100.0, 100.0), 50.0)

transform = coord_map(UNIT_CIRCLE, circle_region) ∘
            logpolar_to_cartesian() ∘
            coord_map(logpolar_rect, CANONICAL_SQUARE)
```

# See Also
- [`CANONICAL_SQUARE`](@ref): The canonical square [-1,1]²
- [`UNIT_CIRCLE`](@ref): The unit circle
- [`logpolar_map`](@ref): Convenience function for log-polar transformations
"""
# Forward: CANONICAL_SQUARE → geometry
coord_map(::CanonicalSquare, ::CanonicalSquare) = identity  # Identity map
coord_map(::CanonicalSquare, ::UnitCircle) = identity  # They're the same

function coord_map(::CanonicalSquare, target::HyperRectangle{2})
    c = center(target)
    w = GeometryBasics.widths(target)
    A = SMatrix{2,2}(w[1]/2, 0, 0, w[2]/2)
    return AffineMap(A, c)
end

function coord_map(::CanonicalSquare, target::Ellipse)
    c = GeometryBasics.origin(target)
    a = target.a
    b = target.b
    θ = target.θ

    # RotMatrix{2} assumes y-up convention, but images use y-down
    # Apply y-reflection by negating the b (minor axis) component
    S = SMatrix{2,2}(a, 0, 0, -b)
    R = RotMatrix{2}(θ)
    A = R * S

    return AffineMap(A, c)
end

function coord_map(::CanonicalSquare, target::Circle)
    c = GeometryBasics.origin(target)
    r = radius(target)
    A = SMatrix{2,2}(r, 0, 0, r)
    return AffineMap(A, c)
end

# Inverse: geometry → CANONICAL_SQUARE (use symmetry)
coord_map(source::HyperRectangle{2}, ::CanonicalSquare) = inv(coord_map(CANONICAL_SQUARE, source))
coord_map(source::Ellipse, ::CanonicalSquare) = inv(coord_map(CANONICAL_SQUARE, source))
coord_map(source::Circle, ::CanonicalSquare) = inv(coord_map(CANONICAL_SQUARE, source))

# Unit circle mappings (delegate to CANONICAL_SQUARE since they're the same for radius-1 circle)
coord_map(::UnitCircle, ::UnitCircle) = identity
coord_map(::UnitCircle, target::Circle) = coord_map(CANONICAL_SQUARE, target)
coord_map(::UnitCircle, target::Ellipse) = coord_map(CANONICAL_SQUARE, target)
coord_map(source::Circle, ::UnitCircle) = coord_map(source, CANONICAL_SQUARE)
coord_map(source::Ellipse, ::UnitCircle) = coord_map(source, CANONICAL_SQUARE)

# Cross-geometry mappings (via canonical square)
coord_map(source::HyperRectangle{2}, target::HyperRectangle{2}) =
    coord_map(CANONICAL_SQUARE, target) ∘ coord_map(source, CANONICAL_SQUARE)
coord_map(source::HyperRectangle{2}, target::Circle) =
    coord_map(CANONICAL_SQUARE, target) ∘ coord_map(source, CANONICAL_SQUARE)
coord_map(source::HyperRectangle{2}, target::Ellipse) =
    coord_map(CANONICAL_SQUARE, target) ∘ coord_map(source, CANONICAL_SQUARE)
coord_map(source::Circle, target::HyperRectangle{2}) =
    coord_map(CANONICAL_SQUARE, target) ∘ coord_map(source, CANONICAL_SQUARE)
coord_map(source::Circle, target::Ellipse) =
    coord_map(CANONICAL_SQUARE, target) ∘ coord_map(source, CANONICAL_SQUARE)
coord_map(source::Ellipse, target::HyperRectangle{2}) =
    coord_map(CANONICAL_SQUARE, target) ∘ coord_map(source, CANONICAL_SQUARE)
coord_map(source::Ellipse, target::Circle) =
    coord_map(CANONICAL_SQUARE, target) ∘ coord_map(source, CANONICAL_SQUARE)

"""
Transform conic by any homogeneous transformation: Q' = H^{-T} * Q * H^{-1} (contravariant).

Works with all transform types (HomRotMat, HomTransMat, EuclideanMat, SimilarityMat, AffineMat, etc.)
and all conic types (HomCircleMat, HomEllipseMat).

Result type is determined by trait promotion:
- Circle + Similarity (rotation/translation/uniform scale) → Circle (preserves circularity)
- Circle + Anisotropic/Affine/Projective → Ellipse (breaks circularity)
- Ellipse + Any transform → Ellipse

# Examples
```julia
# Similarity transform preserves circles
R = HomRotMat{Float64}(...)
Q_circle = HomCircleMat(circle)
Q_rotated = R(Q_circle)  # → HomCircleMat (still a circle!)

# Affine transform breaks circularity
A = AffineMat{Float64}(...)
Q_ellipse = A(Q_circle)  # → HomEllipseMat (now an ellipse)

# Homography for image warping
H = PlanarHomographyMat(camera)
Q_warped = H(Q_circle)  # → HomEllipseMat
ellipse = Ellipse(Q_warped)
```
"""
# Generate call operator methods for all transform types
for TransformType in (:HomRotMat, :HomTransMat, :HomScaleIsoMat, :HomScaleAnisoMat,
                      :EuclideanMat, :SimilarityMat, :AffineMat, :PlanarHomographyMat)
    for ConicType in (:HomCircleMat, :HomEllipseMat)
        @eval begin
            function (H::$TransformType)(Q::$ConicType)
                T1 = eltype(H)
                T2 = eltype(Q)
                T = promote_type(T1, T2)
                ResultType = conic_result_type($TransformType, $ConicType)

                H_mat = SMatrix{3,3,T}(H)
                Q_mat = SMatrix{3,3,T}(Q)

                # Apply conic transformation: Q' = H^{-T} * Q * H^{-1}
                invH = inv(H_mat)
                Q_warped = transpose(invH) * Q_mat * invH

                return similar_type(ResultType, T)(Tuple(Q_warped))
            end
        end
    end
end
