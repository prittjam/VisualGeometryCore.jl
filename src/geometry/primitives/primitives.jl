"""
    Primitives

Geometric primitives submodule for VisualGeometryCore.

Provides geometric primitive types and operations:
- **Rectangles**: Rect constructors from intervals and ranges
- **Conics**: Ellipse and Circle types with conic matrix representations
- **Traits**: Conic trait system for dispatch
- **Operations**: Geometric operations (dilate, intersect, containment)

# Main Types
- `Ellipse`, `Circle`, `Rect`
- `ConicTrait`, `CircleTrait`, `EllipseTrait`

# Main Functions
- `dilate`, `intersects`, `gradient`
- `is_ellipse`, `conic_trait`

# Example
```julia
using VisualGeometryCore.Primitives

ellipse = Ellipse(Point2(0.0, 0.0), 5.0, 3.0, π/4)
circle = Circle(Point2(1.0, 2.0), 2.0)
dilated = dilate(circle, 2.0)
```
"""
module Primitives

# Import from parent module
using ..VisualGeometryCore: GeometryBasics, Point2, Point2f, Rect, Rect2, Vec2, HyperRectangle
import ..VisualGeometryCore: Circle  # Import to extend
using ..VisualGeometryCore: StaticArrays, SVector, SMatrix, @SMatrix, @SVector
using ..VisualGeometryCore: LinearAlgebra
using ..VisualGeometryCore: IntervalSets, ClosedInterval, leftendpoint, rightendpoint

# Import transform types (defined in transforms/homogeneous.jl)
using ..VisualGeometryCore: HomEllipseMat, HomCircleMat

# Import blob types (needed for Circle construction)
using ..VisualGeometryCore: AbstractBlob

# Rectangle constructors
include("rectangles.jl")

# Conic sections
include("conics/ellipse.jl")
include("conics/circle.jl")
include("conics/traits.jl")
include("conics/gradient.jl")

# Spatial queries and operations
include("containment.jl")
include("operators.jl")

# ==============================================================================
# Exports
# ==============================================================================

# Ellipse and Circle types
export Ellipse, Circle

# Conic traits
export ConicTrait, CircleTrait, EllipseTrait, conic_trait

# Geometric operations
export dilate, intersects

# Conic utilities
export is_ellipse, gradient

end # module Primitives
