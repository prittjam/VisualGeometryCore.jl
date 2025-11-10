# =============================================================================
# Geometric Primitives Module
# =============================================================================
#
# This module defines geometric primitives (rectangles, ellipses, circles)
# and their associated operations.
#
# Organization:
# - rectangles.jl: Rect constructors from intervals and ranges
# - conics/: Conic sections (ellipses and circles)
#   - ellipse.jl: Ellipse type and HomEllipseMat conversions
#   - circle.jl: Circle construction from blobs
#   - traits.jl: Conic trait system
#   - gradient.jl: Gradient computation for conics
# - containment.jl: Spatial containment queries (Base.in)
# - operators.jl: Geometric operations (+, *, intersection)

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
