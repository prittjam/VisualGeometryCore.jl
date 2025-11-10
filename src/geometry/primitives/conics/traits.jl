# =============================================================================
# Conic Trait Dispatch
# =============================================================================
#
# Note: ConicTrait, CircleTrait, and EllipseTrait types are defined in transforms.jl
# since they're needed there for transformation result type dispatch.

# Delegate HomEllipseMat(::Circle) to HomCircleMat for type correctness
HomEllipseMat(c::Circle) = HomCircleMat(c)

"""
    conic_trait(Q::AbstractHomEllipseMat) -> ConicTrait

Determine if a conic matrix represents a circle or ellipse using Holy Traits.

Dispatches on the concrete type:
- HomCircleMat → CircleTrait()
- HomEllipseMat → EllipseTrait()

# Example
```julia
circle = Circle(Point2(5.0, 3.0), 2.5)
Q = HomCircleMat(circle)
conic_trait(Q)  # CircleTrait()

ellipse = Ellipse(Point2(5.0, 3.0), 3.0, 2.0, π/4)
Q = HomEllipseMat(ellipse)
conic_trait(Q)  # EllipseTrait()
```
"""
conic_trait(::HomCircleMat) = CircleTrait()
conic_trait(::HomEllipseMat) = EllipseTrait()

# Note: conic_trait(::Type{...}) is defined in transforms.jl for use with promote_rule
