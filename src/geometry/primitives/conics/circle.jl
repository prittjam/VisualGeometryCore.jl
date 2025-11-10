# =============================================================================
# Circle Construction and Operations
# =============================================================================


# =============================================================================
# Circle Construction from Blobs
# =============================================================================

"""
    GeometryBasics.Circle(blob::AbstractBlob, cutoff::Real=1.0) -> Circle

Construct a GeometryBasics.Circle from a blob with radius `cutoff * σ`.

The `cutoff` parameter determines the effective radius as a multiple of σ
(e.g., 3.0 for 3σ radius). Units are preserved from the blob.

# Arguments
- `blob::AbstractBlob`: Blob with center and scale σ
- `cutoff::Real`: Radius multiplier (default: 1.0, typically 2-4 for detection)

# Returns
- `GeometryBasics.Circle`: Circle with center and radius preserving blob units

# Example
```julia
blob = IsoBlob(Point2(100.0mm, 200.0mm), 5.0mm)
circle = Circle(blob)       # Circle with radius 5.0mm (1σ)
circle = Circle(blob, 3.0)  # Circle with radius 15.0mm (3σ)
```
"""
GeometryBasics.Circle(blob::AbstractBlob, cutoff::Real=1.0) = GeometryBasics.Circle(blob.center, cutoff * blob.σ)
"""
    Circle(Q::Union{HomEllipseMat, HomCircleMat}) -> Circle

Extract circle parameters from conic matrix.

Converts the conic to an Ellipse first, then checks if it's geometrically a circle
(i.e., a ≈ b). If so, returns a Circle with the average of a and b as the radius.

Throws an error if the conic doesn't represent a circle (a and b differ significantly).

# Example
```julia
Q = HomCircleMat(Circle(Point2(1.0, 2.0), 3.0))
circle = Circle(Q)  # Extract circle parameters

# Works for HomEllipseMat too if it's geometrically a circle
H = HomScaleIsoMat(...)  # Isotropic scaling
Q_scaled = H(Q)
circle_scaled = Circle(Q_scaled)
```
"""
function GeometryBasics.Circle(Q::Union{HomEllipseMat{T}, HomCircleMat{T}}; atol=1e-6) where {T}
    # Convert to ellipse first
    ellipse = Ellipse(Q)

    # Check if it's geometrically a circle (a ≈ b)
    if !isapprox(ellipse.a, ellipse.b; atol=atol)
        error("Conic does not represent a circle: a=$(ellipse.a), b=$(ellipse.b) (difference: $(abs(ellipse.a - ellipse.b)))")
    end

    # Return circle with average of a and b as radius
    r = (ellipse.a + ellipse.b) / 2
    return GeometryBasics.Circle(ellipse.center, r)
end
