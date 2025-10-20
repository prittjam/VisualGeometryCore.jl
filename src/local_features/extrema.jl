"""
Extrema detection in scale space following VLFeat's implementation.
"""

using LinearAlgebra
using StaticArrays

"""
    Extremum3D

A 3D extremum in scale space with integer and refined (sub-pixel) coordinates.

# Fields
- `octave::Int`: Octave index (negative for upsampled, positive for downsampled)
- `xi::Int`, `yi::Int`, `zi::Int`: Integer coordinates in octave space (1-indexed)
- `x::Float64`, `y::Float64`, `z::Float64`: Refined sub-pixel coordinates in octave space
- `sigma::Float64`: Scale (sigma) at refined position
- `step::Float64`: Sampling step (2^octave) for coordinate conversion
- `peakScore::Float64`: Interpolated response value at refined position
- `edgeScore::Float64`: Edge rejection score (ratio of Hessian eigenvalues)

# Coordinate Systems

This structure stores coordinates in **octave space** (the coordinates within the
upsampled/downsampled octave image). To convert to **input image coordinates**:

```julia
input_x = (extremum.x - 1) * extremum.step
input_y = (extremum.y - 1) * extremum.step
```

The `-1` accounts for Julia's 1-based indexing, while VLFeat uses 0-based indexing.

## Example
For an extremum at octave=-1 (2× upsampled) with x=105.5, step=0.5:
- Octave coordinate: x = 105.5 (within 256×256 upsampled image)
- Input coordinate: x = (105.5 - 1) × 0.5 = 52.25 (in 128×128 input image)

# See also
- [`refine_extremum_3d`](@ref): Computes refined coordinates
- [`detect_extrema`](@ref): High-level detection interface
"""
struct Extremum3D
    octave::Int
    xi::Int
    yi::Int
    zi::Int
    x::Float64
    y::Float64
    z::Float64
    sigma::Float64
    step::Float64
    peakScore::Float64
    edgeScore::Float64
end

"""
    find_extrema_3d(octave_3d::Array{T,3}, threshold::Float64) where T

Find discrete local extrema in a 3D scale-space octave using vectorized operations.

Checks each interior point against its 26 neighbors (3×3×3 cube) efficiently
using broadcasting and reduction.

# Arguments
- `octave_3d`: 3D array [height, width, depth] containing response values
- `threshold`: Minimum absolute value for extrema (0.8 × peak_threshold typically)

# Returns
- `Vector{Tuple{Int,Int,Int}}`: List of (x, y, z) integer coordinates of extrema

# Implementation
Uses directional-difference reduction for efficient 26-neighbor checks.
For each point, computes min/max of (center - neighbor) across all 26 directions.
If minDiff > 0, then center is strictly greater than ALL neighbors (maximum).
If maxDiff < 0, then center is strictly less than ALL neighbors (minimum).
"""
function find_extrema_3d(octave_3d::Array{T,3}, threshold::Float64) where T
    H, W, L = size(octave_3d)
    I = 2:H-1
    J = 2:W-1
    K = 2:L-1

    # Extract numeric values and get center region
    D = Float32.(getfield.(octave_3d, :val))
    @views center = D[I, J, K]

    # Accumulate min and max differences across all 26 neighbors
    minDiff = fill(Float32(Inf), size(center))
    maxDiff = fill(Float32(-Inf), size(center))

    for dz in -1:1, dy in -1:1, dx in -1:1
        (dx==0 && dy==0 && dz==0) && continue
        @views neigh = D[I .+ dy, J .+ dx, K .+ dz]
        @views diff = center .- neigh
        @. minDiff = min(minDiff, diff)
        @. maxDiff = max(maxDiff, diff)
    end

    # Find maxima (strictly greater than all neighbors)
    maxima_mask = (minDiff .> 0) .& (center .>= threshold)
    # Find minima (strictly less than all neighbors)
    minima_mask = (maxDiff .< 0) .& (center .<= -threshold)

    # Convert mask to coordinates (x, y, z) where z is in K indices
    extrema = Tuple{Int,Int,Int}[]

    for (idx, is_max) in enumerate(maxima_mask)
        if is_max
            # Convert linear index to (y, x, z) in center region
            cart = CartesianIndices(maxima_mask)[idx]
            y, x, z = Tuple(cart)
            # Adjust back to full octave coordinates
            push!(extrema, (x+1, y+1, z+1))
        end
    end

    for (idx, is_min) in enumerate(minima_mask)
        if is_min
            cart = CartesianIndices(minima_mask)[idx]
            y, x, z = Tuple(cart)
            push!(extrema, (x+1, y+1, z+1))
        end
    end

    return extrema
end

"""
    refine_extremum_3d(octave_3d::Array{T,3}, x::Int, y::Int, z::Int,
                       octave_num::Int, first_subdivision::Int, octave_resolution::Int,
                       base_scale::Float64, step::Float64) where T

Refine an extremum to sub-pixel accuracy using quadratic interpolation.

Implements Brown & Lowe's method: iteratively fits a quadratic around the
extremum and moves to the fitted peak location. This implementation exactly
matches VLFeat's behavior, including several critical implementation details.

# Arguments
- `octave_3d`: 3D array [height, width, depth]
- `x, y, z`: Integer coordinates of discrete extremum (1-indexed)
- `octave_num`: Octave index
- `first_subdivision`: First subdivision index in this octave
- `octave_resolution`: Number of subdivisions per octave
- `base_scale`: Base scale parameter (typically 1.6 or 2.015874)
- `step`: Sampling step (2^octave) for coordinate conversion

# Returns
- `(extremum, converged)`: Tuple containing:
  - `extremum::Extremum3D`: Refined extremum with sub-pixel coordinates
  - `converged::Bool`: True if refinement succeeded, false otherwise

# Implementation Details

This function exactly replicates VLFeat's `vl_refine_local_extreum_3`
(covdet.c:1206-1343) with the following critical behaviors:

## Iterative Refinement (up to 5 iterations)
1. Computes gradient ∇f and Hessian H at current position using standard finite differences
   - Second derivatives: `Dxx = f(x+1) + f(x-1) - 2f(x)`
   - Mixed derivatives: `Dxy = 0.25 * (f(x+1,y+1) + f(x-1,y-1) - f(x-1,y+1) - f(x+1,y-1))`
2. Solves H·b = -∇f for offset vector b
3. Moves to adjacent pixel if |b[i]| > 0.6

## Critical VLFeat Behaviors
- **Only moves in x,y dimensions, NOT z** (covdet.c:1279-1285)
  - `dx` and `dy` are updated based on b[1] and b[2]
  - `dz` is always 0 (scale dimension is NOT updated)
  - Convergence checks only x,y: `if dx == 0 && dy == 0`
  - This prevents oscillation in scale space

- **Accepts non-converged refinements** (covdet.c:1286-1314)
  - Computes extremum even if loop doesn't converge
  - Uses last position reached after max iterations
  - Returns true if |b[i]| < 1.5 and within bounds
  - This allows features that oscillate between adjacent pixels

## Rejection Criteria
Returns `(nothing, false)` if:
- Goes out of bounds during iteration
- Final offset |b[i]| >= 1.5 (unstable refinement)
- Refined position outside [1, width] × [1, height] × [1, depth]
- Hessian is singular (|det(H)| < 1e-10)

# References
- VLFeat covdet.c:1206-1343 (`vl_refine_local_extreum_3`)
- Brown & Lowe (2002): "Invariant Features from Interest Point Groups"

# See also
- [`Extremum3D`](@ref): Return type containing refined coordinates
- [`detect_extrema`](@ref): High-level detection interface
"""
function refine_extremum_3d(octave_3d::Array{T,3}, x::Int, y::Int, z::Int,
                           octave_num::Int, first_subdivision::Int, octave_resolution::Int,
                           base_scale::Float64, step::Float64) where T
    height, width, depth = size(octave_3d)

    dx, dy, dz = 0, 0, 0
    max_iterations = 5

    # Track last valid b vector and derivatives for non-convergent case
    b = @SVector [0.0, 0.0, 0.0]
    Dx, Dy, Dz = 0.0, 0.0, 0.0
    Dxx, Dyy, Dxy = 0.0, 0.0, 0.0
    val_center = 0.0

    for iter in 1:max_iterations
        # Update position
        x += dx
        y += dy
        z += dz

        # Bounds check
        if x < 2 || x > width-1 || y < 2 || y > height-1 || z < 2 || z > depth-1
            return (nothing, false)
        end

        # Access octave data at current position (x, y, z updated above)
        # Inline helper macro to avoid function redefinition
        at(ddx, ddy, ddz) = Float64(octave_3d[y+ddy, x+ddx, z+ddz].val)

        # Compute gradient using central differences
        Dx = 0.5 * (at(1, 0, 0) - at(-1, 0, 0))
        Dy = 0.5 * (at(0, 1, 0) - at(0, -1, 0))
        Dz = 0.5 * (at(0, 0, 1) - at(0, 0, -1))

        # Compute Hessian using second differences
        val_center = at(0, 0, 0)
        Dxx = at(1, 0, 0) + at(-1, 0, 0) - 2.0 * val_center
        Dyy = at(0, 1, 0) + at(0, -1, 0) - 2.0 * val_center
        Dzz = at(0, 0, 1) + at(0, 0, -1) - 2.0 * val_center

        Dxy = 0.25 * (at(1, 1, 0) + at(-1, -1, 0) - at(-1, 1, 0) - at(1, -1, 0))
        Dxz = 0.25 * (at(1, 0, 1) + at(-1, 0, -1) - at(-1, 0, 1) - at(1, 0, -1))
        Dyz = 0.25 * (at(0, 1, 1) + at(0, -1, -1) - at(0, -1, 1) - at(0, 1, -1))

        # Solve H·b = -g for offset b
        H = @SMatrix [Dxx Dxy Dxz; Dxy Dyy Dyz; Dxz Dyz Dzz]
        g = @SVector [Dx, Dy, Dz]

        # Check if Hessian is singular
        if abs(det(H)) < 1e-10
            b = @SVector [0.0, 0.0, 0.0]
            break
        end

        b = H \ (-g)

        # Check for sufficient displacement to move to adjacent pixel
        # IMPORTANT: VLFeat only moves in x,y, NOT z! (see covdet.c:1279-1285)
        # This prevents oscillation and matches VLFeat's behavior exactly
        dx = (b[1] > 0.6 && x < width-1 ? 1 : 0) + (b[1] < -0.6 && x > 2 ? -1 : 0)
        dy = (b[2] > 0.6 && y < height-1 ? 1 : 0) + (b[2] < -0.6 && y > 2 ? -1 : 0)
        dz = 0  # VLFeat does NOT move in z direction

        # Converged if no movement needed (only check x,y like VLFeat)
        if dx == 0 && dy == 0
            break
        end
    end

    # CRITICAL: VLFeat computes extremum even if not converged! (covdet.c:1286-1314)
    # It returns the last position reached as long as offset < 1.5 and within bounds
    # All derivatives (Dx, Dy, Dz, Dxx, Dyy, Dxy) and b are already computed at final position

    # Compute peak score (interpolated response value)
    peakScore = val_center + 0.5 * (Dx * b[1] + Dy * b[2] + Dz * b[3])

    # Compute edge score for edge rejection
    # Uses 2D Hessian in spatial dimensions only
    trace_H = Dxx + Dyy
    det_H = Dxx * Dyy - Dxy * Dxy

    edgeScore = if det_H < 0
        Inf  # Not an extremum (saddle point)
    else
        alpha = (trace_H * trace_H) / det_H
        # Formula from VLFeat: edgeScore = (0.5*α - 1) + sqrt(max(0.25*α - 1, 0) * α)
        0.5 * alpha - 1.0 + sqrt(max(0.25 * alpha - 1.0, 0.0) * alpha)
    end

    # Check offset validity (VLFeat's acceptance criteria)
    if abs(b[1]) >= 1.5 || abs(b[2]) >= 1.5 || abs(b[3]) >= 1.5
        return (nothing, false)
    end

    # Compute refined position
    refined_x = x + b[1]
    refined_y = y + b[2]
    refined_z = z + b[3]

    # Final bounds check
    if refined_x < 1 || refined_x > width || refined_y < 1 || refined_y > height ||
       refined_z < 1 || refined_z > depth
        return (nothing, false)
    end

    # Compute sigma at refined position
    # z is 1-indexed, but subdivisions start at first_subdivision
    subdivision = first_subdivision + (z - 1)
    sigma = base_scale * 2.0^(octave_num + (refined_z + first_subdivision - 1) / octave_resolution)

    extremum = Extremum3D(octave_num, x, y, z,
                         refined_x, refined_y, refined_z,
                         sigma, step, peakScore, edgeScore)
    return (extremum, true)
end

"""
    detect_extrema(response::ScaleSpaceResponse;
                   peak_threshold::Float64=0.001,
                   edge_threshold::Float64=10.0)

Detect scale-space extrema in response pyramid using VLFeat's algorithm.

# Arguments
- `response`: ScaleSpaceResponse containing computed responses (e.g., Hessian determinant)
- `peak_threshold`: Minimum absolute peak score (default: 0.001)
- `edge_threshold`: Maximum edge score for rejection (default: 10.0)

# Returns
- `Vector{Extremum3D}`: Detected and refined extrema

# Algorithm
1. For each octave, find discrete extrema using 26-neighbor check
2. Refine to sub-pixel accuracy using quadratic interpolation
3. Filter by peak threshold and edge threshold

# Example
```julia
# Compute Hessian determinant responses
hessian_resp = ScaleSpaceResponse(ss, vlfeat_hessian_det_transform)
hessian_resp(ss)

# Detect extrema
extrema = detect_extrema(hessian_resp, peak_threshold=0.001, edge_threshold=10.0)
```
"""
function detect_extrema(response::ScaleSpaceResponse{T};
                       peak_threshold::Float64=0.001,
                       edge_threshold::Float64=10.0,
                       base_scale::Float64=1.6,
                       octave_resolution::Int=3,
                       return_discrete::Bool=false) where T
    all_extrema = Extremum3D[]
    all_discrete = Tuple{Int,Int,Int,Int}[]  # (octave, x, y, z)

    # Process each octave
    for octave in response.octaves
        # Get 3D cube and metadata for this octave
        octave_3d = octave.G
        octave_num = octave.octave
        first_subdivision = first(octave.subdivisions)
        step = octave.step

        # Find discrete extrema (use 0.8× threshold for initial detection)
        discrete_extrema = find_extrema_3d(octave_3d, 0.8 * peak_threshold)

        # Store discrete extrema if requested
        if return_discrete
            for (x, y, z) in discrete_extrema
                push!(all_discrete, (octave_num, x, y, z))
            end
        end

        # Refine and filter each extremum
        for (x, y, z) in discrete_extrema
            extremum, converged = refine_extremum_3d(octave_3d, x, y, z,
                                                     octave_num, first_subdivision,
                                                     octave_resolution, base_scale, step)

            if !converged || extremum === nothing
                continue
            end

            # Apply thresholds
            if abs(extremum.peakScore) < peak_threshold
                continue
            end

            if extremum.edgeScore >= edge_threshold
                continue
            end

            push!(all_extrema, extremum)
        end
    end

    return return_discrete ? (all_extrema, all_discrete) : all_extrema
end
