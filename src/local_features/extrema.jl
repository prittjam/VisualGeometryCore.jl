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
    laplacian::Float64  # Trace of Hessian (Lxx + Lyy) for bright/dark polarity
end

"""
    find_extrema_3d(octave_3d::Array{Gray{Float32},3}, threshold::Float64)

Find discrete local extrema in a 3D scale-space octave using vectorized operations.

Checks each interior point against its 26 neighbors (3×3×3 cube) efficiently
using array views and broadcasting.

# Arguments
- `octave_3d`: 3D array [height, width, depth] containing Gray{Float32} response values
- `threshold`: Minimum absolute value for extrema (0.8 × peak_threshold typically)

# Returns
- `Vector{Tuple{Int,Int,Int}}`: List of (x, y, z) integer coordinates of extrema

# Implementation
Uses a view-based approach for efficiency and clarity:
1. Creates a view of the interior (non-boundary) region as `center`
2. Pre-creates 26 array views for each neighbor direction (cheap, no allocation)
3. Accumulates min/max differences across all neighbors via broadcasting
   - If minDiff > 0 → center is strictly greater than ALL neighbors (maximum)
   - If maxDiff < 0 → center is strictly less than ALL neighbors (minimum)
4. Combines masks and uses `findall` for coordinate extraction
5. No loops over candidate extrema, fully vectorized

# Performance Notes
- Uses Float32 precision (matching VLFeat's float type)
- Views have no memory overhead (just pointer arithmetic)
- All 26-neighbor comparisons use broadcasting
- Coordinate extraction uses `findall` (optimized builtin)
"""
function find_extrema_3d(octave_3d::Array{Gray{Float32},3}, threshold::Float64)
    H, W, L = size(octave_3d)

    # Extract numeric values (Float32, matching VLFeat's float precision)
    D = Float32.(getfield.(octave_3d, :val))

    # Create view of interior (non-boundary) region
    center = @view D[2:H-1, 2:W-1, 2:L-1]

    # Pre-create views for all 26 neighbors (cheap!)
    # These are the 3×3×3 cube around center, excluding center itself
    neighbors = [
        @view D[2+dy:H-1+dy, 2+dx:W-1+dx, 2+dz:L-1+dz]
        for dz in -1:1, dy in -1:1, dx in -1:1
        if !(dx==0 && dy==0 && dz==0)
    ]

    # Accumulate min and max differences across all neighbors
    minDiff = fill(Float32(Inf), size(center))
    maxDiff = fill(Float32(-Inf), size(center))

    for neigh in neighbors
        diff = center .- neigh
        @. minDiff = min(minDiff, diff)
        @. maxDiff = max(maxDiff, diff)
    end

    # Find maxima (strictly greater than all neighbors)
    maxima_mask = (minDiff .> 0) .& (center .>= threshold)
    # Find minima (strictly less than all neighbors)
    minima_mask = (maxDiff .< 0) .& (center .<= -threshold)

    # Convert mask to coordinates (x, y, z) - vectorized
    extrema_mask = maxima_mask .| minima_mask
    extrema_indices = findall(extrema_mask)

    # Convert to (x, y, z) tuples, adjusting from center region to full octave coords
    # CartesianIndex gives (y, x, z), so we swap to (x, y, z) and add 1 for offset
    extrema = [(idx[2]+1, idx[1]+1, idx[3]+1) for idx in extrema_indices]

    return extrema
end

"""
    refine_extremum_3d(derivatives::NamedTuple, x::Int, y::Int, z::Int,
                       octave_num::Int, first_subdivision::Int, octave_resolution::Int,
                       base_scale::Float64, step::Float64)

Refine an extremum to sub-pixel accuracy using pre-computed 3D derivative responses.

This version uses pre-computed derivatives from `DERIVATIVE_KERNELS_3D` applied to
the entire octave cube, enabling GPU-friendly batch processing.

# Arguments
- `derivatives`: NamedTuple with fields (response, ∇x, ∇y, ∇z, ∇²xx, ∇²yy, ∇²zz, ∇²xy, ∇²xz, ∇²yz)
  Each field is a 3D array [height, width, depth] containing the derivative values
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
"""
function refine_extremum_3d(derivatives::NamedTuple, laplacian_cube::Union{Nothing,AbstractArray},
                           x::Int, y::Int, z::Int,
                           octave_num::Int, first_subdivision::Int, octave_resolution::Int,
                           base_scale::Float64, step::Float64)
    # Extract derivative arrays from NamedTuple
    response_3d = derivatives.response
    dx_3d = derivatives.∇x
    dy_3d = derivatives.∇y
    dz_3d = derivatives.∇z
    dxx_3d = derivatives.∇²xx
    dyy_3d = derivatives.∇²yy
    dzz_3d = derivatives.∇²zz
    dxy_3d = derivatives.∇²xy
    dxz_3d = derivatives.∇²xz
    dyz_3d = derivatives.∇²yz

    height, width, depth = size(response_3d)

    # Initialize position and step variables
    x_current = x
    y_current = y
    z_current = z
    dx, dy, dz = 0, 0, 0

    # Iterative refinement (up to 5 iterations like VLFeat)
    local b, Dx, Dy, Dz, Dxx, Dyy, Dzz, Dxy, Dxz, Dyz, val_center

    for iter in 1:5
        x_current += dx
        y_current += dy
        z_current += dz

        # Bounds check
        if x_current < 2 || x_current > width-1 || y_current < 2 || y_current > height-1 || z_current < 2 || z_current > depth-1
            return (nothing, false)
        end

        # Access pre-computed derivatives at current INTEGER position
        val_center = Float64(response_3d[y_current, x_current, z_current].val)
        Dx = Float64(dx_3d[y_current, x_current, z_current].val)
        Dy = Float64(dy_3d[y_current, x_current, z_current].val)
        Dz = Float64(dz_3d[y_current, x_current, z_current].val)
        Dxx = Float64(dxx_3d[y_current, x_current, z_current].val)
        Dyy = Float64(dyy_3d[y_current, x_current, z_current].val)
        Dzz = Float64(dzz_3d[y_current, x_current, z_current].val)
        Dxy = Float64(dxy_3d[y_current, x_current, z_current].val)
        Dxz = Float64(dxz_3d[y_current, x_current, z_current].val)
        Dyz = Float64(dyz_3d[y_current, x_current, z_current].val)

        # Solve H·b = -g to find peak of fitted 3D paraboloid
        H = @SMatrix [Dxx Dxy Dxz; Dxy Dyy Dyz; Dxz Dyz Dzz]
        g = @SVector [Dx, Dy, Dz]

        # Check if Hessian is singular
        if abs(det(H)) < 1e-10
            b = @SVector [0.0, 0.0, 0.0]
            break
        end

        b = H \ (-g)

        # If offset is large, move to adjacent INTEGER pixel
        # IMPORTANT: VLFeat only moves in x,y, NOT z!
        dx = (b[1] > 0.6 && x_current < width-1 ? 1 : 0) + (b[1] < -0.6 && x_current > 2 ? -1 : 0)
        dy = (b[2] > 0.6 && y_current < height-1 ? 1 : 0) + (b[2] < -0.6 && y_current > 2 ? -1 : 0)
        dz = 0  # VLFeat does NOT move in z direction

        # Converged if no movement needed
        if dx == 0 && dy == 0
            break
        end
    end

    # Compute peak score
    peakScore = val_center + 0.5 * (Dx * b[1] + Dy * b[2] + Dz * b[3])

    # Compute edge score
    trace_H = Dxx + Dyy
    det_H = Dxx * Dyy - Dxy * Dxy

    edgeScore = if det_H < 0
        Inf
    else
        alpha = (trace_H * trace_H) / det_H
        0.5 * alpha - 1.0 + sqrt(max(0.25 * alpha - 1.0, 0.0) * alpha)
    end

    # Check offset validity
    if abs(b[1]) >= 1.5 || abs(b[2]) >= 1.5 || abs(b[3]) >= 1.5
        return (nothing, false)
    end

    # Compute refined position
    refined_x = x_current + b[1]
    refined_y = y_current + b[2]
    refined_z = z_current + b[3]

    # Final bounds check
    if refined_x < 1 || refined_x > width || refined_y < 1 || refined_y > height ||
       refined_z < 1 || refined_z > depth
        return (nothing, false)
    end

    # Compute sigma
    subdivision = first_subdivision + (z_current - 1)
    sigma = base_scale * 2.0^(octave_num + (refined_z + first_subdivision - 1) / octave_resolution)

    # Get Laplacian value from pre-computed cube (if available)
    # This is the Laplacian of the IMAGE, not of the response
    laplacian = if laplacian_cube !== nothing
        Float64(laplacian_cube[y_current, x_current, z_current].val)
    else
        # Fallback: compute from response derivatives (less accurate)
        Dxx + Dyy
    end

    extremum = Extremum3D(octave_num, x_current, y_current, z_current,
                         refined_x, refined_y, refined_z,
                         sigma, step, peakScore, edgeScore, laplacian)
    return (extremum, true)
end

"""
    detect_extrema(response::ScaleSpaceResponse;
                   peak_threshold::Float64=0.001,
                   edge_threshold::Float64=10.0,
                   base_scale::Float64=1.6,
                   octave_resolution::Int=3,
                   return_discrete::Bool=false,
                   laplacian_resp::Union{Nothing,ScaleSpaceResponse}=nothing)

Detect scale-space extrema in response pyramid using VLFeat's algorithm.

# Arguments
- `response`: ScaleSpaceResponse containing Hessian determinant responses
- `peak_threshold`: Minimum absolute peak score (default: 0.001)
- `edge_threshold`: Maximum edge score for rejection (default: 10.0)
- `base_scale`: Base scale parameter (default: 1.6)
- `octave_resolution`: Number of subdivisions per octave (default: 3)
- `return_discrete`: Return discrete extrema locations (default: false)
- `laplacian_resp`: Optional Laplacian response for bright/dark blob classification

# Returns
- `Vector{Extremum3D}`: Detected and refined extrema
- If `return_discrete=true`: Tuple of (extrema, discrete_locations)

# Algorithm
1. For each octave, find discrete extrema using 26-neighbor check
2. Refine to sub-pixel accuracy using quadratic interpolation
3. Filter by peak threshold and edge threshold
4. Classify bright/dark blobs using Laplacian if provided

# Example
```julia
# Compute Hessian components
ixx = ScaleSpaceResponse(ss, DERIVATIVE_KERNELS.xx)(ss)
iyy = ScaleSpaceResponse(ss, DERIVATIVE_KERNELS.yy)(ss)
ixy = ScaleSpaceResponse(ss, DERIVATIVE_KERNELS.xy)(ss)

# Compute determinant and Laplacian
hessian_det = hessian_determinant_response(ixx, iyy, ixy)
laplacian = laplacian_response(ixx, iyy)

# Detect extrema
extrema = detect_extrema(hessian_det;
    peak_threshold=0.001,
    edge_threshold=10.0,
    laplacian_resp=laplacian)
```
"""
function detect_extrema(response::ScaleSpaceResponse;
                       peak_threshold::Float64=0.001,
                       edge_threshold::Float64=10.0,
                       base_scale::Float64=1.6,
                       octave_resolution::Int=3,
                       return_discrete::Bool=false,
                       laplacian_resp::Union{Nothing,ScaleSpaceResponse}=nothing)
    all_extrema = Extremum3D[]
    all_discrete = Tuple{Int,Int,Int,Int}[]  # (octave, x, y, z)

    # Pre-compute 3D derivatives for the entire response (GPU-friendly)
    derivatives_resp = (
        ∇x = ScaleSpaceResponse(response, DERIVATIVE_KERNELS_3D.dx)(response),
        ∇y = ScaleSpaceResponse(response, DERIVATIVE_KERNELS_3D.dy)(response),
        ∇z = ScaleSpaceResponse(response, DERIVATIVE_KERNELS_3D.dz)(response),
        ∇²xx = ScaleSpaceResponse(response, DERIVATIVE_KERNELS_3D.dxx)(response),
        ∇²yy = ScaleSpaceResponse(response, DERIVATIVE_KERNELS_3D.dyy)(response),
        ∇²zz = ScaleSpaceResponse(response, DERIVATIVE_KERNELS_3D.dzz)(response),
        ∇²xy = ScaleSpaceResponse(response, DERIVATIVE_KERNELS_3D.dxy)(response),
        ∇²xz = ScaleSpaceResponse(response, DERIVATIVE_KERNELS_3D.dxz)(response),
        ∇²yz = ScaleSpaceResponse(response, DERIVATIVE_KERNELS_3D.dyz)(response)
    )

    # Process each octave
    for octave in response.octaves
        # Get 3D cube and metadata for this octave
        octave_3d = octave.G
        octave_num = octave.octave
        first_subdivision = first(octave.subdivisions)
        step = octave.step

        # Pre-compute derivatives for this octave
        derivatives_cubes = (
            response = octave_3d,
            ∇x = derivatives_resp.∇x[octave_num].G,
            ∇y = derivatives_resp.∇y[octave_num].G,
            ∇z = derivatives_resp.∇z[octave_num].G,
            ∇²xx = derivatives_resp.∇²xx[octave_num].G,
            ∇²yy = derivatives_resp.∇²yy[octave_num].G,
            ∇²zz = derivatives_resp.∇²zz[octave_num].G,
            ∇²xy = derivatives_resp.∇²xy[octave_num].G,
            ∇²xz = derivatives_resp.∇²xz[octave_num].G,
            ∇²yz = derivatives_resp.∇²yz[octave_num].G
        )

        # Get Laplacian cube if available
        laplacian_cube = laplacian_resp !== nothing ? laplacian_resp[octave_num].G : nothing

        # Find discrete extrema (use 0.8× threshold for initial detection)
        discrete_extrema = find_extrema_3d(octave_3d, 0.8 * peak_threshold)

        # Store discrete extrema if requested
        if return_discrete
            for (x, y, z) in discrete_extrema
                push!(all_discrete, (octave_num, x, y, z))
            end
        end

        # Refine and filter each extremum using pre-computed derivatives
        for (x, y, z) in discrete_extrema
            extremum, converged = refine_extremum_3d(derivatives_cubes, laplacian_cube, x, y, z,
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

"""
    IsoBlobDetection(extremum::Extremum3D)

Convert an Extremum3D to an IsoBlobDetection.

Converts from octave space coordinates to input image coordinates (in `pd` units)
and determines blob polarity based on the sign of the response (peakScore).

# Arguments
- `extremum::Extremum3D`: Detected extremum in octave space

# Returns
- `IsoBlobDetection`: Blob detection with:
  - `center`: Position in input image coordinates (pd units)
  - `σ`: Scale (sigma) in pd units
  - `response`: Absolute value of peakScore
  - `edge_score`: Edge rejection score
  - `polarity`: PositiveFeature (bright blob) or NegativeFeature (dark blob)

# Coordinate Conversion
Converts from octave space to input image space:
```julia
input_x = (extremum.x - 1) * extremum.step
input_y = (extremum.y - 1) * extremum.step
```

# Polarity Determination
- `peakScore > 0` → BlobFeature (blob-like structure: both bright and dark blobs)
- `peakScore < 0` → SaddleFeature (saddle point / edge-like structure)

**IMPORTANT**: The Hessian determinant is POSITIVE for all blob structures (both bright peaks
and dark valleys), since det(H) = Lxx*Lyy > 0 when both eigenvalues have the same sign.
Negative responses indicate saddle points where eigenvalues have opposite signs.

To distinguish bright vs dark blobs, use the Laplacian (trace of Hessian) sign:
- Laplacian < 0: bright blob (Lxx + Lyy < 0, both negative at peak)
- Laplacian > 0: dark blob (Lxx + Lyy > 0, both positive at valley)

VLFeat's basic Hessian detector does not distinguish bright vs dark - use HESSIAN_LAPLACE for that.

# Example
```julia
# Convert VLFeat extrema to blob detections
blobs = [IsoBlobDetection(ext) for ext in extrema]

# Filter by structure type
blob_like = filter(b -> b.polarity == BlobFeature, blobs)
saddle_like = filter(b -> b.polarity == SaddleFeature, blobs)
```
"""
function IsoBlobDetection(extremum::Extremum3D)
    # Convert from octave space to input image coordinates
    input_x = (extremum.x - 1) * extremum.step
    input_y = (extremum.y - 1) * extremum.step

    # Create center point with pd units
    center = Point2(input_x * pd, input_y * pd)

    # Determine structure type based on sign of Hessian determinant
    # IMPORTANT: Hessian determinant is POSITIVE for blobs (both bright and dark)
    # The polarity distinguishes blob-like vs saddle/edge-like structures
    # Positive det(H): BlobFeature (true blobs - both bright and dark)
    # Negative det(H): SaddleFeature (saddle points/edges - not blob-like)
    polarity = extremum.peakScore > 0 ? BlobFeature : SaddleFeature

    # Determine bright vs dark using Laplacian (trace of Hessian)
    # Laplacian < 0: bright blob (intensity peak, Lxx + Lyy < 0)
    # Laplacian > 0: dark blob (intensity valley, Lxx + Lyy > 0)
    laplacian_sign = extremum.laplacian < -1e-10 ? -1 : (extremum.laplacian > 1e-10 ? 1 : 0)

    # Use absolute value for response strength
    response = abs(extremum.peakScore)

    return IsoBlobDetection(center, extremum.sigma * pd, response, extremum.edgeScore, laplacian_sign, polarity)
end
