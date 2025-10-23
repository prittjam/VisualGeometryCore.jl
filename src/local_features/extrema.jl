"""
Extrema detection in scale space following VLFeat's implementation.
"""

"""
    Extremum3D

A 3D extremum in scale space with integer and refined (sub-pixel) coordinates.

# Fields
- `octave::Int`: Octave index (negative for upsampled, positive for downsampled)
- `xi::Int`, `yi::Int`, `zi::Int`: Integer coordinates in octave space (1-indexed)
- `x::Float64`, `y::Float64`, `z::Float64`: Refined sub-pixel coordinates in octave space
- `sigma::Float64`: Scale (sigma) at refined position
- `step::Float64`: Sampling step (2^octave) for coordinate conversion
- `peak_score::Float64`: Interpolated response value at refined position
- `edge_score::Float64`: Edge rejection score (ratio of Hessian eigenvalues)

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
    peak_score::Float64
    edge_score::Float64
end

"""
    nonmaximal_suppression(octave_3d::Array{Gray{Float32},3}, threshold::Float64)

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
function nonmaximal_suppression(octave_3d::Array{Gray{Float32},3}, threshold::Float64)
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
- `extremum::Union{Extremum3D,Nothing}`: Refined extremum with sub-pixel coordinates, or `nothing` if refinement failed
"""
function refine_extremum_3d(derivatives::NamedTuple,
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

    # Initialize position (x, y, z) as SVector for elegant indexing
    # Note: In array indexing we use (y, x, z) order
    pos = SVector(x, y, z)

    # Iterative refinement (up to 5 iterations like VLFeat)
    local b, g, val_center
    local H  # Hessian matrix

    for iter in 1:5
        # Bounds check using broadcasting
        if any(pos .< SVector(2, 2, 2)) || any(pos .> SVector(width-1, height-1, depth-1))
            return nothing
        end

        # Access pre-computed derivatives at current INTEGER position
        # Array indexing: [y, x, z]
        val_center = Float64(response_3d[pos.y, pos.x, pos.z].val)

        # First derivatives (gradient)
        g = SVector(
            Float64(dx_3d[pos.y, pos.x, pos.z].val),
            Float64(dy_3d[pos.y, pos.x, pos.z].val),
            Float64(dz_3d[pos.y, pos.x, pos.z].val)
        )

        # Second derivatives (Hessian)
        H = @SMatrix [
            Float64(dxx_3d[pos.y, pos.x, pos.z].val)  Float64(dxy_3d[pos.y, pos.x, pos.z].val)  Float64(dxz_3d[pos.y, pos.x, pos.z].val);
            Float64(dxy_3d[pos.y, pos.x, pos.z].val)  Float64(dyy_3d[pos.y, pos.x, pos.z].val)  Float64(dyz_3d[pos.y, pos.x, pos.z].val);
            Float64(dxz_3d[pos.y, pos.x, pos.z].val)  Float64(dyz_3d[pos.y, pos.x, pos.z].val)  Float64(dzz_3d[pos.y, pos.x, pos.z].val)
        ]

        # Check if Hessian is singular
        if abs(det(H)) < 1e-10
            b = @SVector [0.0, 0.0, 0.0]
            break
        end

        # Solve H·b = -g to find peak of fitted 3D paraboloid
        b = H \ (-g)

        # If offset is large, move to adjacent INTEGER pixel
        # IMPORTANT: VLFeat only moves in x,y, NOT z!
        d = SVector(
            (b.x > 0.6 && pos.x < width-1 ? 1 : 0) + (b.x < -0.6 && pos.x > 2 ? -1 : 0),
            (b.y > 0.6 && pos.y < height-1 ? 1 : 0) + (b.y < -0.6 && pos.y > 2 ? -1 : 0),
            0  # VLFeat does NOT move in z direction
        )

        # Update position
        pos += d

        # Converged if no movement needed in x,y (check only spatial components)
        if iszero(d[SVector(1, 2)])  # Check x,y components only
            break
        end
    end

    # =========================================================================
    # VALIDITY CHECKS - Fail fast before computing final values
    # =========================================================================

    # Check 1: Offset validity
    any(abs.(b) .>= 1.5) && return nothing

    # Check 2: Refined position bounds
    refined_pos = pos + b
    (any(refined_pos .< 1) || any(refined_pos .> SVector(width, height, depth))) && return nothing

    # Check 3: Saddle point rejection (det_H < 0)
    # Reject saddle points - these have opposite-sign eigenvalues
    det_H = H[1,1] * H[2,2] - H[1,2] * H[1,2]  # Dxx*Dyy - Dxy²
    det_H < 0 && return nothing  # Saddle point - not a true extremum

    # =========================================================================
    # COMPUTE FINAL VALUES - All validity checks passed
    # =========================================================================

    # Compute peak score using dot product
    # This is the second-order Taylor expansion at the refined position:
    # f(x₀ + b) = f(x₀) + ½·g^T·b  (where H·b = -g)
    peak_score = val_center + 0.5 * dot(g, b)

    # Compute edge score (2D Hessian in spatial plane only)
    # Edge score measures the ratio of principal curvatures to detect edges
    trace_H = H[1,1] + H[2,2]  # Dxx + Dyy
    alpha = (trace_H * trace_H) / det_H
    edge_score = 0.5 * alpha - 1.0 + sqrt(max(0.25 * alpha - 1.0, 0.0) * alpha)

    # Compute sigma at refined scale position
    subdivision = first_subdivision + (pos.z - 1)
    sigma = base_scale * 2.0^(octave_num + (refined_pos.z + first_subdivision - 1) / octave_resolution)

    # Create refined extremum
    extremum = Extremum3D(octave_num, pos.x, pos.y, pos.z,
                         refined_pos.x, refined_pos.y, refined_pos.z,
                         sigma, step, peak_score, edge_score)
    return extremum
end

"""
    find_extrema_3d(response::ScaleSpaceResponse, threshold::Float64=0.0)

Find discrete extrema across all octaves using non-maximal suppression.

This function applies a 26-neighbor check (3×3×3 cube in scale-space) to find
local maxima and minima at integer grid positions. Points are kept only if they
are strictly greater (maxima) or strictly less (minima) than all 26 neighbors.

Returns extrema grouped by octave to avoid flattening and re-grouping later.

# Arguments
- `response`: ScaleSpaceResponse containing responses (e.g., Hessian determinant)
- `threshold`: Minimum absolute value for extrema detection (default: 0.0)

# Returns
- `Vector{Tuple{Int, Vector{Tuple{Int,Int,Int}}}}`: List of (octave_index, positions)
  where positions is a vector of (x, y, z) coordinates in that octave

# Example
```julia
hessian_det = hessian_determinant_response(ixx, iyy, ixy)
discrete = find_extrema_3d(hessian_det, 0.8 * peak_threshold)
# discrete = [(octave1, [(x1,y1,z1), (x2,y2,z2), ...]), (octave2, [...]), ...]
```
"""
function find_extrema_3d(response::ScaleSpaceResponse, threshold::Float64=0.0)
    # Pre-allocate result with known size
    result = Vector{Tuple{Int, Vector{Tuple{Int,Int,Int}}}}()
    sizehint!(result, length(response.octaves))

    # Process each octave and return octave-grouped structure
    for octave in response.octaves
        # Find discrete extrema in this octave using 26-neighbor check
        positions = nonmaximal_suppression(octave.cube, threshold)

        # Store (octave_index, positions) tuple
        push!(result, (octave.index, positions))
    end

    return result
end

"""
    refine_extrema(response::ScaleSpaceResponse,
                   discrete_extrema::Vector{Tuple{Int, Vector{Tuple{Int,Int,Int}}}};
                   peak_threshold::Float64=0.001,
                   edge_threshold::Float64=10.0,
                   base_scale::Float64=1.6,
                   octave_resolution::Int=3)

Refine discrete extrema to sub-pixel accuracy with filtering.

Takes octave-grouped extrema positions (from `find_extrema_3d`) and
refines them using quadratic interpolation in 3D. Filters out extrema that fail
validation or don't meet the peak and edge thresholds.

# Arguments
- `response`: ScaleSpaceResponse containing responses (e.g., Hessian determinant)
- `discrete_extrema`: Vector of (octave_index, positions) from `find_extrema_3d`
  where positions is Vector{Tuple{Int,Int,Int}} of (x, y, z) coordinates
- `peak_threshold`: Minimum absolute peak score (default: 0.001)
- `edge_threshold`: Maximum edge score for rejection (default: 10.0)
- `base_scale`: Base scale parameter (default: 1.6)
- `octave_resolution`: Number of subdivisions per octave (default: 3)

# Returns
- `Vector{Extremum3D}`: Refined and filtered extrema with sub-pixel coordinates

# Example
```julia
# Step-wise processing
hessian_det = hessian_determinant_response(ixx, iyy, ixy)
discrete = find_extrema_3d(hessian_det, 0.8 * 0.003)
refined = refine_extrema(hessian_det, discrete,
    peak_threshold=0.003,
    edge_threshold=10.0)
```
"""
function refine_extrema(response::ScaleSpaceResponse,
                       discrete_extrema::Vector{Tuple{Int, Vector{Tuple{Int,Int,Int}}}};
                       peak_threshold::Float64=0.001,
                       edge_threshold::Float64=10.0,
                       base_scale::Float64=1.6,
                       octave_resolution::Int=3)
    # Pre-compute 3D derivatives for the entire response (GPU-friendly)
    derivatives_resp = (
        ∇x = ScaleSpaceResponse(response, DERIVATIVE_KERNELS_3D.dx),
        ∇y = ScaleSpaceResponse(response, DERIVATIVE_KERNELS_3D.dy),
        ∇z = ScaleSpaceResponse(response, DERIVATIVE_KERNELS_3D.dz),
        ∇²xx = ScaleSpaceResponse(response, DERIVATIVE_KERNELS_3D.dxx),
        ∇²yy = ScaleSpaceResponse(response, DERIVATIVE_KERNELS_3D.dyy),
        ∇²zz = ScaleSpaceResponse(response, DERIVATIVE_KERNELS_3D.dzz),
        ∇²xy = ScaleSpaceResponse(response, DERIVATIVE_KERNELS_3D.dxy),
        ∇²xz = ScaleSpaceResponse(response, DERIVATIVE_KERNELS_3D.dxz),
        ∇²yz = ScaleSpaceResponse(response, DERIVATIVE_KERNELS_3D.dyz)
    )

    # Count total discrete extrema (upper bound before filtering)
    total_discrete = sum(length(positions) for (_, positions) in discrete_extrema)

    # Pre-allocate result with upper bound
    all_extrema = Vector{Extremum3D}()
    sizehint!(all_extrema, total_discrete)

    # Process each octave and refine extrema
    for (octave_num, positions) in discrete_extrema
        # Skip if no extrema in this octave
        isempty(positions) && continue

        # Get octave metadata
        octave = response[octave_num]
        first_subdivision = first(octave.subdivisions)
        step = octave.step

        # Pre-compute derivatives for this octave
        derivatives_cubes = (
            response = octave.cube,
            ∇x = derivatives_resp.∇x[octave_num].cube,
            ∇y = derivatives_resp.∇y[octave_num].cube,
            ∇z = derivatives_resp.∇z[octave_num].cube,
            ∇²xx = derivatives_resp.∇²xx[octave_num].cube,
            ∇²yy = derivatives_resp.∇²yy[octave_num].cube,
            ∇²zz = derivatives_resp.∇²zz[octave_num].cube,
            ∇²xy = derivatives_resp.∇²xy[octave_num].cube,
            ∇²xz = derivatives_resp.∇²xz[octave_num].cube,
            ∇²yz = derivatives_resp.∇²yz[octave_num].cube
        )

        # Refine and filter extrema for this octave
        for (x, y, z) in positions
            extremum = refine_extremum_3d(derivatives_cubes, x, y, z,
                                         octave_num, first_subdivision,
                                         octave_resolution, base_scale, step)

            # Filter by thresholds
            # Only accept blobs (positive det(H)), not saddles (negative det(H))
            # VLFeat's Hessian detector only outputs blobs
            if extremum !== nothing &&
               extremum.peak_score >= peak_threshold &&
               extremum.edge_score < edge_threshold
                push!(all_extrema, extremum)
            end
        end
    end

    return all_extrema
end

"""
    detect_features(img;
                    method::Symbol=:hessian_laplace,
                    peak_threshold::Float64=0.001,
                    edge_threshold::Float64=10.0,
                    first_octave::Int=-1,
                    octave_resolution::Int=3,
                    first_subdivision::Int=-1,
                    last_subdivision::Int=3,
                    base_scale::Float64=2.015874)

Top-level blob detection from an image using Hessian-Laplace detector.

This is the highest-level API - just pass an image and get blob detections back.
Matches VLFeat's `vl_covdet_new(VL_COVDET_METHOD_HESSIAN_LAPLACE)` workflow.

# Arguments
- `img`: Input image (Gray or color, will be converted to Gray{Float32})
- `method`: Detection method (currently only `:hessian_laplace` supported)
- `peak_threshold`: Minimum absolute peak score (default: 0.001)
- `edge_threshold`: Maximum edge score for rejection (default: 10.0)
- `first_octave`: First octave index (default: -1, meaning 2× upsampled)
- `octave_resolution`: Number of subdivisions per octave (default: 3)
- `first_subdivision`: First subdivision index (default: -1)
- `last_subdivision`: Last subdivision index (default: 3)
- `base_scale`: Base scale parameter (default: 2.015874, VLFeat's sqrt(2)^1.5)

# Returns
- `Vector{IsoBlobDetection}`: Detected blob features with:
  - Position and scale in pixel units (pd)
  - **Coordinate convention**: OpenCV/VLFeat (first pixel center at (0, 0))
  - Response strength (Hessian determinant value)
  - Edge score
  - Laplacian scale score for bright/dark classification:
    - `< 0`: Bright blob (intensity peak)
    - `> 0`: Dark blob (intensity valley)
    - `NaN`: Laplacian not computed (basic Hessian detector)

Use `change_image_origin(blob; from=:vlfeat, to=:matlab)` or
`change_image_origin(blob; from=:opencv, to=:makie)` to convert
to other coordinate conventions if needed.

# Example
```julia
using FileIO

# Load image
img = load("test.png")

# Detect all blobs
blobs = detect_features(img)

# Filter by type
bright_blobs = filter(b -> b.laplacian_scale_score < 0, blobs)
dark_blobs = filter(b -> b.laplacian_scale_score > 0, blobs)
```

See also: `detect_features(::ScaleSpaceResponse)` for lower-level control.
"""
function detect_features(img;
                        method::Symbol=:hessian_laplace,
                        peak_threshold::Float64=0.001,
                        edge_threshold::Float64=10.0,
                        first_octave::Int=-1,
                        octave_resolution::Int=3,
                        first_subdivision::Int=-1,
                        last_subdivision::Int=3,
                        base_scale::Float64=2.015874,
                        pixel_convention::Symbol=:colmap)
    # Only support hessian_laplace for now
    if method != :hessian_laplace
        throw(ArgumentError("Only :hessian_laplace method is currently supported"))
    end

    # Convert to Gray{Float32} if needed
    img_gray = Gray{Float32}.(img)

    # Build scale space
    ss = ScaleSpace(img_gray;
                   first_octave=first_octave,
                   octave_resolution=octave_resolution,
                   first_subdivision=first_subdivision,
                   last_subdivision=last_subdivision)

    # Compute Hessian components
    ixx = ScaleSpaceResponse(ss, DERIVATIVE_KERNELS.xx)
    iyy = ScaleSpaceResponse(ss, DERIVATIVE_KERNELS.yy)
    ixy = ScaleSpaceResponse(ss, DERIVATIVE_KERNELS.xy)

    # Compute determinant (for detection) and Laplacian (for bright/dark classification)
    hessian_det = hessian_determinant_response(ixx, iyy, ixy)
    laplacian = laplacian_response(ixx, iyy)

    # Detect features using the ScaleSpaceResponse method
    blobs = detect_features(hessian_det;
                          peak_threshold=peak_threshold,
                          edge_threshold=edge_threshold,
                          base_scale=base_scale,
                          octave_resolution=octave_resolution,
                          laplacian_resp=laplacian)

    # Convert to requested pixel coordinate convention (default :colmap)
    # IsoBlobDetection uses 0-indexed coordinates (like :opencv)
    return change_image_origin.(blobs; from=:opencv, to=pixel_convention)
end

"""
    detect_features(response::ScaleSpaceResponse;
                    peak_threshold::Float64=0.001,
                    edge_threshold::Float64=10.0,
                    base_scale::Float64=1.6,
                    octave_resolution::Int=3,
                    laplacian_resp::Union{Nothing,ScaleSpaceResponse}=nothing)

Detect blob features in scale space using VLFeat's Hessian detector algorithm.

This is a mid-level API for when you already have computed Hessian responses.
For simple image-to-blobs detection, use `detect_features(img)` instead.

# Arguments
- `response`: ScaleSpaceResponse containing Hessian determinant responses
- `peak_threshold`: Minimum absolute peak score (default: 0.001)
- `edge_threshold`: Maximum edge score for rejection (default: 10.0)
- `base_scale`: Base scale parameter (default: 1.6)
- `octave_resolution`: Number of subdivisions per octave (default: 3)
- `laplacian_resp`: Optional Laplacian response for bright/dark blob classification

# Returns
- `Vector{IsoBlobDetection}`: Detected blob features

# Example
```julia
# Build scale space
ss = ScaleSpace(img; first_octave=-1, octave_resolution=3)

# Compute Hessian components
ixx = ScaleSpaceResponse(ss, DERIVATIVE_KERNELS.xx)
iyy = ScaleSpaceResponse(ss, DERIVATIVE_KERNELS.yy)
ixy = ScaleSpaceResponse(ss, DERIVATIVE_KERNELS.xy)

# Compute determinant and Laplacian
hessian_det = hessian_determinant_response(ixx, iyy, ixy)
laplacian = laplacian_response(ixx, iyy)

# Detect features
features = detect_features(hessian_det;
    peak_threshold=0.001,
    edge_threshold=10.0,
    laplacian_resp=laplacian)
```

See also: `detect_features(img)` for top-level API, `find_extrema_3d`, `refine_extrema` for low-level access.
"""
function detect_features(extrema::Vector{Extremum3D}; laplacian_resp::Union{Nothing,ScaleSpaceResponse}=nothing)
    # Convert to IsoBlobDetection (matches VLFeat's VlCovDetFeature output)
    if laplacian_resp === nothing
        # No Laplacian available - use default NaN
        return [IsoBlobDetection(extremum) for extremum in extrema]
    else
        # Interpolate Laplacian values for each extremum
        return [IsoBlobDetection(extremum, interpolate_laplacian(laplacian_resp, extremum))
                for extremum in extrema]
    end
end

function detect_features(response::ScaleSpaceResponse;
                        peak_threshold::Float64=0.001,
                        edge_threshold::Float64=10.0,
                        base_scale::Float64=1.6,
                        octave_resolution::Int=3,
                        laplacian_resp::Union{Nothing,ScaleSpaceResponse}=nothing)
    # Find discrete extrema (use 0.8× threshold for initial detection)
    discrete = find_extrema_3d(response, 0.8 * peak_threshold)

    # Refine extrema to sub-pixel accuracy
    refined = refine_extrema(response, discrete;
                            peak_threshold=peak_threshold,
                            edge_threshold=edge_threshold,
                            base_scale=base_scale,
                            octave_resolution=octave_resolution)

    # Convert using the other method
    return detect_features(refined; laplacian_resp=laplacian_resp)
end

# Deprecated alias for backward compatibility - returns raw Extremum3D
function detect_extrema(args...; kwargs...)
    @warn "detect_extrema is deprecated, use detect_features for IsoBlobDetection or refine_extrema for Extremum3D" maxlog=1

    # Keep old behavior: return (refined, discrete) tuple with Extremum3D
    response = args[1]
    discrete = find_extrema_3d(response, 0.8 * get(kwargs, :peak_threshold, 0.001))
    refined = refine_extrema(response, discrete;
                            peak_threshold=get(kwargs, :peak_threshold, 0.001),
                            edge_threshold=get(kwargs, :edge_threshold, 10.0),
                            base_scale=get(kwargs, :base_scale, 1.6),
                            octave_resolution=get(kwargs, :octave_resolution, 3))
    return (refined, discrete)
end

"""
    IsoBlobDetection(extremum::Extremum3D, laplacian_itp=nothing)

Convert an Extremum3D to an IsoBlobDetection.

Converts from octave space coordinates to input image coordinates (in `pd` units)
and optionally computes the Laplacian value for bright vs dark blob classification.

Matches VLFeat's VlCovDetFeature output structure.

# Arguments
- `extremum::Extremum3D`: Detected extremum in octave space
- `laplacian_itp`: Optional interpolation object for Laplacian response (for bright/dark classification)

# Returns
- `IsoBlobDetection`: Blob detection with:
  - `center`: Position in input image coordinates (pd units)
  - `σ`: Scale (sigma) in pd units
  - `response`: Peak score (Hessian determinant value, always positive)
  - `edge_score`: Edge rejection score
  - `laplacian_scale_score`: Laplacian (trace of Hessian) value

# Coordinate Conversion
Converts from octave space to input image space:
```julia
input_x = (extremum.x - 1) * extremum.step
input_y = (extremum.y - 1) * extremum.step
```

# Bright vs Dark Classification

All returned extrema are true blobs because saddle points (det(H) < 0) are rejected
during subpixel refinement. The Hessian determinant is ALWAYS positive since both
eigenvalues have the same sign.

To distinguish **bright vs dark blobs**, use the Laplacian (trace of Hessian) value:
- `laplacian_scale_score < 0`: Bright blob (intensity peak, Lxx + Lyy < 0)
- `laplacian_scale_score > 0`: Dark blob (intensity valley, Lxx + Lyy > 0)
- `laplacian_scale_score ≈ 0`: Neutral/unknown (edge-like or not computed)

This matches VLFeat's behavior:
- Basic Hessian detector: all detections are blobs (laplacianScaleScore = 0.0)
- HESSIAN_LAPLACE variant: adds bright/dark classification via `laplacianScaleScore`

# Example
```julia
# Without Laplacian (laplacian_scale_score = NaN)
blobs = [IsoBlobDetection(ext) for ext in extrema]

# With Laplacian (bright/dark classification)
laplacian_resp = laplacian_response(ixx, iyy)
blobs = detect_features(extrema; laplacian_resp=laplacian_resp)

# Filter by bright/dark
bright_blobs = filter(b -> b.laplacian_scale_score < 0, blobs)
dark_blobs = filter(b -> b.laplacian_scale_score > 0, blobs)
```
"""

"""
    interpolate_laplacian(laplacian_resp::ScaleSpaceResponse, extremum::Extremum3D) -> Float64

Interpolate the Laplacian response at the refined extremum position.

# Arguments
- `laplacian_resp::ScaleSpaceResponse`: Laplacian response in scale space
- `extremum::Extremum3D`: Refined extremum with subpixel coordinates

# Returns
- `Float64`: Interpolated Laplacian value at the extremum position
"""
function interpolate_laplacian(laplacian_resp::ScaleSpaceResponse, extremum::Extremum3D)
    # Get the octave data
    octave = laplacian_resp[extremum.octave]

    # Create interpolator for this octave's cube
    itp = interpolate(octave.cube, BSpline(Linear()))

    # Interpolate at refined subpixel coordinates (y, x, z order for Arrays)
    return Float64(itp(extremum.y, extremum.x, extremum.z))
end

function IsoBlobDetection(extremum::Extremum3D, laplacian_scale_score::Float64=NaN)
    # Convert from octave space to input image coordinates
    input_x = (extremum.x - 1) * extremum.step
    input_y = (extremum.y - 1) * extremum.step

    # Create center point with pd units
    center = Point2(input_x * pd, input_y * pd)

    # Use peak_score directly (always positive since det(H) < 0 cases are rejected)
    response = extremum.peak_score

    return IsoBlobDetection(center, extremum.sigma * pd, response, extremum.edge_score, laplacian_scale_score)
end
