"""
Scale space construction for multi-scale blob detection.

This module implements Gaussian scale space similar to VLFeat, with:
- Multi-octave pyramid structure
- Preallocated image storage
- Flexible octave geometry
- Efficient broadcasting operations via StructArrays
"""

using LinearAlgebra
using ImageFiltering
using ImageFiltering: Kernel, imfilter, centered, Fill
using ImageTransformations: imresize
using StructArrays
using FileIO
using Colors: Gray

"""
    ScaleLevel{T}

A single level in the scale space pyramid with associated metadata.
"""
struct ScaleLevel{T}
    data::T
    octave::Int
    scale::Int
    sigma::Float64
    step::Float64
    size::Size2{Int}
end

"""
    ScaleSpace{T}

Preallocated scale space pyramid with efficient StructArray storage for broadcasting.

T can be:
- `Matrix{Float32}` for smoothed images
- `NamedTuple{(:Ixx, :Iyy, :Ixy)}` for Hessian components
- Any other data structure needed for the scale space
"""
struct ScaleSpace{T}
    first_octave::Int
    last_octave::Int
    octave_resolution::Int
    octave_first_subdivision::Int
    octave_last_subdivision::Int
    base_sigma::Float64
    assumed_camera_sigma::Float64
    levels::StructArray{ScaleLevel{T}}
    input_size::Size2{Int}

    # Inner constructor for building scale space with level factory
    function ScaleSpace(first_octave::Int, last_octave::Int, octave_resolution::Int,
                       octave_first_subdivision::Int, octave_last_subdivision::Int,
                       base_sigma::Float64, assumed_camera_sigma::Float64,
                       input_size::Size2{Int}, level_factory::Function)
        octave_range = first_octave:last_octave
        scale_range = octave_first_subdivision:octave_last_subdivision

        # Create combinations in the order expected by level_index:
        # (o0,s0), (o0,s1), (o0,s2), (o1,s0), (o1,s1), (o1,s2), ...
        # NOT (o0,s0), (o1,s0), (o2,s0), (o0,s1), ... which is what product(octave, scale) gives
        octaves = Int[]
        scales = Int[]
        for o in octave_range
            for s in scale_range
                push!(octaves, o)
                push!(scales, s)
            end
        end

        sigmas = base_sigma .* 2.0 .^ (octaves .+ scales ./ octave_resolution)
        step_values = 2.0 .^ octaves
        sizes = [Size2(max(1, round(Int, input_size.width * 2.0^(-o))),
                       max(1, round(Int, input_size.height * 2.0^(-o)))) for o in octaves]

        data_arrays = level_factory.(sizes)

        T = eltype(data_arrays)
        levels = StructArray{ScaleLevel{T}}((
            data = data_arrays,
            octave = octaves,
            scale = scales,
            sigma = sigmas,
            step = step_values,
            size = sizes
        ))

        return new{T}(first_octave, last_octave, octave_resolution,
                     octave_first_subdivision, octave_last_subdivision,
                     base_sigma, assumed_camera_sigma, levels, input_size)
    end
end

# Public constructors
"""
    ScaleSpace(; size=nothing, width=0, height=0, octave_resolution=3,
               base_sigma=1.6, assumed_camera_sigma=0.5, element_type=Float32)

Create a scale space for storing smoothed images with default geometry.
"""
function ScaleSpace(; size::Union{Size2{Int}, Nothing}=nothing, width::Int=0, height::Int=0,
                   octave_resolution::Int=3, base_sigma::Float64=1.6,
                   assumed_camera_sigma::Float64=0.5, element_type::Type{T}=Float32) where T
    if size === nothing
        size = Size2(width, height)
    end

    min_dim = min(size.width, size.height)
    last_octave = max(floor(Int, log2(min_dim)) - 3, 0)

    # VLFeat applies base_sigma scaling: baseScale = 1.6 * 2^(1/octaveResolution)
    adjusted_base_sigma = base_sigma * 2.0^(1.0 / octave_resolution)

    return ScaleSpace(0, last_octave, octave_resolution, 0, octave_resolution - 1,
                     adjusted_base_sigma, assumed_camera_sigma, size,
                     sz -> Matrix{T}(undef, sz.height, sz.width))
end

function Base.similar(ss::ScaleSpace, level_factory::Function)
    return ScaleSpace(ss.first_octave, ss.last_octave, ss.octave_resolution,
                     ss.octave_first_subdivision, ss.octave_last_subdivision,
                     ss.base_sigma, ss.assumed_camera_sigma, ss.input_size,
                     level_factory)
end

"""
    HessianScaleSpace(; size=nothing, width=0, height=0, ...)

Create a scale space for storing Hessian components (Ixx, Iyy, Ixy).
"""
function HessianScaleSpace(; size::Union{Size2{Int}, Nothing}=nothing, width::Int=0, height::Int=0,
                          octave_resolution::Int=3, base_sigma::Float64=1.6,
                          assumed_camera_sigma::Float64=0.5, element_type::Type{T}=Float32) where T
    if size === nothing
        size = Size2(width, height)
    end

    min_dim = min(size.width, size.height)
    last_octave = max(floor(Int, log2(min_dim)) - 3, 0)

    # VLFeat applies base_sigma scaling: baseScale = 1.6 * 2^(1/octaveResolution)
    adjusted_base_sigma = base_sigma * 2.0^(1.0 / octave_resolution)

    return ScaleSpace(0, last_octave, octave_resolution, 0, octave_resolution - 1,
                     adjusted_base_sigma, assumed_camera_sigma, size,
                     sz -> (Ixx = Matrix{T}(undef, sz.height, sz.width),
                           Iyy = Matrix{T}(undef, sz.height, sz.width),
                           Ixy = Matrix{T}(undef, sz.height, sz.width)))
end

HessianScaleSpace(ss::ScaleSpace{<:AbstractMatrix}, element_type::Type{T}=Float32) where T =
    similar(ss, sz -> (Ixx = Matrix{T}(undef, sz.height, sz.width),
                       Iyy = Matrix{T}(undef, sz.height, sz.width),
                       Ixy = Matrix{T}(undef, sz.height, sz.width)))

"""
    LaplacianScaleSpace(; size=nothing, width=0, height=0, ...)

Create a scale space for storing Laplacian responses.
"""
function LaplacianScaleSpace(; size::Union{Size2{Int}, Nothing}=nothing, width::Int=0, height::Int=0,
                            octave_resolution::Int=3, base_sigma::Float64=1.6,
                            assumed_camera_sigma::Float64=0.5, element_type::Type{T}=Float32) where T
    if size === nothing
        size = Size2(width, height)
    end

    min_dim = min(size.width, size.height)
    last_octave = max(floor(Int, log2(min_dim)) - 3, 0)

    # VLFeat applies base_sigma scaling: baseScale = 1.6 * 2^(1/octaveResolution)
    adjusted_base_sigma = base_sigma * 2.0^(1.0 / octave_resolution)

    return ScaleSpace(0, last_octave, octave_resolution, 0, octave_resolution - 1,
                     adjusted_base_sigma, assumed_camera_sigma, size,
                     sz -> Matrix{T}(undef, sz.height, sz.width))
end

LaplacianScaleSpace(ss::ScaleSpace, element_type::Type{T}=Float32) where T =
    similar(ss, sz -> Matrix{T}(undef, sz.height, sz.width))

# Accessors
get_level(ss::ScaleSpace, octave::Int, scale::Int) = ss.levels.data[level_index(ss, octave, scale)]
get_scale_level(ss::ScaleSpace, octave::Int, scale::Int) = ss.levels[level_index(ss, octave, scale)]

function level_index(ss::ScaleSpace, octave::Int, scale::Int)
    @assert ss.first_octave <= octave <= ss.last_octave
    @assert ss.octave_first_subdivision <= scale <= ss.octave_last_subdivision

    scales_per_octave = ss.octave_last_subdivision - ss.octave_first_subdivision + 1
    octave_offset = octave - ss.first_octave
    scale_offset = scale - ss.octave_first_subdivision

    return octave_offset * scales_per_octave + scale_offset + 1
end

function Base.summary(ss::ScaleSpace{T}) where T
    num_octaves = ss.last_octave - ss.first_octave + 1
    println("ScaleSpace{$T} Summary:")
    println("  Input size: $(ss.input_size.width) × $(ss.input_size.height)")
    println("  Octaves: $(ss.first_octave) to $(ss.last_octave) ($num_octaves total)")
    println("  Scale subdivisions: $(ss.octave_first_subdivision) to $(ss.octave_last_subdivision)")
    println("  Octave resolution: $(ss.octave_resolution)")
    println("  Base sigma: $(ss.base_sigma)")
    println("  Camera sigma: $(ss.assumed_camera_sigma)")
    println("  Total levels: $(length(ss.levels))")
    println("  Sigma range: $(minimum(ss.levels.sigma)) to $(maximum(ss.levels.sigma))")
end

# Utility functions
function effective_sigma(target_sigma::Real, current_sigma::Real)
    target_sigma <= current_sigma && return 0.0
    return sqrt(target_sigma^2 - current_sigma^2)
end

function get_base_image_for_level(input_image::AbstractMatrix, level::ScaleLevel)
    level.octave == 0 && return input_image
    scale_factor = 2.0^(-level.octave)
    h, w = size(input_image)
    new_size = (max(1, round(Int, h * scale_factor)), max(1, round(Int, w * scale_factor)))
    return imresize(input_image, new_size)
end

function downsample_to_octave(img::AbstractMatrix, from_octave::Int, to_octave::Int)
    @assert to_octave >= from_octave
    octave_diff = to_octave - from_octave
    octave_diff == 0 && return img
    scale_factor = 2.0^(-octave_diff)
    h, w = size(img)
    new_size = (max(1, round(Int, h * scale_factor)), max(1, round(Int, w * scale_factor)))
    return imresize(img, new_size)
end

# Filter functions

"""
    gaussian_filter(data::AbstractMatrix, sigma::Float64) -> filtered::AbstractMatrix

Apply Gaussian filtering with the specified sigma.
This is the default filter for scale space construction.
"""
gaussian_filter(data::AbstractMatrix, sigma::Float64) =
    imfilter(data, Kernel.gaussian(sigma), "reflect")

"""
    hessian_filter(Ixx::AbstractMatrix, Iyy::AbstractMatrix, Ixy::AbstractMatrix, image::AbstractMatrix)

Compute Hessian components using central differences.
"""
function hessian_filter(Ixx::AbstractMatrix, Iyy::AbstractMatrix, Ixy::AbstractMatrix, image::AbstractMatrix)
    # Use full 2D kernels for second derivatives
    # d²/dx² kernel (1D in x-direction, centered in y)
    kernel_xx = centered([0 0 0; 1 -2 1; 0 0 0])
    # d²/dy² kernel (1D in y-direction, centered in x)
    kernel_yy = centered([0 1 0; 0 -2 0; 0 1 0])
    # d²/dxdy kernel (mixed partial derivative)
    kernel_xy = centered([0.25 0 -0.25; 0 0 0; -0.25 0 0.25])

    # Apply filters
    Ixx .= imfilter(image, kernel_xx)
    Iyy .= imfilter(image, kernel_yy)
    Ixy .= imfilter(image, kernel_xy)
    return nothing
end

# Scale space population - Functor interface

"""
    (ss::ScaleSpace{<:AbstractMatrix})(input_image::AbstractMatrix, filter_fn::Function)

Populate the scale space with an image using the provided filter function.

# Arguments
- `input_image`: The input image to process
- `filter_fn`: Filter function with signature `(data, sigma) -> filtered_data`

# Example
```julia
ss = ScaleSpace(width=128, height=128)
ss(image, gaussian_filter)  # Use Gaussian filtering
```
"""
function (ss::ScaleSpace{<:AbstractMatrix})(input_image::AbstractMatrix, filter_fn::Function)
    input_h, input_w = size(input_image)
    @assert input_h == ss.input_size.height && input_w == ss.input_size.width

    # Process first octave
    _populate_octave_from_image!(ss, input_image, ss.first_octave, filter_fn)

    # Process remaining octaves
    for o in (ss.first_octave + 1):ss.last_octave
        _populate_octave_from_previous!(ss, o, filter_fn)
    end

    return ss
end

"""
    (hess_ss::ScaleSpace)(smooth_ss::ScaleSpace{<:AbstractMatrix}, filter_fn::Function)

Populate a Hessian scale space from a smoothed scale space using the provided filter.

# Example
```julia
smooth_ss = ScaleSpace(width=128, height=128)
smooth_ss(image, gaussian_filter)
hess_ss = HessianScaleSpace(smooth_ss)
hess_ss(smooth_ss, hessian_filter)
```
"""
function (hess_ss::ScaleSpace)(smooth_ss::ScaleSpace{<:AbstractMatrix}, filter_fn::Function)
    for i in eachindex(hess_ss.levels)
        smooth_level = smooth_ss.levels.data[i]
        hess_level = hess_ss.levels.data[i]
        filter_fn(hess_level.Ixx, hess_level.Iyy, hess_level.Ixy, smooth_level)
    end
    return hess_ss
end

# Internal population functions

"""
Initialize and fill an octave starting from the input image.
"""
function _populate_octave_from_image!(ss::ScaleSpace{<:AbstractMatrix},
                                      input_image::AbstractMatrix, o::Int, filter_fn::Function)
    # Get first level of this octave
    first_level = get_scale_level(ss, o, ss.octave_first_subdivision)

    # Downsample/upsample input image to this octave's resolution
    if o >= 0
        # Downsample by simple decimation (sample every 2^o pixels)
        step = 2^o
        first_level.data .= input_image[1:step:end, 1:step:end]
    else
        # Upsample (not typically used, but VLFeat supports it)
        scale_factor = 2.0^(-o)
        h, w = size(input_image)
        new_size = (round(Int, h * scale_factor), round(Int, w * scale_factor))
        first_level.data .= imresize(input_image, new_size)
    end

    # Apply smoothing to reach target sigma
    target_sigma = first_level.sigma
    image_sigma = ss.assumed_camera_sigma

    if target_sigma > image_sigma
        delta_sigma = sqrt(target_sigma^2 - image_sigma^2)
        # Sigma is in absolute coordinates, but we're working in octave coordinates
        # so divide by step
        kernel_sigma = delta_sigma / first_level.step
        first_level.data .= filter_fn(first_level.data, kernel_sigma)
    end

    # Fill rest of octave by incremental smoothing
    _fill_octave!(ss, o, filter_fn)
end

"""
Initialize and fill an octave from the previous octave.
"""
function _populate_octave_from_previous!(ss::ScaleSpace{<:AbstractMatrix}, o::Int, filter_fn::Function)
    # Pick the level from previous octave that's closest to octaveResolution steps up
    # This is min(octaveFirstSubdivision + octaveResolution, octaveLastSubdivision)
    prev_scale_idx = min(ss.octave_first_subdivision + ss.octave_resolution,
                        ss.octave_last_subdivision)

    prev_level = get_scale_level(ss, o - 1, prev_scale_idx)
    curr_level = get_scale_level(ss, o, ss.octave_first_subdivision)

    # Downsample by simple decimation (take every other pixel)
    curr_level.data .= prev_level.data[1:2:end, 1:2:end]

    # Apply additional smoothing if needed
    target_sigma = curr_level.sigma
    prev_sigma = prev_level.sigma

    if target_sigma > prev_sigma
        delta_sigma = sqrt(target_sigma^2 - prev_sigma^2)
        kernel_sigma = delta_sigma / curr_level.step
        curr_level.data .= filter_fn(curr_level.data, kernel_sigma)
    end

    # Fill rest of octave
    _fill_octave!(ss, o, filter_fn)
end

"""
Fill an octave by incrementally smoothing from the first subdivision.
"""
function _fill_octave!(ss::ScaleSpace{<:AbstractMatrix}, o::Int, filter_fn::Function)
    for s in (ss.octave_first_subdivision + 1):ss.octave_last_subdivision
        curr_level = get_scale_level(ss, o, s)
        prev_level = get_scale_level(ss, o, s - 1)

        target_sigma = curr_level.sigma
        prev_sigma = prev_level.sigma
        delta_sigma = sqrt(target_sigma^2 - prev_sigma^2)

        # Sigma in octave coordinates
        kernel_sigma = delta_sigma / curr_level.step
        curr_level.data .= filter_fn(prev_level.data, kernel_sigma)
    end
end

"""
    laplacian_filter(lap_data::AbstractMatrix, hess_level::NamedTuple)

Compute Laplacian from Hessian components by summing Ixx + Iyy.
"""
function laplacian_filter(lap_data::AbstractMatrix, hess_level::NamedTuple)
    lap_data .= hess_level.Ixx .+ hess_level.Iyy
    return nothing
end

"""
    (lap_ss::ScaleSpace{<:AbstractMatrix})(hess_ss::ScaleSpace, filter_fn::Function)

Populate a Laplacian scale space from a Hessian scale space.

# Example
```julia
smooth_ss = ScaleSpace(width=128, height=128)
smooth_ss(image, gaussian_filter)
hess_ss = HessianScaleSpace(smooth_ss)
hess_ss(smooth_ss, hessian_filter)
lap_ss = LaplacianScaleSpace(smooth_ss)
lap_ss(hess_ss, laplacian_filter)
```
"""
function (lap_ss::ScaleSpace{<:AbstractMatrix})(hess_ss::ScaleSpace, filter_fn::Function)
    for i in eachindex(lap_ss.levels)
        hess_level = hess_ss.levels.data[i]
        lap_level = lap_ss.levels.data[i]
        filter_fn(lap_level, hess_level)
    end
    return lap_ss
end

# Image saving utilities

"""
    save_images(ss::ScaleSpace{<:AbstractMatrix}, output_dir::String; prefix="level")

Save all levels of a scale space as grayscale images for debugging and visualization.

# Arguments
- `ss`: The scale space to save
- `output_dir`: Directory to save images to (will be created if it doesn't exist)
- `prefix`: Filename prefix (default: "level")

# Example
```julia
ss = ScaleSpace(size=Size2(128, 128))
ss(image)
save_images(ss, "scalespace_debug")
```

Each image is saved as `<prefix>_o<octave>_s<scale>.png` with filenames indicating
octave and scale indices. The images are saved as 8-bit grayscale with automatic
normalization to [0, 1] range.
"""
function save_images(ss::ScaleSpace{<:AbstractMatrix}, output_dir::String; prefix::String="level")
    # Create output directory if it doesn't exist
    mkpath(output_dir)

    saved_count = 0
    skipped_count = 0

    # Save each level
    for level in ss.levels
        data = level.data

        # Skip uninitialized levels with NaN values
        if any(isnan, data)
            println("Skipped: o=$(level.octave), s=$(level.scale) (contains NaN values - uninitialized)")
            skipped_count += 1
            continue
        end

        # Normalize to [0, 1] for display
        min_val, max_val = extrema(data)

        # Handle constant images
        if min_val == max_val
            normalized = fill(Gray(0.5f0), size(data))
        else
            normalized = Gray.((data .- min_val) ./ (max_val - min_val))
        end

        # Create filename
        filename = joinpath(output_dir, "$(prefix)_o$(level.octave)_s$(level.scale).png")

        # Save image
        save(filename, normalized)

        println("Saved: $filename (σ=$(round(level.sigma, digits=3)), size=$(level.size.width)×$(level.size.height), range=[$min_val, $max_val])")
        saved_count += 1
    end

    println("\n✓ Saved $saved_count/$((length(ss.levels))) scale space images to $output_dir")
    if skipped_count > 0
        println("⚠ Skipped $skipped_count uninitialized levels")
    end
end
