"""
Gaussian scale space with 3D cube storage for efficient Hessian and Laplacian computation.
"""

# All dependencies are imported in the main VisualGeometryCore.jl file

# =============================================================================
# CORE TYPES AND ABSTRACTIONS
# =============================================================================

"""
    ScaleLevel{T}

A single level in the scale space pyramid with associated metadata.
"""
struct ScaleLevel{T}
    data::T
    octave::Int
    subdivision::Int
    sigma::Float64
end

"""
    AbstractScaleSpace

Abstract base type for scale space structures.
All subtypes must have a `levels::StructArray{ScaleLevel{T}}` field.
"""
abstract type AbstractScaleSpace end

# =============================================================================
# 3D CUBE ARCHITECTURE TYPES
# =============================================================================

"""
    ScaleOctave

Single octave with 3D Gaussian cube and metadata.
"""
struct ScaleOctave
    cube::Array{Gray{Float32},3}    # 3D Gaussian cube (H^o, W^o, Lg)
    index::Int                       # Octave index (-1, 0, 1, 2, ...)
    subdivisions::UnitRange{Int}     # Subdivision range (e.g., 0:(S+2))
    sigmas::Vector{Float64}          # Absolute σ per level (length Lg)
    step::Float64                    # Pixel stride (2^index, can be fractional for upsampled octaves)
end

"""
    ScaleLevelView

Type alias for SubArray-based scale levels that are views into 3D cubes.
"""
const ScaleLevelView = ScaleLevel{SubArray{Gray{Float32},2,Array{Gray{Float32},3}}}

"""
    ScaleSpace

Gaussian scale space with 3D cube storage. Levels are SubArray views into octave cubes.
Supports `ss[octave, subdivision]` and `ss[octave]` indexing.
"""
struct ScaleSpace <: AbstractScaleSpace
    base_sigma::Float64
    camera_psf::Float64
    levels::StructArray{ScaleLevelView}     # Levels with SubArray data
    input_size::Size2{Int}
    kernel_func::Function
    border_policy::String
    octave_resolution::Int
    octaves::Vector{ScaleOctave}            # Direct storage of ScaleOctave objects
end

# =============================================================================
# 3D CUBE VIEW CREATION
# =============================================================================

"""
    ScaleSpace constructor with 3D cube storage.
"""
function ScaleSpace(first_octave::Int, last_octave::Int, octave_resolution::Int,
                     base_sigma::Float64, camera_psf::Float64,
                     input_size::Size2{Int},
                     kernel_func::Function=sigma -> Kernel.gaussian((sigma, sigma), (2*ceil(Int, 3*sigma) + 1, 2*ceil(Int, 3*sigma) + 1)),
                     border_policy::String="replicate";
                     first_subdivision::Int=0,
                     last_subdivision::Int=octave_resolution + 2)  # S+3 levels for Gaussian

    octave_range = first_octave:last_octave
    subdivision_range = first_subdivision:last_subdivision

    # Create combinations in octave-major order (same as original)
    octaves = repeat(collect(octave_range), inner=length(subdivision_range))
    subdivisions = repeat(collect(subdivision_range), outer=length(octave_range))
    sigmas = base_sigma .* 2.0 .^ (octaves .+ subdivisions ./ octave_resolution)

    # Create octave objects with 3D cubes
    octave_objects = ScaleOctave[]
    num_levels = length(subdivision_range)
    
    for o in octave_range
        if o == first(octave_range)
            # First octave size depends on whether it's upsampled
            if o < 0
                # Upsampled octave: 2x input size for each negative octave
                scale_factor = 2^(-o)
                H_o = input_size.height * scale_factor
                W_o = input_size.width * scale_factor
            else
                # Normal or downsampled: matches input or downsampled from it
                H_o = input_size.height
                W_o = input_size.width
            end
        else
            # Subsequent octaves are based on decimation of previous octave
            prev_octave_idx = o - first(octave_range)
            prev_octave = octave_objects[prev_octave_idx]
            H_o = length(1:2:size(prev_octave.cube, 1))
            W_o = length(1:2:size(prev_octave.cube, 2))
        end
        
        cube = Array{Gray{Float32},3}(undef, H_o, W_o, num_levels)
        octave_levels = [s for s in subdivision_range]
        octave_sigmas = base_sigma .* 2.0 .^ (o .+ octave_levels ./ octave_resolution)
        step = 2.0^o
        
        octave_obj = ScaleOctave(cube, o, subdivision_range, octave_sigmas, step)
        push!(octave_objects, octave_obj)
    end

    # Create SubArray views into octave cubes
    data_arrays = SubArray{Gray{Float32},2,Array{Gray{Float32},3}}[]
    
    for (i, (o, s)) in enumerate(zip(octaves, subdivisions))
        octave_obj = octave_objects[findfirst(oct -> oct.index == o, octave_objects)]
        first_sub = first(subdivision_range)
        slice_idx = (s - first_sub) + 1
        view_data = @view octave_obj.cube[:, :, slice_idx]
        push!(data_arrays, view_data)
    end

    # Create StructArray with SubArray-based levels
    levels = StructArray{ScaleLevelView}((
        data = data_arrays,
        octave = octaves,
        subdivision = subdivisions,
        sigma = sigmas
    ))

    return ScaleSpace(base_sigma, camera_psf, levels, input_size, 
                     kernel_func, border_policy, octave_resolution, octave_objects)
end

# ScaleSpace computed properties
first_octave(ss::ScaleSpace) = minimum(ss.levels.octave)
last_octave(ss::ScaleSpace) = maximum(ss.levels.octave)
octave_resolution(ss::ScaleSpace) = ss.octave_resolution

# =============================================================================
# BASE INTERFACE - ITERATION AND INDEXING
# =============================================================================

# Shared iteration interface for all AbstractScaleSpace types
Base.iterate(ss::AbstractScaleSpace) = iterate(ss.levels)
Base.iterate(ss::AbstractScaleSpace, state) = iterate(ss.levels, state)
Base.length(ss::AbstractScaleSpace) = length(ss.levels)
Base.eltype(::Type{ScaleSpace}) = ScaleLevelView
Base.firstindex(ss::AbstractScaleSpace) = firstindex(ss.levels)
Base.lastindex(ss::AbstractScaleSpace) = lastindex(ss.levels)

# Linear indexing support for all AbstractScaleSpace types
Base.getindex(ss::AbstractScaleSpace, i::Int) = ss.levels[i]
Base.setindex!(ss::AbstractScaleSpace, value, i::Int) = (ss.levels[i] = value)



# =============================================================================
# SCALESPACE INDEXING - OCTAVE AND LEVEL ACCESS
# =============================================================================

function Base.size(ss::ScaleSpace)
    num_octaves = last_octave(ss) - first_octave(ss) + 1
    num_subdivisions = octave_resolution(ss) + 3  # S+3 levels
    return (num_octaves, num_subdivisions)
end

function Base.axes(ss::ScaleSpace)
    octave_range = first_octave(ss):last_octave(ss)
    subdivision_range = minimum(ss.levels.subdivision):maximum(ss.levels.subdivision)
    return (octave_range, subdivision_range)
end

"""
    ss[octave] -> ScaleOctave

Access octave by number, returning ScaleOctave with 3D cube and metadata.
"""
function Base.getindex(ss::ScaleSpace, octave::Int)
    octave_range, subdivision_range = axes(ss)
    @boundscheck octave in octave_range || throw(BoundsError(ss, (octave,)))
    
    # Find the octave object in the vector
    first_octave_num = first(octave_range)
    octave_idx = octave - first_octave_num + 1
    
    return ss.octaves[octave_idx]
end

"""
    ss[octave, subdivision] -> ScaleLevel with SubArray data

Access level by octave and subdivision, returning ScaleLevel with view into 3D cube.
"""
function Base.getindex(ss::ScaleSpace, octave::Int, subdivision::Int)
    octave_range, subdivision_range = axes(ss)
    @boundscheck begin
        octave in octave_range || throw(BoundsError(ss, (octave, subdivision)))
        subdivision in subdivision_range || throw(BoundsError(ss, (octave, subdivision)))
    end
    
    subdivisions_per_octave = length(subdivision_range)
    octave_offset = octave - first(octave_range)
    subdivision_offset = subdivision - first(subdivision_range)
    
    linear_idx = octave_offset * subdivisions_per_octave + subdivision_offset + 1
    return ss.levels[linear_idx]
end

Base.getindex(ss::ScaleSpace, I::CartesianIndex{2}) = ss[I[1], I[2]]

function Base.setindex!(ss::ScaleSpace, value, octave::Int, subdivision::Int)
    octave_range, subdivision_range = axes(ss)
    @boundscheck begin
        octave in octave_range || throw(BoundsError(ss, (octave, subdivision)))
        subdivision in subdivision_range || throw(BoundsError(ss, (octave, subdivision)))
    end
    
    subdivisions_per_octave = length(subdivision_range)
    octave_offset = octave - first(octave_range)
    subdivision_offset = subdivision - first(subdivision_range)
    
    linear_idx = octave_offset * subdivisions_per_octave + subdivision_offset + 1
    ss.levels[linear_idx] = value
    return value 
end

Base.setindex!(ss::ScaleSpace, value, I::CartesianIndex{2}) = (ss[I[1], I[2]] = value)

# Range indexing support
Base.getindex(ss::ScaleSpace, octave::Int, subdivisions::AbstractRange) = [ss[octave, s] for s in subdivisions]

# =============================================================================
# SCALESPACE CONVENIENCE CONSTRUCTORS
# =============================================================================

"""
    ScaleSpace(size::Size2{Int}; kwargs...)

Create a 3D cube-based Gaussian scale space with specified size and parameters.
"""
function ScaleSpace(size::Size2{Int};
                     first_octave::Int=0, octave_resolution::Int=3, base_sigma::Float64=1.6,
                     camera_psf::Float64=0.5,
                     kernel_func::Function=sigma -> Kernel.gaussian((sigma, sigma), (2*ceil(Int, 3*sigma) + 1, 2*ceil(Int, 3*sigma) + 1)),
                     border_policy::String="replicate")
    min_dim = min(size.width, size.height)
    last_octave = max(floor(Int, log2(min_dim)) - 3, 0)

    # VLFeat applies base_sigma scaling: baseScale = 1.6 * 2^(1/octaveResolution)
    adjusted_base_sigma = base_sigma * 2.0^(1.0 / octave_resolution)

    return ScaleSpace(first_octave, last_octave, octave_resolution,
                     adjusted_base_sigma, camera_psf, size, kernel_func, border_policy)
end

"""
    ScaleSpace(image::Matrix{Gray{T}}; kwargs...) where T<:Union{FixedPoint, AbstractFloat}

Create and populate a 3D cube-based Gaussian scale space from a grayscale image.
"""
function ScaleSpace(image::Matrix{Gray{T}};
                     first_octave::Int=0, octave_resolution::Int=3, base_sigma::Float64=1.6,
                     camera_psf::Float64=0.5, border_policy::String="replicate",
                     first_subdivision::Int=0, last_subdivision::Int=-1) where T<:Union{FixedPoint, AbstractFloat}
    img_size = Size2(image)

    # Default last_subdivision to octave_resolution + 2 if not specified
    if last_subdivision < 0
        last_subdivision = octave_resolution + 2
    end

    # Compute last octave
    min_dim = min(img_size.width, img_size.height)
    last_octave = max(floor(Int, log2(min_dim)) - 3, 0)

    # VLFeat applies base_sigma scaling
    adjusted_base_sigma = base_sigma * 2.0^(1.0 / octave_resolution)

    # Use VLFeat-compatible kernel sizing
    kernel_func = sigma -> Kernel.gaussian((sigma, sigma), (2*ceil(Int, 3*sigma) + 1, 2*ceil(Int, 3*sigma) + 1))

    # Create scale space with subdivision range
    ss = ScaleSpace(first_octave, last_octave, octave_resolution,
                   adjusted_base_sigma, camera_psf, img_size, kernel_func, border_policy;
                   first_subdivision=first_subdivision, last_subdivision=last_subdivision)

    # Populate the scale space with the image data
    ss(image)
    return ss
end



# =============================================================================
# SCALESPACE COMPUTATION - FUNCTOR INTERFACE
# =============================================================================

"""
    (ss::ScaleSpace)(input::Matrix{Gray{T}}) where T<:Union{FixedPoint, AbstractFloat}

Populate 3D cube scale space pyramid from input image.
Uses 3D cube storage with SubArray views for efficient memory layout.
"""
function (ss::ScaleSpace)(input::Matrix{Gray{T}}) where T<:Union{FixedPoint, AbstractFloat} 
    # Verify input dimensions
    input_h, input_w = size(input)
    @assert input_h == ss.input_size.height && input_w == ss.input_size.width

    # Inner function for incremental smoothing computation (same as original)
    function apply_incremental_smoothing!(dst_data, src_data, target_sigma, prev_sigma, level)
        if target_sigma > prev_sigma
            delta_sigma = sqrt(target_sigma^2 - prev_sigma^2)
            kernel_sigma = delta_sigma / 2.0^level.octave
            # Extract numeric values for filtering
            numeric_src = channelview(src_data)
            numeric_dst = channelview(dst_data)
            filter_kernel = ss.kernel_func(kernel_sigma)
            # Use mutating imfilter! for efficiency - works with SubArrays
            imfilter!(numeric_dst, numeric_src, filter_kernel, ss.border_policy)
        end
    end

    # Populate first octave from input image (same logic as original)
    o = first_octave(ss)
    _, subdivision_range = axes(ss)
    first_subdivision = first(subdivision_range)
    first_level = ss[o, first_subdivision]

    # Downsample/upsample input image to this octave's resolution
    if o >= 0
        # Downsample by decimation for positive octaves
        step = 2^o
        sampled_input = input[1:step:end, 1:step:end]
        first_level.data .= Gray{Float32}.(sampled_input)
    elseif o == -1
        # Use VLFeat-compatible 2x bilinear upsampling for octave -1 only
        vlfeat_upsample!(first_level.data, input)
    else
        error("Unsupported octave $o: only octave -1 upsampling (2x) is supported")
    end

    # Apply smoothing to reach target sigma
    apply_incremental_smoothing!(first_level.data, first_level.data,
                                first_level.sigma, ss.camera_psf, first_level)

    # Fill rest of first octave by incremental smoothing using broadcasting
    rest_subdivision_range = (first_subdivision+1):last(subdivision_range)
    prev_subdivision_range = first_subdivision:(last(subdivision_range)-1)
    # Broadcast the smoothing operation across all subdivisions in this octave
    apply_incremental_smoothing!.(getfield.(ss[o, rest_subdivision_range], :data),
                                 getfield.(ss[o, prev_subdivision_range], :data),
                                 getfield.(ss[o, rest_subdivision_range], :sigma),
                                 getfield.(ss[o, prev_subdivision_range], :sigma),
                                 ss[o, rest_subdivision_range])

    # Populate remaining octaves from previous octave (same logic as original)
    for o in (first_octave(ss) + 1):last_octave(ss)
        # Pick the level from previous octave to downsample
        # VLFeat uses min(octaveResolution + octaveFirstSubdivision, octaveLastSubdivision)
        prev_subdivision_idx = min(octave_resolution(ss) + first_subdivision, last(subdivision_range))

        # Downsample by simple decimation
        # VLFeat C: x=0,2,4,... (0-indexed) → Julia: 1,3,5,... (1-indexed) = 1:2:end ✓
        decimated = ss[o - 1, prev_subdivision_idx].data[1:2:end, 1:2:end]
        ss[o, first_subdivision].data .= decimated

        # Apply additional smoothing if needed
        apply_incremental_smoothing!(ss[o, first_subdivision].data, ss[o, first_subdivision].data,
                                    ss[o, first_subdivision].sigma, ss[o - 1, prev_subdivision_idx].sigma, ss[o, first_subdivision])

        # Fill rest of octave using broadcasting
        rest_subdivision_range = (first_subdivision+1):last(subdivision_range)
        prev_subdivision_range = first_subdivision:(last(subdivision_range)-1)
        # Broadcast the smoothing operation across all subdivisions in this octave
        apply_incremental_smoothing!.(getfield.(ss[o, rest_subdivision_range], :data),
                                     getfield.(ss[o, prev_subdivision_range], :data),
                                     getfield.(ss[o, rest_subdivision_range], :sigma),
                                     getfield.(ss[o, prev_subdivision_range], :sigma),
                                     ss[o, rest_subdivision_range])
    end

    return ss
end

# =============================================================================
# BASE METHODS
# =============================================================================

function Base.summary(ss::ScaleSpace)
    num_octaves = last_octave(ss) - first_octave(ss) + 1
    kernel_info = "Kernel: $(typeof(ss.kernel_func))"
    num_octaves_stored = length(ss.octaves)
    
    println("ScaleSpace Summary:" *
          "\n  Input size: $(ss.input_size.width) × $(ss.input_size.height)" *
          "\n  Octaves: $(first_octave(ss)) to $(last_octave(ss)) ($num_octaves total)" *
          "\n  Subdivisions: $(minimum(ss.levels.subdivision)) to $(maximum(ss.levels.subdivision))" *
          "\n  Octave resolution: $(octave_resolution(ss))" *
          "\n  Base sigma: $(ss.base_sigma)" *
          "\n  Camera PSF: $(ss.camera_psf)" *
          "\n  $kernel_info" *
          "\n  Border policy: $(ss.border_policy)" *
          "\n  Total levels: $(length(ss.levels))" *
          "\n  ScaleOctave objects: $num_octaves_stored" *
          "\n  Sigma range: $(minimum(ss.levels.sigma)) to $(maximum(ss.levels.sigma))")
end